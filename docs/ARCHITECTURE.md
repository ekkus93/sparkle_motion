# Sparkle Motion Architecture

This document captures the current, ADK-first architecture for Sparkle Motion
as of December 2025. It focuses on the shipped runtime instead of historical
scaffolding so that builders, operators, and reviewers share a single source of
truth.

## High-level system

Sparkle Motion is split across two ADK agents and a catalog of FunctionTools:

- `script_agent` plans entire stories by producing validated `MoviePlan`
  artifacts.
- `production_agent` orchestrates the full render pipeline (images → video →
  dialogue → lipsync → assemble → finalize) and publishes artifacts plus
  `StepExecutionRecord` history.
- FunctionTools under `function_tools/` perform the heavy GPU/IO work
  (Diffusers, WAN, TTS, Wav2Lip, ffmpeg) behind small HTTP adapters.

Everything else in the repo serves those flows: notebook dashboards, CLI
helpers, schema registries, and observability glue.

## Runtime inventory (authoritative)

- **ADK agents (runtime):** 2 (`script_agent`, `production_agent`).
- **FunctionTools:** `images_sdxl`, `videos_wan`, `tts_chatterbox`,
  `lipsync_wav2lip`, `assemble_ffmpeg`, plus the helper wrapper that exposes
  `production_agent` as `/invoke`/`/status`/`/artifacts`/`/control/*`.
- **Stage adapters:** Python modules under `src/sparkle_motion/*_stage.py`
  (images, videos, tts, assemble) that normalize retries, telemetry, and
  schema conversions before calling FunctionTools.

## Agents vs FunctionTools

| Component | Type | Responsibility |
| --- | --- | --- |
| `script_agent` | ADK agent | Prompt-to-plan generation, schema validation, artifact publication. |
| `production_agent` | ADK WorkflowAgent | Executes MoviePlans, enforces modes (`dry` vs `run`), records run registry history, packages final deliverables. |
| Stage adapters (`images_stage`, `videos_stage`, `tts_stage`, etc.) | Python modules | Rate limiting, dedupe, retries, policy checks, payload normalization. |
| FunctionTools (`images_sdxl`, `videos_wan`, `tts_chatterbox`, `lipsync_wav2lip`, `assemble_ffmpeg`) | Tool runtimes | Heavy rendering, synthesis, assembly, filesystem shims. |

Stage adapters and FunctionTools intentionally avoid the `_agent` suffix to
keep naming clear: only the two orchestration agents keep it.

## Execution flow

1. **Plan intake** – Notebook/CLI calls `script_agent` to generate a
   `MoviePlan`, or loads a cached artifact from `artifacts/runs/...`.
2. **Production start** – `production_agent` consumes the plan via `/invoke`
   with `mode="dry"` (simulation) or `mode="run"` (full execution).
3. **Stage orchestration** – Per-step adapters drive FunctionTools, emit
   `StepExecutionRecord` entries, and propagate retries/pauses/stops through
   `/control/*` endpoints.
4. **Assembly** – `assemble_ffmpeg` stitches the rendered shots plus dialogue
   timeline into a final MP4 and metadata JSON.
5. **Finalize** – The WorkflowAgent publishes the canonical `video_final`
   manifest row and exposes it through `/artifacts` for notebook preview and
   download helpers.

Resuming a run always targets the `finalize` stage. If the manifest is missing
(or downstream tooling needs a refresh), operators rerun via
`resume_from="finalize"`.

## Run registry & persistence

- `RunRegistry` (SQLite) tracks every run, the associated plan, stage
  transitions, control actions, and manifest URIs. `/status` and `/artifacts`
  read directly from this registry.
- Artifacts are persisted under `artifacts/runs/<run_id>/...` along with the
  manifest JSON that mirrors ArtifactService responses.
- Notebooks include helper cells for plan inspection, production monitoring,
  artifacts browsing, and final deliverable previews. These helpers call the
  same FastAPI endpoints as external operators.

## Schemas & contracts

`configs/schema_artifacts.yaml` plus `sparkle_motion.schema_registry` provide
paths to every JSON schema required at runtime: MoviePlan, AssetRefs,
StageEvent, Checkpoint, StageManifest, and RunContext docs. Tool adapters load
those schemas for validation so that CLI, notebooks, and backend services stay
in sync.

## Observability & controls

- Structured logs stream through `sparkle_motion.observability` helpers and
  land beside the artifacts.
- `/control/pause`, `/control/resume`, and `/control/stop` allow live
  orchestration tweaks; the notebook control panel exposes those buttons and
  shows the acknowledgements streamed from `/status`.
- `/ready` and `/status` endpoints support health probes and dashboards.

## Delivery expectations

Finalize publishes everything required for distribution:

- `video_final` StageManifest row with checksums, duration, frame rate,
  resolution, and download pointers (filesystem shim or ArtifactService).
- Inline preview helpers in `notebooks/sparkle_motion.ipynb` that embed the
  MP4 and surface Drive/ADK fallbacks when needed.
- Resume guidance that always points back to `resume_from="finalize"`.

This document intentionally omits deprecated stages and review scaffolding so
new contributors only learn the supported flow. When additional validation or
inspection stages are reinstated, they should be layered on top of the
orchestration points described here without renaming the core agents.
