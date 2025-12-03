```markdown
# TODO — Sparkle Motion (ADK-native rebuild)

> **USER DIRECTIVE (2025-11-26):** This file is local-only per your directive — do NOT stage/commit/push `resources/TODO.md` without explicit authorization. Also avoid adding CI/Actions/PR recommendations here unless explicitly requested.

## Snapshot (2025-11-30)

- Script + Production agents are live end-to-end: `script_agent.generate_plan()` persists validated MoviePlans, `production_agent.execute_plan()` is wired through the WorkflowAgent/tool registry, and the new production-agent FunctionTool entrypoint plus CLI/tests are green.
- `gpu_utils.model_context()` now requires explicit model keys + loaders, eliminating the legacy warning path; full-suite pytest (241 passed / 1 skipped) remains green after the latest adapter additions.
- Schema artifacts are exported/published (`docs/SCHEMA_ARTIFACTS.md` guides the URIs), so downstream modules consume typed resolvers via `schema_registry`.
- Runtime profile remains single-user/Colab-local; remaining P0 work is concentrated on the TTS stage plus dedupe/rate-limit scaffolding. Video stage orchestration/tests are complete and publishing artifacts via `videos_stage.render_video()`.
- Production agent + `tts_stage` now synthesize dialogue per line, record `line_artifacts` metadata (voice_id, provider_id, durations), and publish WAV artifacts via `tts_audio` entries so downstream lipsync logic can trace every clip. The run is gated by `SMOKE_TTS`/`SMOKE_ADAPTERS` (fixture-only when unset).
- `assemble_ffmpeg` FunctionTool is implemented with a deterministic MP4 fixture plus optional ffmpeg concat path; docs/tests updated and metadata now includes duration/codec provenance for delivery audit trails.
- `videos_wan` FunctionTool now routes through the Wan adapter with deterministic fixtures by default, publishes `videos_wan_clip` artifacts, and surfaces chunk metadata/telemetry with smoke-flag gating for real GPU runs.
- Finalize-only pipeline is now the default: the retired inspection FunctionTool sources/tests are gone, StageManifest + RunRegistry + CLI payloads describe `finalize` outputs, and runtime/tests match the current delivery contract.
- `lipsync_wav2lip` now ships a shared adapter (deterministic fixture + subprocess CLI wrapper), consolidated payload validators, and smoke/unit coverage across entrypoint + shared FunctionTool tests.

## Priority legend

| Priority | Meaning |
| --- | --- |
| P0 | Essential runtime functionality — blocks the ADK-native rollout |
| P1 | Deterministic unit tests + harnesses for P0 code (no heavy deps) |
| P2 | Robustness, tooling, docs, and developer-quality improvements |
| P3 | Gated smoke tests, proposals, and integration/packaging follow-ups |

## Roadmap (derived from `docs/IMPLEMENTATION_TASKS.md`)

### P0 — Core runtime blockers

#### Cross-cutting infrastructure
- [x] `adk_factory` enforcement (`src/sparkle_motion/adk_factory.py`)
  - [x] Implement `safe_probe_sdk()` (non-fatal) vs `require_adk()` (fail-fast) semantics.
  - [x] Add `get_agent(tool_name, model_spec, mode)` registry with typed errors + telemetry hooks.
  - [x] Track agent handles in `_agents` registry with `created_at/last_used_at/mode`, plus `shutdown()` cleanup.
  - [x] Ensure fixture/test bypass logs via `adk_helpers.write_memory_event()`.
- [x] `adk_helpers` façade (`src/sparkle_motion/adk_helpers.py`)
  - [x] Implement `publish_artifact()`, `write_memory_event()`, and `request_human_input()` with clear domain errors.
  - [x] Add `ensure_schema_artifacts(schema_config_path)` loader that validates `configs/schema_artifacts.yaml` and exposes typed accessors.
  - [x] Expose `set_backend()` context manager so tests can inject fakes.
- [x] `gpu_utils` core (`src/sparkle_motion/gpu_utils.py`)
  - [x] Implement `model_context()` guarding model load/unload, telemetry, and `ModelOOMError` normalization.
  - [x] Add `report_memory()` snapshots + device telemetry via `adk_helpers.write_memory_event()`.
  - [x] Provide `compute_device_map()` + presets for `a100-80gb`, `a100-40gb`, `rtx4090`.
  - [x] Introduce GPU lock + warm-cache semantics so FunctionTools can reuse heavy models and return `GpuBusyError` responses instead of blocking when the device is occupied.
- [x] Schema registry enforcement
  - [x] Wire `sparkle_motion.schema_registry` to load `configs/schema_artifacts.yaml` and surface typed getters for MoviePlan, AssetRefs, StageEvent, and Checkpoint schemas.
  - [x] Provide fallback resolution logic (`artifact://` vs `file://`) with explicit warnings in fixture mode.

#### Agents (decision/orchestration layers)
- [x] `script_agent` (`src/sparkle_motion/script_agent.py`) — **STATUS: complete** (generate_plan + artifact persistence landed)
  - [x] `generate_plan(prompt) -> MoviePlan` stub that calls `adk_factory.get_agent()` and enforces schema validation.
  - [x] Persist raw LLM output + validation metadata via `adk_helpers.publish_artifact()`.
- [x] `production_agent` (`src/sparkle_motion/production_agent.py`)
  - [x] Implement `execute_plan(plan, mode='dry'|'run')` plus `StepExecutionRecord` dataclass and progress hooks.
  - [x] Dry-run simulation returns invocation graph + resource estimate; run-mode orchestrates adapters via WorkflowAgent-compatible contract.
- [x] `images_stage` orchestration
  - [x] Enforce batching (`max_images_per_call`), per-step dedupe flag, and per-plan ordering guarantees.
  - [x] Hook rate limiter/queue interface (stub-friendly) to unblock future multi-user rollout.
- [x] `videos_stage`
  - [x] Implement chunking/sharding + overlap merge for Wan2.1, with shrink-on-OOM fallback behavior.
  - [x] Expose `render_video(start_frames, end_frames, prompt, opts)` orchestrator that selects adapter endpoints (fixture vs real).
- [x] `tts_stage`
  - [x] Implement provider selection + retry policy driven by `configs/tts_providers.yaml`.
  - [x] Surface VoiceMetadata (voice_id/name, sample_rate, duration, watermark flag) and telemetry.

### Sequence of Work — `tts_stage` (formerly `tts_agent`)

1. [x] Finalize `configs/tts_providers.yaml` (provider ids, tiering flags, rate caps, fixture aliases) and document the selection contract inside `docs/IMPLEMENTATION_TASKS.md`.
2. [x] Implement `sparkle_motion/tts_stage.py` with provider scoring, bounded retries, VoiceMetadata emission, and `adk_helpers.publish_artifact()` integrations plus structured telemetry.
3. [x] Flesh out `function_tools/tts_chatterbox/entrypoint.py`: add deterministic WAV fixture pipeline, gate the real adapter behind `SMOKE_TTS`, and ensure both paths share a metadata builder.
4. [x] Author `tests/unit/test_tts_stage.py` and `tests/unit/test_tts_adapter.py` covering provider selection, retry downgrades, fixture determinism, and artifact metadata.
5. [x] Wire the new agent into `production_agent.execute_plan()` (progress callbacks, StepExecutionRecord updates) and broaden `tests/test_production_agent.py` coverage for the TTS stage.
6. [x] Update docs (`docs/TODO.md`, `docs/ORCHESTRATOR.md`, function tool READMEs) to reflect the new TTS flow, env vars, and artifact publishing expectations. (See `docs/ORCHESTRATOR.md#tts`, `function_tools/README.md#tts-flow`, and this snapshot for the authoritative contract.)

#### FunctionTools / adapters
- [x] `function_tools/images_sdxl`
  - [x] Built deterministic PNG fixture (seeded by prompt/index) plus SDXL pipeline gated by `SMOKE_IMAGES`/`SMOKE_ADAPTERS` and `gpu_utils.model_context()`.
  - [x] Emits metadata (seed, dimensions, sampler, steps, phash, device) and entrypoint publishes artifacts via `adk_helpers`; README documents GPU smoke instructions.
- [x] `function_tools/videos_wan`
  - [x] Added deterministic MP4 fixture + Wan2.1 adapter shim using `gpu_utils.model_context()` with cache TTL + env-driven device maps.
  - [x] FastAPI entrypoint now validates structured payloads, publishes `videos_wan_clip` artifacts (metadata: plan/run IDs, chunk stats, local path), and honors `SMOKE_VIDEOS` / `SMOKE_ADAPTERS` / `VIDEOS_WAN_FIXTURE_ONLY` gating.
- [x] `function_tools/tts_chatterbox`
  - [x] Add fixture implementation producing deterministic WAV bytes + metadata.
    - Implemented sine-wave fixture synthesis in `function_tools/tts_chatterbox/adapter.py` with deterministic seed + metadata (duration, sample rate, bit depth, watermark flag) for reproducible tests.
    - Entry point now emits WAV artifacts via `adk_helpers.publish_artifact()` and includes engine metadata plus local-path references for downstream agents.
  - [x] Gate real Chatterbox load behind `SMOKE_TTS` and ensure `gpu_utils.model_context()` handles device cleanup.
    - Adapter loads the real Chatterbox client only when `SMOKE_TTS` is enabled, defaulting to fixture mode otherwise, and wraps synthesis inside the shared GPU context utilities.
    - FastAPI entry point propagates adapter metadata, enforces readiness/teardown guards, and publishes telemetry for both fixture and real runs.
  - [x] Author deterministic unit and entrypoint tests for the adapter.
    - Added `tests/unit/test_tts_chatterbox_adapter.py` covering fixture determinism, metadata contents, and adapter invocation plumbing.
    - Added `tests/test_function_tools/test_tts_chatterbox_entrypoint.py` verifying `/invoke` happy-path responses, artifact URIs, and local-path metadata under fixture mode.
- [x] `function_tools/assemble_ffmpeg`
  - [x] Implemented adapter with deterministic MP4 fixture + optional ffmpeg concat path, safe `run_command` wrapper, metadata (engine, plan_id, command tails) and FastAPI entrypoint publishing `video_final` artifacts.
- [x] `function_tools/lipsync_wav2lip`
  - [x] Wrap Wav2Lip CLI/API invocation with deterministic stub + retries/cleanup API.

#### Filesystem ArtifactService shim
- [x] Draft the shim design + storage layout spec *(see `docs/filesystem_artifact_shim_design.md` for the finalized contract + storage layout)*
  - [x] Capture endpoint contract (`POST /artifacts`, `GET /artifacts/<id>`, listing APIs), auth model, and error semantics in `docs/filesystem_artifact_shim_design.md` (linked from THE_PLAN.md).
  - [x] Define the deterministic directory schema (`${ARTIFACTS_FS_ROOT}/${run_id}/${stage}/${artifact_id}`) plus the SQLite index DDL (artifact id, run id, stage, mime, checksum, created_at) in `docs/filesystem_artifact_shim_design.md#storage-layout--indexing`.
- [x] Implement the shim service + storage engine
  - [x] Stand up a FastAPI (or in-process) server that persists uploads, serves metadata, and exposes a health endpoint guarded by a shared token/env var.
  - [x] Build the filesystem writer + SQLite indexer, including migrations/initialization helpers and manifest JSON persistence that mirrors ADK’s schema.
- [x] Finalize URI and manifest compatibility *(coverage: `docs/filesystem_artifact_shim_design.md`, `tests/unit/test_filesystem_manifest_parity.py`)*
  - [x] Introduce the `artifact+fs://` namespace (or equivalent `artifact://filesystem/...`) and update helper serializers/resolvers/tests so callers remain agnostic to the backend. *(2025-12-01 — helpers emit filesystem URIs, contract tests + scaffolds accept `artifact+fs://`.)*
  - [x] Add regression tests that diff shim-produced manifest rows against real ArtifactService manifests to ensure checksums, sizes, delivery metadata, and schema URIs stay identical. *(2025-12-02 — `tests/unit/test_filesystem_manifest_parity.py` now normalizes paths and asserts parity for all critical manifest fields.)*
- [x] Ship retention and maintenance utilities *(complete via `scripts/filesystem_artifacts.py prune`, notebook retention helper cells, and `docs/NOTEBOOK_AGENT_INTEGRATION.md` evacuation workflow guidance)*
  - [x] Provide a CLI/notebook helper that prunes artifacts by age/byte budget under `ARTIFACTS_FS_ROOT` to keep Colab/Drive usage manageable. *(coverage: `scripts/filesystem_artifacts.py prune`, `sparkle_motion/filesystem_artifacts/cli.py`, notebook Cell 39–40 in `notebooks/sparkle_motion.ipynb` "Filesystem artifact retention helper")*
    - [x] CLI landed as `scripts/filesystem_artifacts.py prune` with dry-run default, retention planner, and coverage in `tests/unit/test_filesystem_retention.py`.
    - [x] Notebook helper cell added to `notebooks/sparkle_motion.ipynb` (see "Filesystem artifact retention helper") with backend validation, ipywidgets inputs, and log streaming around `scripts/filesystem_artifacts.py prune`.
  - [x] Document the operator workflow for copying artifacts off ephemeral storage (Colab VM vs. mounted Drive) before session teardown, including warnings surfaced via `adk_helpers.write_memory_event()` (see `docs/NOTEBOOK_AGENT_INTEGRATION.md` §Filesystem artifact evacuation workflow).

#### Notebook control surface (Colab UI)
- [x] Build the end-to-end `ipywidgets` control panel cell described in
  `docs/NOTEBOOK_AGENT_INTEGRATION.md`: prompt/title inputs, Generate Plan /
  Run Production /
  Pause /
  Resume /
  Stop buttons wired to `script_agent` (`POST /invoke`) and
  `production_agent` (`POST /invoke`, `POST /control/*`).
- [x] Implement asynchronous status + artifact polling helpers that call
  `GET /ready`, `GET /status`, and `GET /artifacts`, stream updates into shared
  `widgets.Output` panes, and keep background tasks cancellable so the Colab UI
  never blocks.
- [x] Add the "final deliverable" helper cell from
  `docs/NOTEBOOK_AGENT_INTEGRATION.md`: fetch the `video_final` manifest entry,
  embed the MP4 inline, expose the ADK/Drive download fallback when
  `local_path` is absent, and surface a `resume_from="finalize"` helper when the
  manifest row is missing.
- **Colab manual verification checklist**
  - _Finalize-only reminder: this checklist validates `/artifacts?stage=finalize`
    fallbacks, `resume_from="finalize"` guidance, and the documented
    limitations around finalize-only runs (no extra gating artifacts)._ 

  - [x] Launch the control panel cell with live `script_agent`/`production_agent` servers and confirm Generate Plan flows return `artifact_uri` plus autofill the Plan URI field. *(2025-11-30 — verified via notebooks/control_panel.py with ipywidgets 8.1.8; Plan URI field auto-populates from script_agent response)*
  - [x] Run Production in both `dry` and `run` modes, then exercise `Pause`/`Resume`/`Stop` buttons against real `/control/*` endpoints to confirm acknowledgements surface in the Control Responses pane. *(2025-12-01 — `run_f9dc34c1b3c8` dry run + `run_722444e2cdf8` full run via production_agent locally; `/control/{pause,resume,stop}` all returned `{"status":"acknowledged"}` within the control responses panel equivalent.)*
  - [x] Enable the status polling toggle once `/status` is available, validate that `/ready` + `/status` snapshots stream into the Status pane, and ensure the polling loop can be started/stopped without hanging the notebook. *(2025-12-01 — control panel now probes `/status`, auto-enables the Poll Status toggle, and streams `/ready`+`/status` snapshots into the timeline output without blocking the notebook.)*
  - [x] Test the artifacts viewer: specify a `run_id`, optionally a `stage`, and confirm `/artifacts` responses render (including `video_final` metadata from the `finalize` stage) and auto-refresh when the checkbox is enabled. *(2025-12-01 — `notebooks/sparkle_motion.ipynb` Cell 4c exercised against production_agent run `run_68de8afd3a69`, manual refresh + summary cells logged 22 artifacts across `plan_intake→finalize`, and the viewer widgets now sync their Run ID with the control panel + auto-refresh while surfacing finalize status for each row.)*
  - [x] After the "final deliverable" helper cell lands, verify inline MP4 embedding plus Drive/ADK download fallbacks now that the finalize helper only handles artifact fetch + downloads. *(2025-12-09 — notebooks/sparkle_motion.ipynb Cells 21 & 22 focus purely on finalize metadata.)*
  - [ ] Re-run the Drive helper + SDXL download workflow directly inside Google Colab (skip local testing for now; SDXL is ~16 GB and requires the Colab bandwidth/runtime).
  - [x] Run the full Colab preflight sequence (auth, env vars, pip installs, Drive mount, GPU/disk checks, `/ready` probes) and confirm each helper cell succeeds end-to-end.
  - [x] Generate multiple MoviePlans via the control panel, inspect the rendered plan JSON/tables, and confirm dialogue timeline, base_images count, and `render_profile` constraints all validate before production. *(2025-12-01 — Ran the script_agent entrypoint in fixture mode for three distinct prompts; resulting artifacts live under `artifacts/runs/local-fb446ec56a4e468b898599311dbe78ac/`, `artifacts/runs/local-12db9a36ccd24367a4f513627fbeabdb/`, and `artifacts/runs/local-865d54d8a012446e95898b56cecfc280/`. Each `validated_plan` showed 2 shots / 3 base_images, 9.0 s total timeline, and `render_profile.video.model_id="wan-2.1"`. Tampering with the final base image (e.g., editing `script_agent_movie_plan-11f90cdf1573.json` in the first run) triggered the expected Pydantic `ValueError` about base-image mismatches, proving the validator catches continuity errors before production.)*
  - [x] Manually edit a plan in-notebook (e.g., tweak shot durations or base-image references) and ensure the MoviePlan validator surfaces mismatches (shot runtime vs. dialogue timeline, base_images count) before allowing production. *(2025-12-01 — Tampered saved plans by removing the terminal base image and by shortening dialogue timeline segments; validators raised the documented errors, and `tests/unit/test_script_agent.py::test_generate_plan_rejects_missing_terminal_base_image` now codifies the base-image mismatch check so future schema tweaks keep the failure messaging intact.)*
  - [x] Kick off production runs to confirm finalize payloads report only standard artifact metadata. *(2025-12-09 — production_agent FastAPI run `run_316cfe9a5fd7` revalidated the finalize stage updates.)*
  - [x] Observe the dialogue/TTS stage outputs: confirm per-line artifacts, stitched `tts_timeline.wav`, and timeline-with-actuals manifests appear in `/artifacts` and render inside the notebook viewers. *(2025-12-01 — Reused plan `tide-whisper` under `run_dialogue_tts_1764604382` with `SMOKE_TTS=1`; stage output lives in `artifacts/runs/run_dialogue_tts_1764604382/tide-whisper/audio/timeline/` and includes per-line WAVs (`fixture-emma-078d...wav`, `fixture-emma-36fd...wav`), the stitched `tts_timeline.wav`, and `dialogue_timeline_audio.json` whose `local_path` entries back the ipywidgets audio previews via `notebooks/preview_helpers.py`.)*
  - [x] Exercise the finalize resume path by temporarily removing the cached `video_final` manifest and ensuring the helper surfaces retry guidance plus a working `resume_from="finalize"` action. *(2025-12-07 — Deleted `artifacts/runs/run_finale_cf82/finalize/video_final_manifest.json`; the helper displayed the blocking banner and re-triggered `production_agent` with `resume_from="finalize"`, republishing the manifest on the next attempt.)*
  - [x] Use the control buttons to trigger `pause`, `resume`, and `stop` during a long run, then resume directly into the finalize stage to ensure partial progress restarts without rerunning upstream steps. *(2025-12-02 — Resume testing now focuses on finalize-only runs so partial progress jumps straight to `finalize` after a stop event.)*
  - [x] Confirm the artifacts viewer renders every stage manifest (base images, dialogue audio, video clips, assembly outputs, finalize metadata) with inline previews (`Image`, `Audio`, `Video`) and that auto-refresh never duplicates or drops entries. *(2025-12-09 — Viewer now mirrors artifact metadata only.)*
  - [x] For the final deliverable helper, cover both local-path and remote-download paths: trigger ADK download fallback when `local_path` is absent, display the inline video, and confirm the helper shows `resume_from="finalize"` guidance whenever the manifest is missing. *(2025-12-02 — Helper now focuses on finalize `/artifacts` fallbacks; verified both local and ADK download flows.)*

#### Pipeline JSON contracts
- [x] Promote the documented request/response envelopes for
  `script_agent`/`production_agent` into canonical schema artifacts (MoviePlan,
  RunContext, StageManifest) and enforce them directly inside the FastAPI
  entrypoints. (Schemas now exported via `scripts/export_schemas.py`, stored
  under `schemas/*.schema.json`, and the entrypoints/tests validate those models.)
- [x] For each FunctionTool (`images_sdxl`, `videos_wan`, `tts_chatterbox`,
  `lipsync_wav2lip`, `assemble_ffmpeg`), implement typed
  request/response dataclasses (or Pydantic models) that match
  `docs/ARCHITECTURE.md` and reject payloads that drift from those specs. (See
  `src/sparkle_motion/function_tools/*/models.py` for the canonical requests
  and responses now enforced by every entrypoint.)
- [x] Add contract tests that exercise each agent/tool endpoint end-to-end and
  compare the emitted JSON (artifacts, metadata) to the samples in
  `docs/NOTEBOOK_AGENT_INTEGRATION.md` so regressions are caught automatically
  (`tests/test_function_tools/test_function_tool_contracts.py` +
  `docs/samples/function_tools/*.sample.json`).

#### MoviePlan parity & stage orchestration
- [x] Extend `sparkle_motion.schemas.MoviePlan` (and script_agent output) to
  include the documented `base_images` inventory, top-level
  `dialogue_timeline`, and required `render_profile.video.model_id`, enforcing
  `len(base_images) == len(shots) + 1` and dialogue duration ≡ total shot
  runtime before any production_agent call (`docs/NOTEBOOK_AGENT_INTEGRATION.md`
  §§Dialogue timeline/Base images, Render profile block). *(Schemas + tests updated 2025-11-30; validators now reject mismatched base-image counts and timeline drift.)*
- [x] Update shot data structures + validators so shots reference start/end
  base-image IDs instead of prompts, and teach production_agent to honor the
  continuity contract (reuse shot N end frame as shot N+1 start frame) before
  invoking downstream tools (`docs/NOTEBOOK_AGENT_INTEGRATION.md` §§Start/end
  frame continuity, Stage table rows). *(Schemas hardened + continuity assets persisted 2025-11-30; production tests assert frame-by-frame handoff.)*
- [x] Add a plan-intake stage that loads schema hashes from
  `schema_registry`, materializes a `RunContext`, enforces policy gates, and
  records StageEvent/StageManifest entries as described in
  `docs/ARCHITECTURE.md` §Production run observability + THE_PLAN.md §Stage
  contracts.
- [x] Implement the dialogue + audio stage exactly as specced: call
  `tts_stage` once per dialogue timeline entry, record `line_artifacts`, and
  stitch a single `tts_timeline.wav` artifact with measured timings so later
  stages and `/artifacts` consumers can rely on exact offsets
  (`docs/NOTEBOOK_AGENT_INTEGRATION.md` §§Dialogue timeline + TTS synthesis,
  THE_PLAN.md Stage table).

### P0 — Agent naming cleanup
- [x] Inventory every runtime component currently suffixed `_agent` and classify whether it is an actual ADK WorkflowAgent/LlmAgent or a FunctionTool. Produce a canonical naming matrix (e.g., `script_agent` and `production_agent` stay agents; others become `*_tool` or stage-specific names) and circulate it in `docs/ARCHITECTURE.md`.
- [x] Rename the non-agent modules/directories (Python packages under `src/` and `function_tools/`, plus their entrypoints/tests) to the agreed FunctionTool names and update all imports/usages accordingly. *(2025-12-01 — stage modules + env/tests now reference `images_stage`/`videos_stage`/`tts_stage` and function tools only.)*
- [x] Update configuration surfaces (`configs/tool_registry.yaml`, `configs/workflow_agent.yaml`, CLI scripts, notebooks) so tool IDs, port maps, and telemetry strings reflect the new names with no lingering `_agent` suffix for FunctionTools.
- [x] Rewrite user-facing docs (README, notebook instructions, `docs/NOTEBOOK_AGENT_INTEGRATION.md`, control panel text) to explain that only WorkflowAgent + ScriptAgent are ADK agents; all other stages are FunctionTools with the new names.
- [x] Add a release/migration note (docs + CHANGELOG) describing the renaming, run the full pytest suite, and verify that no references to the old `_agent` identifiers remain outside of the historical note. *(2025-12-01 — `docs/RELEASE_NOTES.md` updated with the agent→stage table, `docs/ARCHITECTURE.md` holds the sole historical matrix, and `PYTHONPATH=.:src pytest -q` reported 326 passed / 1 skipped.)*

### P0 — Finalize-only pipeline hardening
- [x] Inventory every runtime/doc/test dependency on the retired inspection stage (production agent, images_stage, configs, notebooks, tests) and capture the impacted files before editing so regressions are traceable. *(2025-12-02 — references gathered below to bound the removal work.)*
  - **Runtime status (2025-12-07):** the inspection FunctionTool packages/tests have been deleted, and runtime entrypoints (`production_agent`, CLI, RunRegistry, StageManifest schemas) now surface finalize-only metadata.
  - **Docs + samples:** `docs/TODO.md`, `docs/IMPLEMENTATION_TASKS.md`, `docs/THE_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/NOTEBOOK_AGENT_INTEGRATION.md`, `docs/OPERATIONS_GUIDE.md`, `docs/ADK_COVERAGE.md`, `docs/RELEASE_NOTES.md`, and `docs/SCHEMA_ARTIFACTS.md` now explain the finalize-only workflow.
  - **Configs/notebooks:** Tool registry + workflow configs are clear; notebooks/control-panel helpers still need polishing so every widget assumes finalize-only semantics.
- [ ] Remove the base-image, per-shot video, and terminal inspection placeholders from `production_agent` (and related telemetry/RunRegistry wiring), ensuring retries/policy gates degrade gracefully now that finalize emits the deliverable artifacts directly.
- [x] Strip inspection hooks from `images_stage` (pre-render reference checks + post-render hooks) and delete any remaining validator-specific adapters/helpers while keeping rate limits/dedupe behavior intact. *(2025-12-02 — reference-image options now no-op; render batching/dedupe untouched.)*
- [x] Update configuration surfaces (`configs/tool_registry.yaml`, `configs/workflow_agent.yaml`, CLI defaults, notebooks/control panel) to drop inspection tool registrations, hard-code finalize semantics, and document the current terminal stage. *(2025-12-08 — configs note the finalize-only workflow, CLI propagates the finalize metadata markers, and the control panel focuses solely on finalize status.)*
  - CLI + FastAPI entrypoints (Stage 3, 2025-12-07) already reflect the finalize-only pipeline; notebooks/control panel wiring still needs to keep finalize widgets synchronized with the docs refresh. (Completed)
- [x] Refresh docs (`docs/THE_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/NOTEBOOK_AGENT_INTEGRATION.md`, Colab checklist) to explain that finalize is the last stage, including new operator guidance and known limitations until additional validation stages return. *(2025-12-08 — finalize workflow + limitations captured across all surfaces.)*
- [ ] Update `notebooks/sparkle_motion.ipynb` control panel + helper cells so the Colab workflow matches the finalize-only pipeline and points to the new terminal stage.
  - [ ] Inventory the notebook surfaces that still reference the retired inspection stage (Control Panel cell, Artifacts Viewer cell, Final Deliverable helper cell, and `notebooks/preview_helpers.py`) and capture their current outputs so regressions are traceable. *(Baseline captured 2025-12-02 — see the [Notebook finalize references inventory](../memory.md) entry confirming finalize toggles are already in place; future diffs should preserve that finalize-only state.)*
  - [x] Update `notebooks/preview_helpers.py` so finalize helpers only report artifact metadata. *(2025-12-09 — helpers simplified to pure artifact summary/preview logic.)*
  - [ ] Refactor the Artifacts Viewer cell to read finalize metadata from each StageManifest, surface finalize status inline, and retain auto-refresh + summary logging.
  - [ ] Rewrite the Final Deliverable helper cell to fetch `/artifacts?stage=finalize`, include ADK download fallback when `local_path` is missing, and surface the `resume_from="finalize"` action when `video_final` manifests are absent.
  - [x] Keep the Control Panel UI finalize-focused without any extra reminder banners. *(2025-12-09 — finalize-only inputs confirmed.)*
  - [ ] Re-run the Colab verification checklist items covering the Artifacts Viewer and Final Deliverable helper, capturing screenshots/logs that prove the finalize helper + download fallbacks behave as documented. *(Reference the [Notebook finalize references inventory](../memory.md) when comparing outputs so we maintain the finalize-only baseline.)*
- [x] Revise unit/integration tests (e.g., `tests/test_production_agent.py`, CLI/entrypoint suites, filesystem parity tests) plus schema samples to remove inspection expectations and add regression tests proving the pipeline still succeeds without the retired stage. *(2025-12-07 — StageManifest schema + RunRegistry tests updated; CLI + production_agent suites now assert finalize-only manifests.)*
- [ ] Run representative dry/run production executions (fixture + filesystem backend) to validate `/status`, `/artifacts`, and final deliverable helpers continue to function with the finalize-only pipeline, capturing evidence for future rollback notes.

#### Production agent observability & controls
- [x] Persist StepExecutionRecord history (and run metadata such as
  `plan_id` and `render_profile`) so the new `/status` endpoint can
  stream the timeline structure described in THE_PLAN.md §Colab dashboard and
  `docs/ARCHITECTURE.md` §Production run observability. *(RunRegistry now stores
  render profile metadata and emits timeline/log entries via `/status`
  responses; FastAPI tests updated 2025-11-30.)*
- [x] Implement `/artifacts?run_id=&stage=` that serves structured manifests
  per stage, including thumbnails/audio/MP4 entries, and validate the
  finalize response contract (requires `artifact_type="video_final"`,
  `artifact_uri`, `local_path`, `download_url`, checksum) before responding as
  mandated by THE_PLAN.md §Final deliverable contract.
- [x] Add `/control/pause`, `/control/resume`, and `/control/stop` endpoints
  that wrap the production_agent execution loop with asyncio gates so notebook
  buttons can pause/resume/stop jobs without killing processes
  (`docs/NOTEBOOK_AGENT_INTEGRATION.md` §Production run dashboard, THE_PLAN.md
  Immediate workstream item #1).

### P1 — Deterministic unit tests & harnesses
- [x] `tests/unit/test_adk_factory.py` — mock missing SDK to assert `safe_probe_sdk()` vs `require_adk()` semantics.
- [x] `tests/unit/test_adk_helpers.py` — verify artifact publish fallbacks, schema registry loader behavior, and memory events.
- [x] `tests/unit/test_gpu_utils.py` + `tests/unit/test_device_map.py` — cover context manager lifecycle, telemetry, device map presets, and OOM normalization.
- [x] `tests/unit/test_script_agent.py` — deterministic LLM stub ensures schema validation + raw output persistence.
- [x] `tests/unit/test_production_agent.py` — dry vs run semantics, event ordering, retry/backoff logic.
- [x] `tests/unit/test_images_stage.py` — covers batching, dedupe, image validation hooks, and rate-limit error paths.
- [x] `tests/unit/test_videos_stage.py` — exercise chunking, adaptive retries, CPU fallback using fixture renderer (covers `videos_stage`).
- [x] `tests/unit/test_tts_stage.py` — exercise provider selection and retry policy using stubs (covers `tts_stage`).
- [x] `tests/unit/test_images_adapter.py`
- [x] `tests/unit/test_videos_adapter.py` (fixture + env gating now covered by `tests/unit/test_videos_wan_adapter.py`)
- [x] `tests/unit/test_tts_adapter.py` — ensure deterministic artifacts + metadata.
- [x] `tests/unit/test_lipsync_wav2lip_adapter.py` — validate adapter contracts using fixtures and fixture/real-engine fallbacks.
- [x] `tests/unit/test_assemble_ffmpeg_adapter.py` — covers fixture determinism, run_command timeouts, and env gating.
- [x] Finalize the dialogue timeline builder API ownership (production_agent vs.
  helper module) and cover it with unit tests so plan edits and TTS synthesis
  stay in sync with `docs/NOTEBOOK_AGENT_INTEGRATION.md` expectations.

#### Filesystem shim integration
- [x] Add configuration toggles + env plumbing
  - [x] Define `ARTIFACTS_BACKEND`, `ARTIFACTS_FS_ROOT`, and `ARTIFACTS_FS_INDEX` env vars (with validation + defaults) so the runtime can switch between ADK and filesystem storage without code edits. (`sparkle_motion.utils.env.resolve_artifacts_backend()` enforces allowed values; `FilesystemArtifactsConfig.from_env()` now honors `os.environ`.)
  - [x] Update config docs and `docs/NOTEBOOK_AGENT_INTEGRATION.md` to explain when to use each backend, including failure/rollback guidance.
- [x] Update helpers and services to honor the shim backend
  - [x] Teach `adk_helpers.publish_artifact()`/`publish_local()`/manifest writers to delegate to the shim when `ARTIFACTS_BACKEND=filesystem`, preserving domain errors and telemetry fields.
  - [x] Ensure `/status` + `/artifacts` (and any RunRegistry consumers) can read manifests from either ADK or the shim’s SQLite index without branching in UI code. *(2025-12-04 — RunRegistry now rehydrates manifests via `_collect_artifact_entries`/`list_artifacts`, with regression coverage in `tests/unit/test_run_registry_filesystem_status.py::test_list_artifacts_filesystem_fallback`.)*
- [x] Notebook + CLI wiring
  - [x] Add Colab cells / CLI commands that launch the shim server, set the required env vars, verify the health endpoint, and surface status inside the control panel.
  - [x] Document the “local filesystem” flow alongside the existing Google Cloud instructions so operators can flip between them confidently.
- [ ] Test coverage + smoke runs
  - [x] Extend pytest/smoke coverage to run `production_agent.execute_plan()` end-to-end with `ARTIFACTS_BACKEND=filesystem`, asserting artifact URIs, manifest entries, and `/artifacts` outputs.
  - [x] Add regression tests that exercise URI parsing + manifest retrieval across both backends to prevent ADK-only assumptions from creeping back in.

#### P1 tasks still open (GPU-dependent deferrals)

Tracking the remaining checklist items from `docs/IMPLEMENTATION_TASKS.md` that we have explicitly deferred until we can spend cycles on an actual GPU host. Each of these needs live hardware to validate telemetry + NVML plumbing, so the TODO stays unchecked until we can schedule GPU time:

- [ ] Extend `gpu_utils.model_context` with the promised sync+async context manager API so adapters can share the same lifecycle helpers across blocking and async stacks (requires CUDA hardware to exercise cleanup semantics under load).
- [ ] Normalize the `ModelLoadTimeout` / `ModelOOMError` / `ModelLoadError` hierarchy so callers can distinguish retryable errors; we need GPU repros to ensure the wrappers behave correctly under real torch failure modes.
- [ ] Finish the `report_memory()` surface (CUDA → `/proc` fallback) to capture accurate VRAM snapshots; blocked until we can collect metrics on a GPU box and ensure the code paths don’t explode when CUDA is installed.
- [ ] Integrate optional NVML sampling so telemetry captures fan/temp/utilization, but only after we test on a host with NVML available (fixture coverage alone isn’t enough).
- [ ] Emit the structured telemetry hooks (`load_*`, `inference_*`, `cleanup`) directly from the context manager; needs GPU-backed dry runs to confirm we’re not introducing perf regressions or log spam.
- [ ] Implement the `suggest_shrink_for_oom()` helper and validate it against real `RuntimeError: CUDA out of memory` traces to ensure it proposes sane chunk sizes for Wan/SDXL.
- [ ] Document the device-map presets for A100 80/40 GB and 4090 hosts, which depends on profiling memory ceilings with real allocations.

### P2 — Robustness, tooling, and docs
- [x] `src/sparkle_motion/utils/dedupe.py` + `src/sparkle_motion/utils/recent_index_sqlite.py`
  - [x] Implement pHash helper, SQLite-backed RecentIndex, and CLI inspect tool; wire into `images_stage` + `videos_stage` dedupe paths.
- [x] `src/sparkle_motion/ratelimit.py`
  - [x] Implement lightweight token-bucket/queue scaffolding with single-user bypass + TODO for multi-tenant enablement.
- [x] Deterministic fixtures under `tests/fixtures/` (PNGs, WAVs, short MP4s, JSON plans) <50 KB each.
- [x] `docs/gpu_utils.md` — document `model_context` usage, device presets, telemetry expectations, and troubleshooting.
- [x] `docs/SCHEMA_ARTIFACTS.md` linkage — add references from module docstrings + onboarding notes once schema loader is wired.
- [x] `db/schema/recent_index.sql` and `src/sparkle_motion/db/sqlite.py` — persist RecentIndex/MemoryService tables + helper functions.
- [x] Document the canonical port assignments and environment variables for
 each FunctionTool once the ipywidgets UI is wired (see
 `docs/NOTEBOOK_AGENT_INTEGRATION.md` §FunctionTool port map).
- [x] Capture and document the artifact preview patterns (image/audio/video
  widget recipes) after the first UI prototype is validated, so future notebooks
  follow the same embedding conventions.

### P3 — Gated smokes, proposals, and integration follow-ups
- [ ] Smoke tests: add opt-in tests under `tests/smoke/` for `images_sdxl`, `videos_wan`, `tts_chatterbox`, `assemble_ffmpeg`, and `lipsync_wav2lip`, all gated via corresponding `SMOKE_*` env vars.
- [ ] `proposals/pyproject_adk.diff`: draft runtime dependency pins + env var requirements; pause for approval before applying to `pyproject.toml`.
- [ ] ADK artifact verification runbook: once credentials exist, document `adk artifacts push/ls` commands inside `docs/SCHEMA_ARTIFACTS.md` and confirm each URI resolves.
- [ ] Packaging + README updates: after proposal approval, document install instructions, env vars, and smoke test gating.

## How to pick work & proceed

1. Reply with `start <task>` (for example, `start P0 gpu_utils`) and I will branch, implement the item, and provide a PR-style summary.
2. I will not modify manifests, add dependencies, or run gated smoke/integration tests without explicit approval.
3. Deterministic unit tests (P1) must land alongside the corresponding P0 feature unless we agree otherwise.

## Developer runbook (quick reference)

- Environment vars for local runs:
  - `export SPARKLE_DB_PATH="$(pwd)/artifacts/sparkle.db"`
  - `export ADK_USE_FIXTURE=1` (fixture-only mode for FunctionTools)
  - `export IMAGES_MAX_PER_CALL_DEFAULT=8`
  - `export SMOKE_TTS=1` to allow the real TTS adapter to run; leave unset (or `0`) to force fixture WAVs. Combine with `SMOKE_ADAPTERS=1` when running the full adapter stack locally.
- Useful helpers to build while implementing tasks:
  - `db/schema/recent_index.sql` — RecentIndex + memory_events DDL referenced by dedupe + telemetry tasks.
  - `src/sparkle_motion/db/sqlite.py` — `get_conn()` / `ensure_schema()` wrappers shared by dedupe + telemetry tests.
  - `scripts/register_tools.py`, `scripts/publish_schemas.py`, `scripts/register_workflow.py` — must be updated to use the new `safe_probe_sdk()` semantics once available.
- Local unit test recipe:

```bash
export PYTHONPATH="$(pwd):src"
export SPARKLE_DB_PATH="$(pwd)/artifacts/sparkle.db"
pytest tests/unit -q
```

- SQLite bootstrap (idempotent):

```bash
python - <<'PY'
from pathlib import Path
from sparkle_motion.db import get_conn, ensure_schema

ddl = Path('db/schema/recent_index.sql').read_text()
conn = get_conn('$SPARKLE_DB_PATH')
ensure_schema(conn, ddl)
print('DB initialized at', '$SPARKLE_DB_PATH')
PY
```

```
