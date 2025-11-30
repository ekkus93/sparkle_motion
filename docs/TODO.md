```markdown
# TODO — Sparkle Motion (ADK-native rebuild)

> **USER DIRECTIVE (2025-11-26):** This file is local-only per your directive — do NOT stage/commit/push `resources/TODO.md` without explicit authorization. Also avoid adding CI/Actions/PR recommendations here unless explicitly requested.

## Snapshot (2025-11-30)

- Script + Production agents are live end-to-end: `script_agent.generate_plan()` persists validated MoviePlans, `production_agent.execute_plan()` is wired through the WorkflowAgent/tool registry, and the new production-agent FunctionTool entrypoint plus CLI/tests are green.
- `gpu_utils.model_context()` now requires explicit model keys + loaders, eliminating the legacy warning path; full-suite pytest (241 passed / 1 skipped) remains green after the latest adapter additions.
- Schema + QA artifacts are exported/published (`docs/SCHEMA_ARTIFACTS.md` guides the URIs, QA policy bundle lives under `artifacts/qa_policy/v1/`), so downstream modules consume typed resolvers via `schema_registry`.
- Runtime profile remains single-user/Colab-local; remaining P0 work is concentrated on the TTS agent plus dedupe/rate-limit scaffolding. Videos agent orchestration/tests are complete and publishing artifacts via `videos_agent.render_video()`.
- Production agent + `tts_agent` now synthesize dialogue per line, record `line_artifacts` metadata (voice_id, provider_id, durations), and publish WAV artifacts via `tts_audio` entries so downstream lipsync and QA logic can trace every clip. The run is gated by `SMOKE_TTS`/`SMOKE_ADAPTERS` (fixture-only when unset).
- `assemble_ffmpeg` FunctionTool is implemented with a deterministic MP4 fixture plus optional ffmpeg concat path; docs/tests updated and metadata now includes duration/codec provenance for QA automation.
- `videos_wan` FunctionTool now routes through the Wan adapter with deterministic fixtures by default, publishes `videos_wan_clip` artifacts, and surfaces chunk metadata/telemetry with smoke-flag gating for real GPU runs.
- `qa_qwen2vl` adapter + entrypoint hardened with structured metadata propagation, frame-id plumbing, download limits, and smoke-level assertions so upstream agents can depend on the augmented fields.
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
  - [x] Wire `sparkle_motion.schema_registry` to load `configs/schema_artifacts.yaml` and surface typed getters for MoviePlan, AssetRefs, QAReport, StageEvent, Checkpoint, QA policy bundle.
  - [x] Provide fallback resolution logic (`artifact://` vs `file://`) with explicit warnings in fixture mode.

#### Agents (decision/orchestration layers)
- [x] `script_agent` (`src/sparkle_motion/script_agent.py`) — **STATUS: complete** (generate_plan + artifact persistence landed)
  - [x] `generate_plan(prompt) -> MoviePlan` stub that calls `adk_factory.get_agent()` and enforces schema validation.
  - [x] Persist raw LLM output + validation metadata via `adk_helpers.publish_artifact()`.
- [x] `production_agent` (`src/sparkle_motion/production_agent.py`)
  - [x] Implement `execute_plan(plan, mode='dry'|'run')` plus `StepExecutionRecord` dataclass and progress hooks.
  - [x] Dry-run simulation returns invocation graph + resource estimate; run-mode orchestrates adapters via WorkflowAgent-compatible contract.
- [x] `images_agent` orchestration
  - [x] Enforce batching (`max_images_per_call`), per-step dedupe flag, and per-plan ordering guarantees.
  - [x] Integrate QA pre-check via `qa_qwen2vl.inspect_frames()` when reference images provided.
  - [x] Hook rate limiter/queue interface (stub-friendly) to unblock future multi-user rollout.
- [x] `videos_agent`
  - [x] Implement chunking/sharding + overlap merge for Wan2.1, with shrink-on-OOM fallback behavior.
  - [x] Expose `render_video(start_frames, end_frames, prompt, opts)` orchestrator that selects adapter endpoints (fixture vs real).
- [x] `tts_agent`
  - [x] Implement provider selection + retry policy driven by `configs/tts_providers.yaml`.
  - [x] Surface VoiceMetadata (voice_id/name, sample_rate, duration, watermark flag) and telemetry.

### Sequence of Work — `tts_agent`

1. [x] Finalize `configs/tts_providers.yaml` (provider ids, tiering flags, rate caps, fixture aliases) and document the selection contract inside `docs/IMPLEMENTATION_TASKS.md`.
2. [x] Implement `sparkle_motion/tts_agent.py` with provider scoring, bounded retries, VoiceMetadata emission, and `adk_helpers.publish_artifact()` integrations plus structured telemetry.
3. [x] Flesh out `function_tools/tts_chatterbox/entrypoint.py`: add deterministic WAV fixture pipeline, gate the real adapter behind `SMOKE_TTS`, and ensure both paths share a metadata builder.
4. [x] Author `tests/unit/test_tts_agent.py` and `tests/unit/test_tts_adapter.py` covering provider selection, retry downgrades, fixture determinism, and artifact metadata.
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
- [x] `function_tools/qa_qwen2vl`
  - Implemented adapter + FastAPI entrypoint with deterministic fixtures, Qwen2-VL invocation hooks, telemetry, and artifact publishing.
  - CLI payload builder plus shared smoke/unit tests now cover prompt/frame validation, policy metadata, and human-review fallbacks.
  - [x] Implement `inspect_frames(frames, prompts) -> QAReport` with structured Qwen2-VL inference, JSON parsing, and GPU model-context integration.
- [x] `function_tools/assemble_ffmpeg`
  - [x] Implemented adapter with deterministic MP4 fixture + optional ffmpeg concat path, safe `run_command` wrapper, metadata (engine, plan_id, command tails) and FastAPI entrypoint publishing `video_final` artifacts.
- [x] `function_tools/lipsync_wav2lip`
  - [x] Wrap Wav2Lip CLI/API invocation with deterministic stub + retries/cleanup API.

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
  embed the MP4 inline, warn when `qa_skipped` is true, handle missing
  `video_final` rows by surfacing a retry action (`resume_from="qa_publish"`),
  and provide Drive/download fallbacks.
- **Colab manual verification checklist**
  - [ ] Launch the control panel cell with live `script_agent`/`production_agent` servers and confirm Generate Plan flows return `artifact_uri` plus autofill the Plan URI field.
  - [ ] Run Production in both `dry` and `run` modes, then exercise `Pause`/`Resume`/`Stop` buttons against real `/control/*` endpoints to confirm acknowledgements surface in the Control Responses pane.
  - [ ] Enable the status polling toggle once `/status` is available, validate that `/ready` + `/status` snapshots stream into the Status pane, and ensure the polling loop can be started/stopped without hanging the notebook.
  - [ ] Test the artifacts viewer: specify a `run_id`, optionally a `stage`, and confirm `/artifacts` responses render (including `video_final` metadata) and auto-refresh when the checkbox is enabled.
  - [ ] After the "final deliverable" helper cell lands, verify inline MP4 embedding, QA badge rendering when `qa_skipped` is true, and Drive download fallbacks inside Colab.
   - [ ] Run the full Colab preflight sequence (auth, env vars, pip installs, Drive mount, GPU/disk checks, `/ready` probes) and confirm each helper cell succeeds end-to-end.
   - [ ] Generate multiple MoviePlans via the control panel, inspect the rendered plan JSON/tables, and confirm dialogue timeline, base_images count, and `render_profile` constraints all validate before production.
   - [ ] Manually edit a plan in-notebook (e.g., tweak shot durations or base-image references) and ensure the MoviePlan validator surfaces mismatches (shot runtime vs. dialogue timeline, base_images count) before allowing production.
   - [ ] Kick off production runs with `qa_mode="full"` and `qa_mode="skip"`, verifying that StepExecutionRecord history, QA badges, and `/status` responses reflect the requested mode.
   - [ ] Observe the dialogue/TTS stage outputs: confirm per-line artifacts, stitched `tts_timeline.wav`, and timeline-with-actuals manifests appear in `/artifacts` and render inside the notebook viewers.
   - [ ] Validate base-image QA flows by forcing a known failure (bad prompt/fixture), confirming the notebook surfaces the QA report, regenerates via `images_agent`, and only proceeds after a pass.
   - [ ] Validate clip-level QA + retry behavior by injecting a `qa_qwen2vl` failure, checking that the control panel pauses, surfaces retry counts, and resumes automatically once QA passes.
   - [ ] Use the control buttons to trigger `pause`, `resume`, and `stop` during a long run, then exercise `resume_from=<stage>` to ensure partial progress can restart without rerunning prior stages.
   - [ ] Confirm the artifacts viewer renders every stage manifest (base images, TTS audio, video clips, QA reports, assembly outputs) with inline previews (`Image`, `Audio`, `Video`) and that auto-refresh never duplicates or drops entries.
   - [ ] For the final deliverable helper, cover both local-path and remote-download paths: trigger ADK download fallback when `local_path` is absent, display the inline video, verify QA warnings when `qa_skipped` is true, and test the "resume from qa_publish" action when `video_final` is missing.

#### Pipeline JSON contracts
- [x] Promote the documented request/response envelopes for
  `script_agent`/`production_agent` into canonical schema artifacts (MoviePlan,
  RunContext, StageManifest) and enforce them directly inside the FastAPI
  entrypoints. (Schemas now exported via `scripts/export_schemas.py`, stored
  under `schemas/*.schema.json`, and the entrypoints/tests validate those models.)
- [x] For each FunctionTool (`images_sdxl`, `videos_wan`, `tts_chatterbox`,
  `lipsync_wav2lip`, `assemble_ffmpeg`, `qa_qwen2vl`), implement typed
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
  `tts_agent` once per dialogue timeline entry, record `line_artifacts`, and
  stitch a single `tts_timeline.wav` artifact with measured timings so later
  stages and `/artifacts` consumers can rely on exact offsets
  (`docs/NOTEBOOK_AGENT_INTEGRATION.md` §§Dialogue timeline + TTS synthesis,
  THE_PLAN.md Stage table).
- [x] Integrate `qa_qwen2vl` twice within production_agent: (1) base-image QA
  right after SDXL renders, retrying failed images before video, and (2)
  per-shot video QA with retry budgets + `qa_skipped` annotations when
  `qa_mode="skip"` is requested (reference `docs/NOTEBOOK_AGENT_INTEGRATION.md`
  §§Base images + QA, Clip-level QA + retries, THE_PLAN.md §§Video QA rows).
- [x] Add the terminal `qa_publish` stage that inspects the final MP4/audio
  pair, writes QA reports, and publishes the `video_final` manifest entry that
  downstream `/artifacts` consumers expect (THE_PLAN.md §Final deliverable
  contract, `docs/NOTEBOOK_AGENT_INTEGRATION.md` Final deliverable helper).

#### Production agent observability & controls
- [x] Persist StepExecutionRecord history (and run metadata such as
  `plan_id`, `render_profile`, and `qa_mode`) so the new `/status` endpoint can
  stream the timeline structure described in THE_PLAN.md §Colab dashboard and
  `docs/ARCHITECTURE.md` §Production run observability. *(RunRegistry now stores
  render profile + qa_mode metadata and emits timeline/log entries via `/status`
  responses; FastAPI tests updated 2025-11-30.)*
- [x] Implement `/artifacts?run_id=&stage=` that serves structured manifests
  per stage, including thumbnails/audio/MP4 entries, and validate the
  `qa_publish` response contract (requires `artifact_type="video_final"`,
  `artifact_uri`, `local_path`, `download_url`, checksum) before responding as
  mandated by THE_PLAN.md §Final deliverable contract.
- [x] Add `/control/pause`, `/control/resume`, and `/control/stop` endpoints
  that wrap the production_agent execution loop with asyncio gates so notebook
  buttons can pause/resume/stop jobs without killing processes
  (`docs/NOTEBOOK_AGENT_INTEGRATION.md` §Production run dashboard, THE_PLAN.md
  Immediate workstream item #1).
- [ ] Thread a `qa_mode` option from `/invoke` through production_agent, store
  it with run state, and ensure status/artifact responses badge `qa_skipped`
  runs exactly as the docs require (THE_PLAN.md §Live status polling,
  `docs/NOTEBOOK_AGENT_INTEGRATION.md` QA modes subsection).
- [x] Thread a `qa_mode` option from `/invoke` through production_agent, store
  it with run state, and ensure status/artifact responses badge `qa_skipped`
  runs exactly as the docs require (THE_PLAN.md §Live status polling,
  `docs/NOTEBOOK_AGENT_INTEGRATION.md` QA modes subsection).

### P1 — Deterministic unit tests & harnesses
- [x] `tests/unit/test_adk_factory.py` — mock missing SDK to assert `safe_probe_sdk()` vs `require_adk()` semantics.
- [x] `tests/unit/test_adk_helpers.py` — verify artifact publish fallbacks, schema registry loader behavior, and memory events.
- [x] `tests/unit/test_gpu_utils.py` + `tests/unit/test_device_map.py` — cover context manager lifecycle, telemetry, device map presets, and OOM normalization.
- [x] `tests/unit/test_script_agent.py` — deterministic LLM stub ensures schema validation + raw output persistence.
- [x] `tests/unit/test_production_agent.py` — dry vs run semantics, event ordering, retry/backoff logic.
- [x] `tests/unit/test_images_agent.py` — covers batching, dedupe, QA hooks, and rate-limit error paths.
- [x] `tests/unit/test_videos_agent.py` — exercise chunking, adaptive retries, CPU fallback using fixture renderer.
- [x] `tests/unit/test_tts_agent.py` — exercise provider selection and retry policy using stubs.
- [x] `tests/unit/test_images_adapter.py`
- [x] `tests/unit/test_videos_adapter.py` (fixture + env gating now covered by `tests/unit/test_videos_wan_adapter.py`)
- [x] `tests/unit/test_tts_adapter.py` — ensure deterministic artifacts + metadata.
- [x] `tests/unit/test_qa_qwen2vl.py` — validate QA adapter structured parsing using mocked Qwen responses.
- [x] `tests/unit/test_lipsync_wav2lip_adapter.py` — validate adapter contracts using fixtures and fixture/real-engine fallbacks.
- [x] `tests/unit/test_assemble_ffmpeg_adapter.py` — covers fixture determinism, run_command timeouts, and env gating.
- [ ] Finalize the dialogue timeline builder API ownership (production_agent vs.
  helper module) and cover it with unit tests so plan edits and TTS synthesis
  stay in sync with `docs/NOTEBOOK_AGENT_INTEGRATION.md` expectations.

### P2 — Robustness, tooling, and docs
- [x] `src/sparkle_motion/utils/dedupe.py` + `src/sparkle_motion/utils/recent_index_sqlite.py`
  - [x] Implement pHash helper, SQLite-backed RecentIndex, and CLI inspect tool; wire into `images_agent` + `videos_agent` dedupe paths.
- [x] `src/sparkle_motion/ratelimit.py`
  - [x] Implement lightweight token-bucket/queue scaffolding with single-user bypass + TODO for multi-tenant enablement.
- [x] Deterministic fixtures under `tests/fixtures/` (PNGs, WAVs, short MP4s, JSON plans) <50 KB each.
- [x] `docs/gpu_utils.md` — document `model_context` usage, device presets, telemetry expectations, and troubleshooting.
- [x] `docs/SCHEMA_ARTIFACTS.md` linkage — add references from module docstrings + onboarding notes once schema loader is wired.
- [x] `db/schema/recent_index.sql` and `src/sparkle_motion/db/sqlite.py` — persist RecentIndex/MemoryService tables + helper functions.
- [ ] Document the canonical port assignments and environment variables for
  each FunctionTool once the ipywidgets UI is wired (per
  `docs/NOTEBOOK_AGENT_INTEGRATION.md`).
- [ ] Capture and document the artifact preview patterns (image/audio/video
  widget recipes) after the first UI prototype is validated, so future notebooks
  follow the same embedding conventions.

### P3 — Gated smokes, proposals, and integration follow-ups
- [ ] Smoke tests: add opt-in tests under `tests/smoke/` for `images_sdxl`, `videos_wan`, `tts_chatterbox`, `assemble_ffmpeg`, `lipsync_wav2lip`, `qa_qwen2vl`, all gated via corresponding `SMOKE_*` env vars.
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
