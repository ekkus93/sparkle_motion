```markdown
# TODO — Sparkle Motion (ADK-native rebuild)

> **USER DIRECTIVE (2025-11-26):** This file is local-only per your directive — do NOT stage/commit/push `resources/TODO.md` without explicit authorization. Also avoid adding CI/Actions/PR recommendations here unless explicitly requested.

## Snapshot (2025-11-28)

- `docs/THE_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/SCHEMA_ARTIFACTS.md` are now aligned; `docs/IMPLEMENTATION_TASKS.md` remains the authoritative engineering backlog.
- Runtime is still single-user/Colab-local; rate-limiter + queue semantics are intentionally deferred until we add multi-user capacity.
- Schema artifact URIs are cataloged but not yet verified against ADK ArtifactService because no ADK environment is connected.
- Next steps: execute the Implementation Tasks below; no runtime code was changed during the documentation alignment, so everything listed here is still outstanding.

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
- [x] Schema registry enforcement
  - [x] Wire `sparkle_motion.schema_registry` to load `configs/schema_artifacts.yaml` and surface typed getters for MoviePlan, AssetRefs, QAReport, StageEvent, Checkpoint, QA policy bundle.
  - [x] Provide fallback resolution logic (`artifact://` vs `file://`) with explicit warnings in fixture mode.

#### Agents (decision/orchestration layers)
- [x] `script_agent` (`src/sparkle_motion/script_agent.py`) — **STATUS: complete** (generate_plan + artifact persistence landed)
  - [x] `generate_plan(prompt) -> MoviePlan` stub that calls `adk_factory.get_agent()` and enforces schema validation.
  - [x] Persist raw LLM output + validation metadata via `adk_helpers.publish_artifact()`.
- [ ] `production_agent` (`src/sparkle_motion/production_agent.py`)
  - [ ] Implement `execute_plan(plan, mode='dry'|'run')` plus `StepExecutionRecord` dataclass and progress hooks.
  - [ ] Dry-run simulation returns invocation graph + resource estimate; run-mode orchestrates adapters via WorkflowAgent-compatible contract.
- [ ] `images_agent` orchestration
  - [ ] Enforce batching (`max_images_per_call`), per-step dedupe flag, and per-plan ordering guarantees.
  - [ ] Integrate QA pre-check via `qa_qwen2vl.inspect_frames()` when reference images provided.
  - [ ] Hook rate limiter/queue interface (stub-friendly) to unblock future multi-user rollout.
- [ ] `videos_agent`
  - [ ] Implement chunking/sharding + overlap merge for Wan2.1, with shrink-on-OOM fallback behavior.
  - [ ] Expose `render_video(start_frames, end_frames, prompt, opts)` orchestrator that selects adapter endpoints (fixture vs real).
- [ ] `tts_agent`
  - [ ] Implement provider selection + retry policy driven by `configs/tts_providers.yaml`.
  - [ ] Surface VoiceMetadata (voice_id/name, sample_rate, duration, watermark flag) and telemetry.

#### FunctionTools / adapters
- [ ] `function_tools/images_sdxl`
  - [ ] Build deterministic stub returning 16×16 PNGs seeded by `(prompt, seed)`; real model path must use `gpu_utils.model_context()`.
  - [ ] Emit artifact metadata (seed, dimensions, pHash) and publish via `adk_helpers`.
- [ ] `function_tools/videos_wan`
  - [ ] Provide chunked render stub + metadata; wrap real Wan2.1 pipeline inside `model_context()` and ensure CUDA cleanup.
  - [ ] Support `SMOKE_ADAPTERS` flag to toggle heavy loads.
- [ ] `function_tools/tts_chatterbox`
  - [ ] Add fixture implementation producing deterministic WAV bytes + metadata.
  - [ ] Gate real Chatterbox load behind `SMOKE_TTS` and ensure `gpu_utils.model_context()` handles device cleanup.
- [ ] `function_tools/qa_qwen2vl`
  - [ ] Implement `inspect_frames(frames, prompts) -> QAReport` stub plus hooks for real Qwen-2-VL invocation later.
- [ ] `function_tools/assemble_ffmpeg`
  - [ ] Implement deterministic ffmpeg planner using safe subprocess helper and returning `ArtifactRef`.
- [ ] `function_tools/lipsync_wav2lip`
  - [ ] Wrap Wav2Lip CLI/API invocation with deterministic stub + retries/cleanup API.

### P1 — Deterministic unit tests & harnesses
- [x] `tests/unit/test_adk_factory.py` — mock missing SDK to assert `safe_probe_sdk()` vs `require_adk()` semantics.
- [x] `tests/unit/test_adk_helpers.py` — verify artifact publish fallbacks, schema registry loader behavior, and memory events.
- [x] `tests/unit/test_gpu_utils.py` + `tests/unit/test_device_map.py` — cover context manager lifecycle, telemetry, device map presets, and OOM normalization.
- [x] `tests/unit/test_script_agent.py` — deterministic LLM stub ensures schema validation + raw output persistence.
- [ ] `tests/unit/test_production_agent.py` — dry vs run semantics, event ordering, retry/backoff logic.
- [ ] `tests/unit/test_images_agent.py`, `tests/unit/test_videos_agent.py`, `tests/unit/test_tts_agent.py` — exercise batching, chunking, dedupe, QA integration, provider selection using stubs.
- [ ] `tests/unit/test_images_adapter.py`, `tests/unit/test_videos_adapter.py`, `tests/unit/test_tts_adapter.py` — ensure deterministic artifacts + metadata.
- [ ] `tests/unit/test_qa_qwen2vl.py`, `tests/unit/test_assemble_ffmpeg.py`, `tests/unit/test_lipsync.py` — validate adapter contracts using fixtures.

### P2 — Robustness, tooling, and docs
- [ ] `src/sparkle_motion/utils/dedupe.py` + `src/sparkle_motion/utils/recent_index_sqlite.py`
  - [ ] Implement pHash helper, SQLite-backed RecentIndex, and CLI inspect tool; wire into `images_agent` + `videos_agent` dedupe paths.
- [ ] `src/sparkle_motion/ratelimit.py`
  - [ ] Implement lightweight token-bucket/queue scaffolding with single-user bypass + TODO for multi-tenant enablement.
- [ ] Deterministic fixtures under `tests/fixtures/` (PNGs, WAVs, short MP4s, JSON plans) <50 KB each.
- [ ] `docs/gpu_utils.md` — document `model_context` usage, device presets, telemetry expectations, and troubleshooting.
- [ ] `docs/SCHEMA_ARTIFACTS.md` linkage — add references from module docstrings + onboarding notes once schema loader is wired.
- [x] `db/schema/recent_index.sql` and `src/sparkle_motion/db/sqlite.py` — persist RecentIndex/MemoryService tables + helper functions.

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
