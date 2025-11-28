```markdown
# TODO — Sparkle Motion (ADK-native rebuild)

> **USER DIRECTIVE (2025-11-26):** This file is local-only per your directive — do NOT stage/commit/push `resources/TODO.md` without explicit authorization. Also avoid adding CI/Actions/PR recommendations here unless explicitly requested.

## Snapshot (2025-11-27)

Recent work completed locally and pushed to `origin/master` (commit `a0783e0`):
- Implemented `src/sparkle_motion/adk_factory.py` (guarded SDK probing, fail‑loud agent creation).
- Implemented observability and telemetry helpers (`src/sparkle_motion/observability.py`, `src/sparkle_motion/telemetry.py`).
- Wired per-tool agent creation and telemetry to multiple FunctionTools, including:
   - `function_tools/videos_wan` (pilot) — default model wired to `Wan-AI/Wan2.1-I2V-14B-720P` in fixture-mode.
   - `function_tools/images_sdxl`, `function_tools/script_agent`, `function_tools/tts_chatterbox`, `function_tools/qa_qwen2vl`, `function_tools/assemble_ffmpeg`, `function_tools/lipsync_wav2lip`.
- Implemented retry/resume helpers and `MemoryService` tests (in-memory + sqlite-backed variants).
- Removed legacy `ScriptAgent` compatibility test and related artifacts per user instruction.
- Updated/added unit and smoke tests; full local test suite passed (integration tests gated/skipped by default).
- Added fixture-mode smoke tests for several FunctionTools and pushed them; note: test runs produced `artifacts/` files that were included in the last commit.

Test status (local run with `PYTHONPATH="$(pwd):src" pytest -q -r s`):
- Full test suite completed with zero failures (skips: gated ADK integration tests).

Skipped tests (reason):
- ADK integration tests are gated and require credentials (enable with `ADK_PUBLISH_INTEGRATION=1`, `ADK_PROJECT`, and valid `GOOGLE_APPLICATION_CREDENTIALS`).

## Priorities (current)

Follow the P0→P1→P2 ordering. Items marked `done` are complete locally; items left `todo` are next.

- P0 (Essential) — done:
   - `adk_factory` — done
   - Per-tool agent creation & fail‑loud semantics — done
   - Pilot wiring (`videos_wan`) — done
   - Normalize `function_tools` (removed legacy duplicates) — done

 - P0 (Essential) — todo:
    - Fix inconsistent ADK probe semantics:
       - Problem: `adk_helpers.probe_sdk()` currently exits the process on import failure (`raise SystemExit(1)`), but many ops scripts expect a non-fatal `None` to allow CLI/local fallbacks.
       - Goal: introduce a non-fatal probe API (e.g., `probe_sdk(non_fatal=True)` or `safe_probe()` returning `None` when SDK missing) and a `require_adk()` helper for fail-fast paths.
       - Acceptance: `scripts/register_tools.py`, `scripts/publish_schemas.py`, and `scripts/register_workflow.py` should use the non-fatal probe and fall back to CLI/local-only behavior; runtime entrypoints (e.g., `function_tools/script_agent/entrypoint.py`) should call `require_adk()` to remain fail-fast.

    - Guarding vs. fatal behavior: ensure ops/util scripts use non-fatal probe, entrypoints remain fatal:
       - Problem: Some callers currently assume non-fatal probe semantics and may be short-circuited by `probe_sdk()` raising SystemExit.
       - Goal: Audit callers, update scripts to use non-fatal probe, and document the recommended usage in `docs/` (short note) so the distinction is explicit for contributors.
       - Acceptance: Add unit tests that cover non-fatal probe behavior (mocking missing SDK) and one test asserting `require_adk()` raises SystemExit when ADK unavailable.

- P1 (Reliability & observability) — done:
   - Observability & telemetry hooks — done (wired to multiple FunctionTools)
   - Retry/resume helpers — done
   - MemoryService tests — done
   - Per-tool smoke tests — done (several fixture-mode smoke tests added and passing; additional coverage optional)

- P2 (Delivery & housekeeping) — partial:
 - P2 (Delivery & housekeeping) — progress:
   - Add per-tool smoke tests for remaining FunctionTools (`images_sdxl`, `tts_chatterbox`, etc.) — completed (fixture-mode smoke tests added under `tests/smoke/` and validated locally).
   - Packaging: propose adding ADK runtime dependency to `pyproject.toml` or documentation — draft copied to tracked docs at `docs/PACKAGING_PROPOSAL.md`; a local copy remains in `resources/` per workspace policy. No manifest edits applied.

## Next recommended actions (pick one)
1. `clean-artifacts` — Remove generated test artifacts under `artifacts/` and add `artifacts/` to `.gitignore`. Recommended if you do not want test artifacts checked into VCS.
2. `add-smoke-tests` — (already done for several tools) Add or extend fixture-mode smoke tests for any remaining FunctionTools.
3. `prepare-packaging-proposal` — Draft a short `pyproject.toml` diff and a docs note describing the ADK runtime dependency and required env vars (will not apply changes without explicit approval).
4. `run-adk-integration` — Run the gated ADK integration test(s) (requires credentials/env vars). Provide credentials or set env vars when you want me to run this.

Notes about the `artifacts/` files:
- The recent smoke test run generated a number of artifact files under `artifacts/adk/...` and they were included in the most recent commit (`a0783e0`). If you prefer them untracked, choose `clean-artifacts` and I will remove them and add an appropriate `.gitignore` entry.

If you want me to proceed with any of these, reply with the action name (`clean-artifacts`, `add-smoke-tests`, `prepare-packaging-proposal`, or `run-adk-integration`). I will not commit or push additional changes to this file unless you explicitly authorize `commit-push-todo`.
# TODO — Sparkle Motion (ADK-native rebuild)

> **USER DIRECTIVE (2025-11-26):** This file is local-only per your directive — do NOT stage/commit/push `resources/TODO.md` without explicit authorization. Also avoid adding CI/Actions/PR recommendations here unless explicitly requested.

## Snapshot (2025-11-27)

Recent work completed locally and pushed to `origin/master` (latest local pushes):
- Implemented `src/sparkle_motion/adk_factory.py` (guarded SDK probing, fail‑loud agent creation).
- Implemented observability and telemetry helpers (`src/sparkle_motion/observability.py`, `src/sparkle_motion/telemetry.py`).
- Wired per-tool agent creation and telemetry to multiple FunctionTools, including:
   - `function_tools/videos_wan` (pilot) — default model wired to `Wan-AI/Wan2.1-I2V-14B-720P` in fixture-mode.
   - `function_tools/images_sdxl`, `function_tools/script_agent`, `function_tools/tts_chatterbox`, `function_tools/qa_qwen2vl`, `function_tools/assemble_ffmpeg`, `function_tools/lipsync_wav2lip`.
- Implemented retry/resume helpers and `MemoryService` tests (in-memory + sqlite-backed variants).
- Removed legacy `ScriptAgent` compatibility test and related artifacts per user instruction.
- Updated/added unit and smoke tests; full local test suite passed (integration tests gated/skipped by default).
- Added fixture-mode smoke tests for several FunctionTools and pushed them.

Test status (local run with `PYTHONPATH="$(pwd):src" pytest -q -r s`):
- Full test suite completed with zero failures (skips: gated ADK integration tests).

Skipped tests (reason):
- ADK integration tests are gated and require credentials (enable with `ADK_PUBLISH_INTEGRATION=1`, `ADK_PROJECT`, and valid `GOOGLE_APPLICATION_CREDENTIALS`).

## Priorities (current)

Follow the P0→P1→P2 ordering. Items marked `done` are complete locally; items left `todo` are next.

- P0 (Essential) — done:
   - `adk_factory` — done
   - Per-tool agent creation & fail‑loud semantics — done
   - Pilot wiring (`videos_wan`) — done
   - Normalize `function_tools` (removed legacy duplicates) — done

 - P1 (Reliability & observability) — done:
   - Observability & telemetry hooks — done (wired to multiple FunctionTools)
   - Retry/resume helpers — done
   - MemoryService tests — done
   - Per-tool smoke tests — done (several smoke tests added and passing; some tools may still be missing explicit fixture-mode coverage)

 - P2 (Delivery & housekeeping) — progress:
    - Add per-tool smoke tests for remaining FunctionTools (`images_sdxl`, `tts_chatterbox`, etc.) — completed (fixture-mode smoke tests added under `tests/smoke/` and validated locally).
   - Packaging: added optional ADK extra to repo-root `pyproject.toml` and a short install/usage note to `README.md`. Changes were committed and pushed (commit `15b7452`). The `resources/` TODO remains local-only per directive.
    - CI smoke harness: added `make smoke` target to run `tests/smoke` in isolation — completed.
    - Clean artifacts: artifacts were removed from VCS and `artifacts/` added to `.gitignore` — done

## Next recommended actions (pick one)
1. `prepare-packaging-manifest` — Completed: pinned `google-adk==1.19.0` in repository `pyproject.toml` and added an install/usage note to `README.md`. Changes were committed and pushed in commit `15b7452`.
2. `run-adk-integration` — Run the gated ADK integration test(s) (requires credentials/env vars). Provide credentials or set env vars when you want me to run this.
3. `finalize-todo` — I can produce a concise diff summary of the `resources/TODO.md` updates and present it; will not commit this file unless you authorize `commit-push-todo`.

Notes about artifacts & packaging draft:
- The recent smoke test run generated artifacts under `artifacts/`; I removed them from VCS and added `artifacts/` to `.gitignore` so they will not be tracked going forward.
- I created a local packaging proposal at `resources/PACKAGING_PROPOSAL.md` (draft). That file is intentionally kept in `resources/` and is untracked — it is for maintainers and not user-facing instructions. If you want it shared, tell me where to move it (e.g., `docs/drafts/`) and whether to clean its "DRAFT" wording.

If you want me to proceed with any of the next actions, reply with the action name (`add-smoke-tests`, `prepare-packaging-proposal`, or `run-adk-integration`). I will not commit or push additional changes to this file unless you explicitly authorize `commit-push-todo`.

## Prioritized Engineering ToDo (derived from `docs/IMPLEMENTATION_TASKS.md`)

This section is a concise engineering ToDo list derived from
`docs/IMPLEMENTATION_TASKS.md`, aligned to the updated architecture and
rollout notes in `docs/ARCHITECTURE.md` and `docs/THE_PLAN.md`.

Priority mapping:
- P0: Essential functionality (implement before code-phase completion)
- P1: Unit tests for P0
- P2: Non-essential features / helpers
- P3: Unit tests for P2 and packaging/proposal drafts

## Authoritative Implementation ToDo (P0 → P3)

Below is an expanded, implementer-facing ToDo derived from
`docs/IMPLEMENTATION_TASKS.md`. Each priority section contains granular
subtasks with acceptance criteria and file references so engineers can pick
work items and produce reviewable PRs.

Priority mapping:
- P0: Essential functionality (must be implemented before code-phase completion)
- P1: Unit tests that validate P0 behavior (run without heavy deps)
- P2: Non-essential helpers and robustness improvements
- P3: Tests and proposal artifacts for P2 features; packaging proposal drafts

P0 — Essential functionality (breakdown)

- `gpu_utils.model_context` core implementation
  - Subtasks:
    1. Add `src/sparkle_motion/gpu_utils.py` with `model_context()` context manager.
    2. Implement `ModelContext` object with `pipeline`/`model` attr and `report_memory()`.
    3. Define `ModelOOMError` and normalize all CUDA OOMs to this type (include `stage`).
    4. Emit telemetry via `adk_helpers.write_memory_event()` at load/start/exit.
  - Acceptance: `with model_context(...) as ctx:` yields `ctx.pipeline`; unit test
    asserts `report_memory()` returns device snapshots and OOM are raised as `ModelOOMError`.

- `gpu_utils` device_map & sharding helpers
  - Subtasks:
    1. Implement `compute_device_map(host_profile, model_size)` helper.
    2. Add presets for `a100-80gb`, `a100-40gb`, `rtx4090` in `gpu_utils.presets`.
    3. Document usage in `docs/gpu_utils.md` with examples.
  - Acceptance: `tests/unit/test_device_map.py` verifies expected maps for sample hosts.

- `production_agent` orchestrator + step records
  - Subtasks:
    1. Create `src/sparkle_motion/production_agent.py` skeleton with `execute_plan(plan, mode)`.
    2. Define `StepExecutionRecord` dataclass and progress event API.
    3. Implement dry-run simulation that returns an invocation graph and resource estimates.
  - Acceptance: `tests/unit/test_production_agent.py` verifies dry vs run semantics and step records.

- `adk_factory` probe & factory
  - Subtasks:
    1. Add `src/sparkle_motion/adk_factory.py` with `safe_probe_sdk()` (non-fatal) and `require_adk()` (fail-fast).
    2. Implement `get_agent(tool_name, model_spec, mode)` factory helper.
    3. Add tests that mock missing SDK and assert `safe_probe_sdk()` returns `None` while `require_adk()` raises.
  - Acceptance: scripts can call `safe_probe_sdk()`; runtime entrypoints call `require_adk()`.

- `script_agent.generate_plan` (LLM stubbed)
  - Subtasks:
    1. Implement `script_agent.generate_plan(prompt)` using `adk_factory.get_agent('script_agent',...)`.
    2. Validate generated `MoviePlan` against canonical schema and persist raw LLM output.
    3. Provide deterministic LLM stub for unit tests.
  - Acceptance: `tests/unit/test_script_agent.py` asserts schema validation and raw output persistence.

- Per-media agents + adapter stubs (images, videos, tts)
  - Images: `images_agent` orchestration + `function_tools/images_sdxl` stub.
    - Subtasks: batching/chunking, rate-limiter hook, prompt QA integration, dedupe flag handling.
    - Acceptance: `tests/unit/test_images_agent.py` validates batching, ordering, dedupe using deterministic stubs.
  - Videos: `videos_agent` + `function_tools/videos_wan` stub.
    - Subtasks: chunking/sharding, overlap/reassembly, OOM fallback semantics.
    - Acceptance: `tests/unit/test_videos_agent.py` validates chunk/reassembly and shrink-on-oom paths.
  - TTS: `tts_agent` + `function_tools/tts_chatterbox` stub.
    - Subtasks: provider selection, VoiceMetadata handling, watermark metadata propagation.
    - Acceptance: `tests/unit/test_tts_agent.py` verifies metadata and watermark flag behavior.

P1 — Unit tests for P0 (deterministic, fast)

- Add deterministic stubs and unit tests that exercise P0 behavior without heavy deps:
  - `tests/unit/test_gpu_utils.py` — test load/cleanup/oom normalization and `report_memory()`.
  - `tests/unit/test_production_agent.py` — assert dry-run simulation and run orchestration using stubs.
  - `tests/unit/test_script_agent.py` — plan generation and raw LLM output audit.
  - `tests/unit/test_images_agent.py`, `tests/unit/test_videos_agent.py`, `tests/unit/test_tts_agent.py` — use adapter stubs to validate batching, chunking and provider selection.
  - Acceptance: All P1 tests run under `pytest` without needing `torch`/heavy libs and cover success/failure paths.

P2 — Non-essential helpers & robustness

- `utils/dedupe.py` implementation
  - Subtasks: pHash wrapper, small pluggable recent-index API (LRU/Redis), CLI inspect tool.
  - Acceptance: `tests/unit/test_dedupe.py` verifies pHash values and cache eviction behavior.

- Multi-GPU presets & sharding helpers (docs + small helpers)
  - Subtasks: common host profiles, `compute_device_map()` examples, docs `docs/gpu_utils.md`.
  - Acceptance: documented presets and unit test coverage for mapping heuristics.

- Deterministic test fixtures
  - Subtasks: add small pngs/wavs/mp4s under `tests/fixtures/` to support unit tests.
  - Acceptance: fixtures are referenced by unit tests and are small (<50 KB each).

P3 — Tests & proposals (review artifacts only)

- Integration smoke tests (opt-in)
  - Subtasks: add gated smoke tests that run only when corresponding env var(s) set: `SMOKE_ADK`, `SMOKE_TTS`, `SMOKE_LIPSYNC`, etc.
  - Acceptance: smoke tests are skipped by default and documented in `README.md`.

- `proposals/pyproject_adk.diff`
  - Subtasks: draft proposal diff with suggested runtime deps (pins and rationale) and required env vars; include minimal CI notes for GPU runners.
  - Acceptance: a review-ready patch under `proposals/` (do not apply without explicit approval).

  ## Per-Agent & FunctionTool Checklists (checkable)

  Below are explicit, checkable subtasks for each Agent and FunctionTool mentioned
  above. Use these to mark progress in PRs or local work. Each checkbox is a
  single, reviewable unit of work and includes the file(s) likely to change.

  ### `script_agent` (plan generation)
  - [ ] Add `src/sparkle_motion/script_agent.py` skeleton with `generate_plan(prompt)` API. [P0]
  - [ ] Wire `script_agent` to `adk_factory.get_agent('script_agent', ...)` for LLM creation. [P0]
  - [ ] Add schema validation step using the canonical `MoviePlan` schema. [P0]
  - [ ] Persist raw LLM output for auditing (artifact or test fixture path). [P1]
  - [ ] Add `tests/unit/test_script_agent.py` with deterministic LLM stub. [P1]
  - Acceptance: `generate_plan()` returns a valid `MoviePlan` and raw output is saved.

  ### `production_agent` (orchestration)
  - [ ] Create `src/sparkle_motion/production_agent.py` with `execute_plan(plan, mode)`. [P0]
  - [ ] Implement `StepExecutionRecord` dataclass in `src/sparkle_motion/types.py` or nearby. [P0]
  - [ ] Implement dry-run simulation that returns an invocation graph (no heavy calls). [P0]
  - [ ] Implement basic retry/backoff and progress event hooks (callback or pub/sub). [P0]
  - [ ] Add `tests/unit/test_production_agent.py` covering dry vs run semantics. [P1]
  - Acceptance: dry run returns invocation graph; run mode executes using stubs and emits step records.

  ### `adk_factory` (SDK probe & factory)
  - [ ] Add `src/sparkle_motion/adk_factory.py` with `safe_probe_sdk()` and `require_adk()`. [P0]
  - [ ] Implement `get_agent(tool_name, model_spec, mode)` factory helper. [P0]
  - [ ] Add tests to mock missing ADK and assert `safe_probe_sdk()` returns `None` and `require_adk()` raises. [P1]
  - Acceptance: scripts can call `safe_probe_sdk()` for non-fatal behavior; entrypoints call `require_adk()`.

  ### `gpu_utils` (model_context & helpers)
  - [ ] Add `src/sparkle_motion/gpu_utils.py` with `model_context()` context manager API. [P0]
  - [ ] Implement `ModelContext` object and `report_memory()` method. [P0]
  - [ ] Normalize OOMs to `ModelOOMError` with `stage` metadata. [P0]
  - [ ] Add `compute_device_map()` helper and device presets (a100/4090) in `gpu_utils.presets`. [P0]
  - [ ] Document `docs/gpu_utils.md` with usage and examples. [P2]
  - [ ] Add `tests/unit/test_gpu_utils.py` and `tests/unit/test_device_map.py`. [P1]
  - Acceptance: context manager yields `ctx.pipeline`; `report_memory()` returns snapshots; device maps validated by tests.

  ### `images_agent` (decision layer)
  - [ ] Implement `src/sparkle_motion/images_agent.py` orchestration API `render(prompt, opts)`. [P0]
  - [ ] Implement batching/chunking logic (respect `max_images_per_call`). [P0]
  - [ ] Integrate a rate-limiter hook (pluggable implementation placed in `src/sparkle_motion/ratelimit.py`). [P1]
  - [ ] Wire pre-render QA via `qa_qwen2vl.inspect_frames()` (call stubbed in unit tests). [P1]
  - [ ] Add dedupe flag handling using `utils/dedupe.py` when enabled. [P2]
  - [ ] Add `tests/unit/test_images_agent.py` for batching, QA, dedupe behaviors using deterministic stubs. [P1]
  - Acceptance: agent splits large requests, honors ordering, and performs QA/dedupe as configured.

  ### `function_tools/images_sdxl` (adapter stub)
  - [ ] Create `function_tools/images_sdxl/entrypoint.py` with `render_images(prompt, opts)`. [P0]
  - [ ] Use `gpu_utils.model_context('sdxl', weights=...)` in adapter implementation. [P0]
  - [ ] Provide a deterministic test stub that returns small 16x16 PNGs (seeded by prompt+seed). [P1]
  - [ ] Add `tests/unit/test_images_adapter.py` to assert artifact metadata and pHash determinism. [P1]
  - Acceptance: adapter stub returns deterministic artifacts consumable by `images_agent` tests.

  ### `videos_agent` (decision layer)
  - [ ] Implement `src/sparkle_motion/videos_agent.py` with chunking/sharding and reassembly orchestration. [P0]
  - [ ] Add OOM fallback semantics (shrink chunk or degrade quality, retry policy). [P0]
  - [ ] Add `tests/unit/test_videos_agent.py` for chunking/reassembly and OOM fallback using stubs. [P1]
  - Acceptance: agent correctly reassembles chunks and triggers fallback on simulated OOMs.

  ### `function_tools/videos_wan` (Wan2.1 adapter stub)
  - [ ] Create `function_tools/videos_wan/entrypoint.py` with `render_video(start_frames, end_frames, prompt, opts)`. [P0]
  - [ ] Use `gpu_utils.model_context()` for loading pipeline when not in stub mode. [P0]
  - [ ] Provide deterministic frame-sequence stubs for unit testing. [P1]
  - [ ] Add `tests/unit/test_videos_adapter.py` to validate frame outputs and metadata. [P1]
  - Acceptance: adapter stub produces predictable frames and metadata for the agent tests.

  ### `tts_agent` (decision layer)
  - [ ] Implement `src/sparkle_motion/tts_agent.py` with `synthesize(text, voice_config)` API. [P0]
  - [ ] Implement provider selection and failover rules (configurable list in `configs/tts_providers.yaml`). [P0]
  - [ ] Add policy checks (lightweight text moderation) and telemetry hooks. [P1]
  - [ ] Add `tests/unit/test_tts_agent.py` verifying provider selection, retries, and metadata propagation using stubs. [P1]
  - Acceptance: `tts_agent` picks providers correctly and propagates VoiceMetadata and watermarked flags.

  ### `function_tools/tts_chatterbox` (adapter stub)
  - [ ] Create `function_tools/tts_chatterbox/entrypoint.py` with `synthesize(text, voice_config) -> ArtifactRef`. [P0]
  - [ ] Provide deterministic WAV stub and VoiceMetadata fields (`duration_s`, `sample_rate`, `voice_id`). [P1]
  - [ ] When real provider is enabled (gated), load via `ChatterboxTTS.from_pretrained()` inside `gpu_utils.model_context()`. [P3]
  - [ ] Add `tests/unit/test_tts_adapter.py` for metadata and watermark flag behavior. [P1]
  - Acceptance: stub produces WAV artifact and correct VoiceMetadata; gated integration runs under `SMOKE_TTS`.

  ### `qa_qwen2vl` (QA adapter)
  - [ ] Implement `function_tools/qa_qwen2vl/entrypoint.py` with `inspect_frames(frames, prompts) -> QAReport`. [P2]
  - [ ] Add `tests/unit/test_qa_qwen2vl.py` that verifies QAReport shape using simple inputs. [P2]
  - Acceptance: QA adapter returns structured `QAReport` usable by agents for policy decisions.

  ### `assemble_ffmpeg` (deterministic assembler)
  - [ ] Implement `function_tools/assemble_ffmpeg/assemble.py` with `assemble_clips(movie_plan, clips, audio, out_path, opts)`. [P2]
  - [ ] Add safe subprocess helper `run_command()` with structured `SubprocessResult`. [P2]
  - [ ] Add unit tests that exercise small in-memory clips via fixtures (no real ffmpeg required in unit tests). [P2]
  - Acceptance: assembler translates a minimal `movie_plan` into a sequence of safe commands and returns `ArtifactRef` on success.

  ### `lipsync_wav2lip` (adapter)
  - [ ] Implement `function_tools/lipsync_wav2lip/entrypoint.py` with `run_wav2lip(face_video, audio, out_path, opts)`. [P2]
  - [ ] Provide a subprocess fallback that calls a pinned upstream commit if Python API not available. [P2]
  - [ ] Add `tests/unit/test_lipsync.py` verifying invocation args and output metadata using fixtures. [P2]
  - Acceptance: adapter provides a safe wrapper and a deterministic test path.

  ### `utils/dedupe.py` and supporting tests
  - [ ] Implement `src/sparkle_motion/utils/dedupe.py` with `phash(image_path)` and `RecentIndex` (LRU) API. [P2]
  - [ ] Add optional Redis-backed adapter `RecentIndexRedis` behind a simple interface. [P3]
  - [ ] Add `tests/unit/test_dedupe.py` exercising pHash and eviction behavior with fixture images. [P2]
  - Acceptance: pHash outputs stable values and RecentIndex behaves as expected under unit tests.

  ## How to pick work & proceed

How to pick work & proceed

- I will not write runtime code or edit manifests until you explicitly approve a specific task (reply with `start <id>` or `start P0 <item>`). When you approve a task I will:
  1. Open a local feature branch and scaffold the minimal implementation and deterministic unit tests.
  2. Run unit tests locally (no heavy deps) and present results and a short PR-style diff summary.
  3. For any dependency needs prepare `proposals/pyproject_adk.diff` and pause for your approval before applying.

If you want me to begin, say which todo id or item to start (for example: `start 2` to begin `gpu_utils: core impl`).


