# The Plan — UPDATED (2025-11-27)

This document is the authoritative plan for converting Sparkle Motion to an
ADK-native, WorkflowAgent-driven system. It reflects the enforced decision that
the ADK Python SDK is required for application entrypoints that instantiate
agents. This plan excludes `resources/` (samples) from runtime counts; those are
references only.

---

Authoritative runtime counts (application code only — `resources/` excluded)

- ADK agents (application runtime): 7
   - Rationale: Option A (per-tool agents) is the authoritative architecture for
      this branch. Each FunctionTool process that needs an LLM or ADK-managed
      model will instantiate its own `LlmAgent` at process startup. After
      normalizing a case-duplicate scaffold (`ScriptAgent` / `script_agent`),
      the canonical set of application FunctionTools is seven unique tools.
- FunctionTools (application-level directories under `function_tools/`): 7
   - Canonical tool directories: `script_agent`, `images_sdxl`, `videos_wan`,
      `tts_chatterbox`, `lipsync_wav2lip`, `assemble_ffmpeg`, `qa_qwen2vl`.
   - Note: there is a case-duplicate pair `ScriptAgent` / `script_agent` in
      the tree; the duplicate scaffold will be removed during normalization.

Notes:
- The `resources/` tree contains many ADK sample projects and examples. Those
  are vendor/sample code for reference only and are explicitly excluded from
  the counts above.
- ADK SDK usage in application entrypoints is mandatory: any attempt to
  instantiate an agent without the SDK or required credentials will raise and
  stop the process. No silent fallback behavior for agent construction is
  permitted in application runtime code.

---

Objectives (short)

1. Workflow-first orchestration: keep the stage graph as a WorkflowAgent YAML
   and run via ADK control plane.
2. ADK-first runtime: application entrypoints that construct agents must use
   the `google.adk` SDK and fail loudly on import/creation errors.
3. Schema-first contracts: publish canonical schemas as ADK artifacts and use
   them at runtime; allow local file fallbacks only for isolated developer
   runs.
4. Tool catalog: package heavy stages as FunctionTools with metadata and
   telemetry. Each FunctionTool process either instantiates its own agent via
   a centralized factory or raises on missing SDK.

---

Immediate actionable workstream (highest priority)

1. Design and add `src/sparkle_motion/adk_factory.py` (supporting per-tool mode)
    - Purpose: centralize ADK SDK probing, credentials checks, model selection,
       and agent construction **but support Option A (per-tool agents)**.
    - API: factory should expose `get_agent(tool_name, model_spec, mode="per-tool")`,
       `create_agent(...)`, and explicit `close_agent(name)` semantics.
    - Behavior: when `mode="per-tool"` the factory constructs a fresh
       `LlmAgent` per `get_agent` call scoped to the calling process; failures
       raise clear RuntimeErrors. The factory also documents a `shared` mode
       for future toggles but Option A is the default for this rollout.

# The Plan — UPDATED (2025-11-27)

This document is the authoritative rollout plan for converting Sparkle Motion
into an ADK-native, WorkflowAgent-driven system. It has been updated to align
with the changes made in `docs/ARCHITECTURE.md` and
`docs/IMPLEMENTATION_TASKS.md` (2025-11-27). Key updates include:

- The explicit `production_agent` role (runtime orchestrator for validated
  MoviePlan execution).
- The formal Agent vs FunctionTool separation and the decision to use Option A
  (per-tool agents) during the initial rollout.
- The mandatory `gpu_utils.model_context` pattern for adapters that load
  models, with OOM normalization to `ModelOOMError` and telemetry points.
- Per-tool gating via `SMOKE_*` flags to avoid accidental heavy installs and
  model loads in normal CI or developer runs.
- The governance requirement: any runtime dependency or manifest change must be
  proposed via `proposals/pyproject_adk.diff` and approved before edits.

---

## Authoritative runtime summary (application code only — `resources/` excluded)

- ADK agents (application runtime): 7 (Option A — per-tool agents by default)
- FunctionTools (compute adapters): 7 canonical directories
  - `script_agent`, `images_sdxl`, `videos_wan`, `tts_chatterbox`,
    `lipsync_wav2lip`, `assemble_ffmpeg`, `qa_qwen2vl`

Notes:
- `resources/` remains reference/sample code and is not counted in runtime
  totals.
- Application entrypoints that instantiate agents must require the ADK SDK
  (`require_adk()` behavior) and fail loudly when missing credentials or the
  SDK. No silent fallbacks in application runtime code.

---

## Objectives (concise)

1. WorkflowAgent-first orchestration: keep the stage graph as a WorkflowAgent
   definition and execute via ADK control plane.
2. ADK-first runtime: application entrypoints that construct agents must use
   the ADK SDK and raise on missing SDK/credentials.
3. Schema-first contracts: publish MoviePlan, AssetRefs, QAReport, and
   Checkpoint schemas as ADK artifacts and consume them at runtime.
4. Clean Agent vs FunctionTool separation: Agents = decision/orchestration;
   FunctionTools = compute adapters invoked by Agents/production_agent.
5. Deterministic testability: seeds propagate through WorkflowAgent session
   metadata; stub harnesses and torch.Generator seeding for unit tests.

---

## Immediate actionable workstream (highest priority)

1. Add `src/sparkle_motion/adk_factory.py` (per-tool agent factory)
   - Centralize SDK probing, credentials checks, and agent construction.
   - API: `get_agent(tool_name, model_spec, mode='per-tool')`, `create_agent(...)`,
     `close_agent(name)`.

2. Normalize `function_tools/` directory names
   - Remove duplicates (e.g., `ScriptAgent` vs `script_agent`) and ensure one
     canonical `entrypoint.py` per tool.

3. Implement `production_agent` at runtime
    - Responsibility: accept validated `MoviePlan` objects and execute them
       with `dry` (simulate) and `run` (execute) semantics, orchestrate steps,
       enforce policy, manage retries/backoff, and emit progress events.
    - Emit structured `StepExecutionRecord` entries per stage and expose
       `GET /status?run_id=<id>` so the Colab UI can poll for `current_stage`,
       timestamps, percent complete, and recent log lines.
    - Publish per-stage artifact manifests via
       `GET /artifacts?run_id=<id>&stage=<name>` listing `asset_uri`,
       `media_type`, thumbnail references, and labels so notebooks can preview
       base images, TTS audio clips, and MP4s immediately after creation.
    - Provide control endpoints `POST /control/pause`, `/control/resume`, and
       `/control/stop` (body `{ "run_id": "..." }`) that gate the stage runner
       via an asyncio Event so users can pause/resume/stop runs without tearing
       down artifacts.
    - Surface `qa_mode` state plus pause/stop transitions inside the status feed
       so the UI can badge “QA skipped” runs and confirm when a pause or stop
       request is honored.

4. Implement `gpu_utils.model_context` (minimal, robust pattern)
   - Context manager for guarded model load/unload, device_map presets,
     metrics snapshotting, and OOM normalization to `ModelOOMError` with
     structured metadata (`stage='load'|'inference'`, memory_snapshot).

5. Per-tool pilot & rollout
   - Pilot `videos_wan` first (largest VRAM footprint). Validate device
     mapping presets, retry/shrink strategies and fail-loud behavior.

6. Proposal & gating
   - Prepare `proposals/pyproject_adk.diff` for any runtime dependency
     additions (e.g., `torch`, `diffusers`, `chatterbox-tts`). Do not edit
     `pyproject.toml` until the proposal is reviewed and approved.

7. Tests & gating
   - Unit tests and deterministic stubs for agents/adapters.
   - Integration smoke tests gated by `SMOKE_ADK=1` (and specific flags such
     as `SMOKE_TTS=1`, `SMOKE_LIPSYNC=1`, `SMOKE_QA=1`, `SMOKE_ADAPTERS=1`).

---

## Operational constraints & hygiene

- Single-job, single-user topology remains the operational model for local
  Colab runs: tools must page models in/out and avoid keeping multiple large
  weights resident simultaneously.
- `gpu_utils.model_context` usage is mandatory for adapters that load models.
- Every adapter must emit memory telemetry via `adk_helpers.write_memory_event()`
  at critical points (post-load, post-inference, post-cleanup).
- Any change adding heavy system dependencies or binary tools requires a
  proposal diff and explicit approval.

---

## ADK helper modules (spec reference)

### `src/sparkle_motion/adk_factory.py`
- **Purpose**: single place that enforces "ADK required" semantics, centralizes agent construction, and provides lifecycle hooks for per-tool agents (Option A) with a documented path to a future shared mode.
- **Public API**:
   - `require_adk(*, allow_fixture: bool = False) -> None` — verifies SDK import + credentials, raising `MissingAdkSdkError` when unavailable (unless fixture allowed).
   - `get_agent(tool_name: str, model_spec: ModelSpec, mode: Literal['per-tool','shared']='per-tool') -> LlmAgent` — constructs/returns an agent for the calling FunctionTool; records provenance + raises `AdkAgentCreationError` on failure.
   - `create_agent(config: AgentConfig) -> LlmAgent` — lower-level helper for bespoke agents (used by future extensions/tests) so tasks do not reach into SDK internals directly.
   - `close_agent(tool_name: str) -> None` — disposes per-tool agents when FunctionTools shut down or Colab resets to avoid leaked handles.
   - `shutdown() -> None` — best-effort cleanup invoked during notebook resets/tests to close all tracked agents.
- **Failure behavior**: all functions raise typed errors (`MissingAdkSdkError`, `AdkAgentCreationError`, `AdkAgentLifecycleError`) instead of returning `None`. Errors must include `tool_name`, `model_spec`, and the underlying SDK exception text for audit. Fixture/test modes must log via `adk_helpers.write_memory_event()` when the real SDK is bypassed.
- **State**: maintains an in-memory registry (`_agents: dict[str, LlmAgentHandle]`) with lightweight metrics (`created_at`, `last_used_at`, `mode`). No persistence; callers are responsible for rehydration on process restart.

### `src/sparkle_motion/adk_helpers.py`
- **Purpose**: thin façade over common ADK operations (artifact publishing, memory timeline writes, human-input requests) so adapters and agents share the same telemetry/publishing behavior without duplicating SDK plumbing.
- **Public API** (minimum set implementers should target):
   - `publish_artifact(*, local_path: Path, artifact_type: str, metadata: dict[str, Any], run_id: str | None = None) -> ArtifactRef` — uploads a file/dir to ADK ArtifactService (or file:// fallback when SDK missing), returning canonical URIs. Raises `ArtifactPublishError` when ADK rejects the upload; surfaces fallback paths explicitly in the returned metadata (`storage='local'|'adk'`).
   - `publish_local(*, payload: bytes | str, suffix: str, metadata: dict[str, Any] | None = None) -> ArtifactRef` — deterministic helper for fixture/unit tests; stores bytes under `runs/<run_id>/` with stable names and records `metadata['fixture']=True` for traceability.
   - `write_memory_event(run_id: str, event_type: str, payload: Mapping[str, Any], *, ts: datetime | None = None) -> None` — appends structured telemetry to ADK MemoryService (or SQLite fallback). Raises `MemoryWriteError` on failure and never swallows exceptions; callers may catch/log but must not ignore errors silently.
   - `request_human_input(*, run_id: str, reason: str, artifact_uri: str | None, metadata: dict[str, Any]) -> str` — wraps ADK review queue creation so QA/production agents do not need to stitch custom JSON; returns review task ID and raises `HumanInputRequestError` if the platform rejects the request.
   - `ensure_schema_artifacts(schema_config_path: Path) -> SchemaRegistry` — loads `configs/schema_artifacts.yaml`, validates URIs, and provides accessors used by ScriptAgent and production_agent to fetch schema versions.
- **Failure behavior**: each helper raises a domain error (`ArtifactPublishError`, `MemoryWriteError`, `HumanInputRequestError`, `SchemaRegistryError`) with machine-readable context (`{"run_id": ..., "event_type": ...}`) so agents can attach the exception payload to ADK memory logs. When falling back to local fixture mode, helpers must emit a warning-level `write_memory_event` stating that data was saved outside ArtifactService.
- **Testing hooks**: module exposes `set_backend(overrides: HelperBackend) -> ContextManager` so unit tests can inject fakes (for example, in-memory artifact storage). IMPLEMENTATION_TASKS references `tests/unit/test_adk_helpers.py`; this spec clarifies where those tests should target.

---

## Stage-by-stage plan (hosted / local FunctionTools)

1. ScriptAgent (ADK LlmAgent)
   - Input: idea + constraints; Output: validated `MoviePlan` artifact + memory
     timeline entry. ScriptAgent must validate against the canonical MoviePlan
     schema artifact and persist raw LLM output for audit.

2. Images (SDXL FunctionTool)
   - Stage: `images_stage` (orchestration layer). Adapter: `images_sdxl` (Diffusers)
   - Requirements: batching rules, token-bucket rate-limiter, pre-render QA
     (text moderation or `qa_qwen2vl` sample check), deterministic stub for
     unit tests, and `gpu_utils.model_context` for pipeline loads.

   **Rate limiter status (single-user deferral)** – The pipeline is expressly
   single-user/single-job today, so we are _not_ building the token-bucket +
   queue semantics during this rollout. Keep the spec below for the future
   multi-tenant milestone; this paragraph is the canonical deferral note.

   Concrete API & types (implementers)

   - `ImagesOpts` (put in `src/sparkle_motion/types.py`):

   ```python
   from typing import Optional, Mapping, TypedDict, Literal

   class ImagesOpts(TypedDict, total=False):
      seed: Optional[int]
      count: int
      max_images_per_call: int
      per_batch_timeout_s: int
      dedupe: bool
      qa: bool
      priority: Literal['low','normal','high']
      negative_prompt: Optional[str]
      metadata: Mapping[str, str]
   ```

   - `ArtifactRef` (returned by `images_stage.render`) — minimal shape:
   ```py
   {
    'uri': str,               # canonical URI (artifact:// or file://)
    'metadata': dict,         # includes seed, prompt, width/height
    'deduped': bool,          # True if resolved to existing canonical
   }
   ```

   - `images_stage` signature (`src/sparkle_motion/images_stage.py`):
   ```py
   def render(prompt: str, opts: ImagesOpts) -> list[dict]:
      """Render images and return ordered list of ArtifactRef dicts."""
   ```

   RateLimiter interface (`src/sparkle_motion/ratelimit.py`):
   ```py
   class RateLimiter:
      def acquire(self, tokens: int = 1) -> None: ...
      def release(self, tokens: int = 1) -> None: ...

   class TokenBucketRateLimiter(RateLimiter):
      def __init__(self, capacity: int, refill_rate_per_sec: float): ...
   ```
   Behavior & defaults:
   - Maintain per-tenant and per-host buckets; default capacity `60` with `refill_rate_per_sec = 1` (60 tokens/min). Each adapter call spends `count` tokens so large batches deplete faster.
   - Support `queue_allowed` semantics: when tokens are exhausted and queueing is enabled, enqueue the batch with TTL (default 600s) and surface a `StepExecutionRecord(status="queued")` until execution resumes; otherwise raise `RateLimitError` immediately.
   - Provide `tests/unit/test_rate_limiter.py` to cover leak-free acquire/release, burst behavior, queue drain, and clock-skew handling (use a fake clock to keep tests deterministic).
   - Instrument limit hits via `adk_helpers.write_memory_event()` so production_agent can surface global backpressure in dashboards.

   QA contract (stub): `function_tools/qa_qwen2vl/entrypoint.py`:
   ```py
   def inspect_frames(frames: list[bytes], prompts: list[str]) -> dict:
      # returns {'ok': bool, 'reason': Optional[str], 'report': dict}
   ```

   Deduplication flow (recommended):
   1. Within-plan dedupe: before calling the adapter, compute deterministic
     keys for each planned image (prompt+seed+index) and collapse identical
     calls so adapters are not invoked redundantly.
   2. Post-adapter dedupe: after receiving image bytes, compute pHash/sha256
     and consult `RecentIndex` (SQLite). If a canonical exists, mark the
     result `deduped=True` and return the canonical URI instead of a newly
     published artifact.

   Error classes to add under `src/sparkle_motion/errors.py`:
   ```py
   class PlanPolicyViolation(RuntimeError): ...
   class PlanResourceError(RuntimeError): ...
   class ModelOOMError(RuntimeError): ...
   class AdapterError(RuntimeError): ...
   ```

   DB schema location: create `db/schema/recent_index.sql` (see `docs/ARCHITECTURE.md`)
   and a DB helper `src/sparkle_motion/db/sqlite.py` exposing `get_conn(path)` and
   `ensure_schema(conn)`.
   Canonical DDL (`docs/ARCHITECTURE.md` excerpt — copy verbatim into `db/schema/recent_index.sql`):

   ```sql
   CREATE TABLE IF NOT EXISTS recent_index (
   	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	phash TEXT NOT NULL UNIQUE,
   	canonical_uri TEXT NOT NULL,
   	last_seen INTEGER NOT NULL,
   	hit_count INTEGER NOT NULL DEFAULT 1
   );
   CREATE INDEX IF NOT EXISTS ix_recent_index_phash ON recent_index(phash);

   CREATE TABLE IF NOT EXISTS memory_events (
   	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	run_id TEXT,
   	timestamp INTEGER NOT NULL,
   	event_type TEXT NOT NULL,
   	payload TEXT NOT NULL
   );
   CREATE INDEX IF NOT EXISTS ix_memory_events_runid ON memory_events(run_id);
   ```

   Helper API (`src/sparkle_motion/utils/recent_index_sqlite.py`):
   - `get_canonical(phash: str) -> Optional[str]`
   - `add_or_get(phash: str, uri: str) -> str`
   - `touch(phash: str, uri: str) -> None`
   - `prune(max_age_s: int, max_entries: int) -> None`

3. Videos (Wan FunctionTool)
    - Agent/Adapter: `videos_stage` orchestrates Wan2.1 (`videos_wan`) adapters. Public API:

       ```python
       def render_video(start_frames: Iterable[Frame], end_frames: Iterable[Frame], prompt: str, opts: dict) -> ArtifactRef:
             ...
       ```

    - Chunking + reassembly: default `chunk_length_frames=64`, `chunk_overlap_frames=4`, `min_chunk_frames=8`. Compute overlapping chunks, track `chunk_index`, and deterministically trim/crossfade overlaps on reassembly. Derive per-chunk seeds via `hash(seed, chunk_index)` for reproducibility.
    - Multi-GPU presets: document balanced `device_map` + `max_memory` (e.g., `{0:"74GiB",1:"74GiB","cpu":"120GiB"}`) for A100 hosts, auto/single-device modes for A6000, sequential/offload for 4090/3090, and CPU fallback for fixture runs. Surface these knobs in `opts` and record chosen map in artifact metadata.
    - OOM strategy: retry transient errors up to `max_retries_per_chunk=2`, shrink chunk length by `adaptive_shrink_factor=0.5` on OOM, then fall back to alternate GPU or CPU. Every attempt records `attempt_index`, `device`, `chunk_length_frames`, and outcome in telemetry.
    - Progress contract: adapters emit `CallbackEvent` (`plan_id`, `step_id`, `chunk_index`, `frame_index`, `progress`, `eta_s`, `phase`). `videos_stage` forwards events through `on_progress` callbacks and via `adk_helpers.write_memory_event()`.
    - Tests: `tests/unit/test_videos_stage.py` covers chunk split/reassembly, adaptive OOM retry, CPU fallback metadata, and progress forwarding. Gated smoke test (`SMOKE_ADK=1` + `SMOKE_ADAPTERS=1`) renders a short Wan2.1 job and asserts artifact publish + progress events.
      - Wan2.1 pilot deliverables (from `docs/IMPLEMENTATION_TASKS.md`):
         - Adapter loads `Wan-AI/Wan2.1-FLF2V-14B-720P` (and siblings) via `WanImageToVideoPipeline` inside `gpu_utils.model_context("wan2.1", weights=MODEL_ID, offload=True, xformers=True)`; enforce cleanup (`del pipeline; torch.cuda.empty_cache(); gc.collect()`).
         - Ship multi-host presets: balanced sharding for dual A100s (`max_memory = {0: "74GiB", 1: "74GiB", "cpu": "120GiB"}`), sequential/offload modes for single 4090/A6000, and CPU/fixture fallback for CI; surface chosen `device_map`/`max_memory` in artifact metadata.
         - Provide helper functions mirroring Wan notebooks (`aspect_ratio_resize`, `center_crop_resize`, `export_to_video` fallback) and deterministic per-chunk seeding (`hash(seed, chunk_index)`), plus callback plumbing for ETA/progress.
         - Publish artifacts through `adk_helpers.publish_artifact()` with metadata (`model_id`, `seed`, `device`, `inference_time_s`, `peak_vram_mb`, `plan_step`) and gate real runs behind `SMOKE_ADK=1`/`SMOKE_ADAPTERS=1`.
         - Test plan: unit tests stub `WanImageToVideoPipeline`, smoke test renders a tiny FLF2V plan (<=8 frames) using approved hardware, and telemetry assertions ensure attempt metadata + adaptive OOM shrink path behave as documented.

4. TTS (Chatterbox FunctionTool)
    - Agent/Adapter: `tts_stage` decision layer plus `tts_chatterbox` adapter (Resemble AI Chatterbox + Chatterbox-Multilingual). Public API:

       ```python
       def synthesize(text: str, voice_config: dict) -> ArtifactRef:
             """Return WAV ArtifactRef with metadata (voice_id, model_id, device, watermarked)."""
       ```

    - Dependencies: upstream repo `https://github.com/resemble-ai/chatterbox`, Hugging Face weights `ResembleAI/chatterbox`, PyPI package `chatterbox-tts`. Python 3.11 is required; all manifest changes go through `proposals/pyproject_adk.diff`.
    - Adapter behavior: instantiate `ChatterboxTTS.from_pretrained(device)` (cuda/cpu), expose `audio_prompt_path`, `language_id`, `cfg_weight`, `exaggeration`, and enforce watermark metadata. When `SMOKE_TTS!=1`, prefer fixture mode to avoid heavy installs.
    - Metadata + policy: publish `artifact_uri`, `duration_s`, `sample_rate`, `voice_id`, `model_id`, `device`, `synth_time_s`, `watermarked`. Enforce policy/content moderation in `tts_stage` (reject harmful text, log via `adk_helpers.write_memory_event()`).
    - Tests: unit tests stub `.generate()` to verify retries/backoff, metadata, and watermark flag handling. Integration smoke gated by `SMOKE_TTS=1` validates real synthesis + watermark awareness.

5. Lipsync (Wav2Lip FunctionTool)
    - Adapter API:

       ```python
       def run_wav2lip(face_video: Path | str, audio: Path | str, out_path: Path | str, *, opts: dict | None = None) -> dict:
             """Return metadata with artifact_uri, duration_s, checkpoint, face_detector, opts, logs."""
       ```

    - Options: `checkpoint_path`, `face_det_checkpoint`, `pads`, `resize_factor`, `nosmooth`, `crop`, `fps`, `gpu`, `verbose`. Require `ffmpeg` and upstream weights (validate/download before run).
    - Implementation notes: prefer in-process API when Wav2Lip Python modules are installed; supply subprocess fallback pinned to upstream commit using a safe `run_command()` helper (kill process groups on timeout, capture logs, retries). Support CUDA + CPU fallback; seed torch when supported for deterministic tests.
    - Security/licensing: upstream is research/non-commercial; enforce policy + sanitized paths and forbid arbitrary shell injection.
    - Tests: unit tests mock inference/subprocess to assert command generation, metadata, retries, and failure cleanup. Gated smoke (`SMOKE_LIPSYNC=1`) runs a short clip to confirm ffmpeg pipeline.

6. Assemble (ffmpeg FunctionTool)
    - Responsibilities: deterministically assemble clips/audio/subtitles via `ffmpeg` using safe subprocess helpers. Public API:

       ```python
       def assemble_clips(movie_plan: dict, clips: list[dict], audio: Optional[dict], out_path: Path, opts: dict) -> ArtifactRef:
             ...

       def run_command(cmd: list[str], cwd: Path, timeout_s: int = 600, retries: int = 1) -> SubprocessResult:
             ...
       ```

    - Implementation checklist: translate plan into concat/filter_complex graphs, manage per-run temp dirs, pin encode args (`libx264`, CRF, pixel format), capture stdout/stderr, enforce timeouts, and publish artifacts with `ffmpeg_version`, command, opts, logs URIs.
    - Determinism: record codec, CRF, seed (if transitions random), OS/ffmpeg versions. Reject arbitrary flags; map structured options only. Ensure cleanup yet retain logs when `debug=True`.
    - Tests: unit tests for command construction and retry behavior (mock `run_command`), assembly unit test with synthetic clips, and gated smoke (`SMOKE_ADK=1` or `SMOKE_ASSEMBLE=1`) that runs ffmpeg on short fixtures.

7. QA (Qwen-2-VL FunctionTool)
    - Adapter: `qa_qwen2vl.inspect_frames(frames, prompts, *, opts=None) -> QAReport`. Canonical model `Qwen/Qwen2-VL-7B-Instruct` via `transformers` + `qwen-vl-utils` (`process_vision_info`).
    - Behavior: batch frames + prompts into chat-template messages, run multimodal QA under `gpu_utils.model_context`, and emit structured reports (`per_frame`, `global_flags`, `confidence_summary`, `artifact_uri`, `logs_uri`). Surface knobs for `min_pixels`, `max_pixels`, `resized_height/width`, `max_new_tokens`, `attn_implementation`, `torch_dtype`, `device_map`.
    - Policy integration: map model responses to policy actions (approve/escalate/reject). On escalation call `request_human_input()` and log via `adk_helpers.write_memory_event()`. Provide lightweight stub for unit tests; gate real model runs behind `SMOKE_QA=1`/`SMOKE_ADK=1`.
    - Tests: unit tests stub model/processor to verify QAReport shape, thresholding, and prompt packaging; integration smoke (gated) runs a small frame set through the real model.
      - QA enforcement flow (authoritative thresholds):
         - Pre-check prompts/reference frames via text moderation or lightweight vision QA. If prompt flagged `disallowed`, raise `PlanPolicyViolation`; if `ambiguous`, mark the step for post-run escalation and limit generation parameters.
         - Post-render: invoke `qa_qwen2vl.inspect_frames` on every artifact before publish. Interpret `QAReport.per_frame` confidences using defaults: `nsfw_confidence >= 0.70` → reject + delete artifact, `0.40–0.69` → quarantine and call `request_human_input()`, `<0.40` → approve/publish.
         - Always attach the serialized `QAReport` to `adk_helpers.write_memory_event()` payloads so production_agent can reconstruct the policy trail. Escalations must persist quarantined artifact URIs, reviewer assignments, and `QAReport.global_flags` for ADK review console parity.
         - Smoke tests with `SMOKE_QA=1` must verify the escalation path by forcing synthetic outputs that cross the threshold and asserting human-input events fire.

---

## Operational blueprint (aligned with `docs/ARCHITECTURE.md`)

### Layered system view
- **Interaction**: Colab notebooks and the CLI call ADK `SessionService` to create sessions and drive runs. They stay thin; ADK owns session + artifact state.
- **Reasoning**: `script_agent` is an ADK `LlmAgent` that turns ideas into MoviePlans. Each FunctionTool process instantiates its own agent via `adk_factory` per Option A.
- **Coordination**: The WorkflowAgent YAML encodes the stage graph, retry rules, and hand-offs; we deploy/run it through the ADK control plane.
- **Execution**: Each hosted stage is a FunctionTool served as a local FastAPI/in-process server (Colab profile). FunctionTools handle heavy compute and publish artifacts via ADK.
- **Persistence**: ADK ArtifactService + MemoryService remain the source of truth; local folders are convenience caches only.

### Service & tool wiring
- **SessionService**: points at SQLite (e.g., `/content/sparkle_session.db`) for the Colab profile; every run creates a managed session and reuses it for resume flows.
- **ArtifactService**: artifacts land in Drive-mounted buckets (e.g., `/content/drive/MyDrive/sparkle_artifacts`). Local fallback roots like `artifacts/` are dev-only and never replace ADK publishes.
- **MemoryService**: backed by SQLite (e.g., `/content/sparkle_memory.db`) so all agents can append/query timelines without bespoke log parsing.
- **Duplicate detection**: RecentIndex lives in SQLite (see canonical DDL above) with helper APIs `get_canonical`, `add_or_get`, `touch`, `prune`. `SPARKLE_DB_PATH` controls the location; enable `PRAGMA journal_mode=WAL` when concurrent writers appear.
- **Tool catalog**: each FunctionTool is published with metadata (IAM scopes, cost hints). WorkflowAgent binds to tool IDs, keeping deployments consistent across environments.

### Workflow walk-through
1. **Session bootstrap** – `SessionService.create_session` issues `run_id`, stores the idea payload, and pre-allocates artifact prefixes (mounted via `adk mount`).
2. **Script planning** – ScriptAgent loads schema URIs from `configs/schema_artifacts.yaml` and publishes the validated MoviePlan plus a memory event.
3. **WorkflowAgent orchestration** – WorkflowAgent runs the `script → images → videos → ... → qa` graph, emits manifest events, and supports resume via `start_stage` overrides.
4. **FunctionTool executions** – Hosted tools listen on `127.0.0.1:<port>` inside Colab, accept artifact IDs, emit telemetry, and publish new ArtifactRefs.
5. **QA & human gating** – `qa_qwen2vl` emits QAReport artifacts + policy decisions. Escalations use `event_actions.request_human_input` and block the workflow until reviewers respond.
6. **Observability** – Operators inspect ADK timelines/metrics and local logs; telemetry export stays disabled, but `run_events.json` can be generated from ADK data when needed.

### Notebook production dashboard (Colab control cell)
- Reuse the ipywidgets-based control-panel pattern established for `script_agent`, but expand it to surface production runs end-to-end.
- Poll `GET /status?run_id=` every 3–5 seconds (or upgrade to SSE later) to display the current stage, elapsed time, percent complete, and a rolling log of `StepExecutionRecord` entries.
- Query `GET /artifacts?run_id=&stage=` as each stage completes so the UI can immediately render base-image thumbnails (`widgets.Image`), per-line TTS clips (`widgets.Audio`), per-shot MP4 previews (`IPython.display.Video`), and final assembly outputs without leaving Python. Sample cells should pin `stage="dialogue_audio"` when demonstrating the dialogue/TTS view so readers instantly see the stitched `tts_timeline.wav` manifest row alongside the per-line entries.
- Surface the shot-scoped manifest rows produced by `production_agent`: `shot_frames`, `shot_dialogue_audio`, `shot_video`, `shot_lipsync_video`, `shot_qa_base_images`, `shot_qa_video`, and `assembly_plan`. Dashboards should render these entries in the stage accordion so operators can see which asset exists, what QA decision was taken (pass/fail/skipped), and the on-disk JSON summary (`qa/qa_base_images/<shot>.json`, etc.) before choosing `resume_from`. Treat the `shot_qa_*` rows as the single source of truth for QA escalations and resume gating.
- Wire “Start production”, “Pause”, “Resume”, and “Stop” buttons to the new control endpoints; disable/enable buttons based on the latest status payload and show confirmation banners when a pause/stop takes effect.
- Present an accordion or tabbed asset gallery so users can inspect artifacts per stage without waiting for the full pipeline, plus an alert banner that highlights failed/stopped states and offers a `resume_from` action.
- Require the `qa_publish` stage to publish an `/artifacts` entry with `artifact_type="video_final"`, `artifact_uri`, `local_path`, and (when remote-only) a signed `download_url`. The notebook must expose a “Final Video” control cell that calls `/artifacts`, embeds the MP4 inline, and invokes `google.colab.files.download()` (or `adk artifacts download`) so users can save the deliverable immediately after QA approval.

### Final deliverable manifest schema

Every `/artifacts` response MUST describe the final movie via a single manifest entry so notebooks, CLIs, and future dashboards never guess about download formats. Production_agent owns the contract below and rejects publishes that fail validation.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `artifact_type` | Literal `"video_final"` | yes | Guards client routing; reject unknown literals. |
| `artifact_uri` | string (`artifact://...`) | yes | Canonical ADK ArtifactService URI for long-term storage. |
| `local_path` | string (absolute) | yes | Path inside the Colab runtime where the MP4 already exists; used for inline playback without re-download. |
| `download_url` | string (HTTPS) | conditional | Signed URL for remote-only runs; optional when `storage_hint="local"`. |
| `storage_hint` | enum `{"adk","local"}` | yes | Tells the notebook whether it can stream from disk or must fetch via `download_url`. |
| `mime_type` | string | yes | Default `video/mp4`; future-proofs alternate containers. |
| `size_bytes` | integer | yes | File size for progress bars/quota tracking. |
| `duration_s` | float | yes | Runtime of the final MP4; must equal stitched audio/video timeline. |
| `frame_rate` | float | yes | Frames per second derived from assemble stage metadata. |
| `resolution_px` | string (`"1280x720"`) | yes | Width x height; notebooks parse for layout hints. |
| `checksum_sha256` | string | yes | Hex digest of the MP4 so clients can validate downloads. |
| `qa_report_uri` | string | yes | Points at the QA artifact that approved the final video. |
| `qa_passed` | boolean | yes | True iff QA probe sequence approved the asset; false when publish halted. |
| `qa_mode` | string (`"full"|"skip"`) | yes | Mirrors run metadata so dashboards badge non-validated runs. |
| `run_id` | string | yes | ADK run/session identifier for traceability. |
| `stage_id` | string | yes | Should be `"qa_publish"`; used when multiple stages emit manifests. |
| `created_at` | ISO 8601 string | yes | Timestamp when the manifest row was generated. |
| `playback_ready` | boolean | yes | Indicates whether the MP4 already lives on the notebook filesystem and can be embedded immediately. |
| `notes` | string | optional | Human-readable extras (e.g., render profile name, retry counts). |

**Validation rules**

1. `artifact_type` MUST equal `video_final`; any other literal is rejected before response serialization.
2. `download_url` is required when `storage_hint="adk"` and omitted otherwise.
3. `checksum_sha256` must be a 64-character lowercase hex string; clients validate before persisting downloads.
4. `resolution_px` uses `"{width}x{height}"` with integer dimensions; pipeline ensures numbers match assemble metadata.
5. `duration_s`, `frame_rate`, and `size_bytes` must be positive.

**Sample manifest row**

```json
{
   "artifact_type": "video_final",
   "artifact_uri": "artifact://sparkle-motion/video_final/run-42/v3",
   "local_path": "/content/sparkle_motion/artifacts/run-42/video_final.mp4",
   "download_url": "https://signed.example.com/run-42/video_final.mp4?sig=...",
   "storage_hint": "local",
   "mime_type": "video/mp4",
   "size_bytes": 187532144,
   "duration_s": 96.04,
   "frame_rate": 24.0,
   "resolution_px": "1280x720",
   "checksum_sha256": "6e8c0d4a29c21a0f236f993b6db481d0ab2adc54a51091be9b2b6ec9efb95275",
   "qa_report_uri": "artifact://sparkle-motion/qa_reports/run-42/video_final",
   "qa_passed": true,
   "qa_mode": "full",
   "run_id": "run-42",
   "stage_id": "qa_publish",
   "created_at": "2025-11-29T04:12:55Z",
   "playback_ready": true,
   "notes": "render_profile=wan-2.1-default"
}
```

Clients should treat any missing `video_final` row as a terminal error and display guidance (e.g., “QA publish not finished — rerun the stage”).

### Data contracts & storage layout
- **MoviePlan / AssetRefs / QAReport / StageEvent / Checkpoint** schemas are published as ADK artifacts, with local fallbacks referenced via `sparkle_motion.schema_registry`.
- **Schema registry reference** – see `docs/SCHEMA_ARTIFACTS.md` for the
   table of canonical URIs, versions, and fallback paths extracted from
   `configs/schema_artifacts.yaml`. Update both sources together when schemas or
   policy bundles change.
- **Artifact storage**: IDs follow `artifact://sparkle-motion/<type>/<run>/<version>`; local scratch copies live under `runs/<run_id>` for inspection only.
- **Memory logs**: long-lived QA decisions, human approvals, and stage failures are stored via MemoryService so agents can replay context without filesystem scraping.

### Enforced ADK usage & short-term decisions
- Application/runtime code MUST import the real `google.adk` SDK in entrypoints; per-tool agents are mandatory under Option A (no silent fallbacks).
- `adk_factory` provides guarded probing yet defaults to per-tool construction; shared mode is documented but not enabled for this rollout.
- `gpu_utils.model_context` is required for every heavy adapter and must emit telemetry + normalized `ModelOOMError` exceptions.
- Manifest changes (torch, diffusers, ffmpeg, chatterbox, etc.) stay proposal-only via `proposals/pyproject_adk.diff` until explicitly approved.
- FunctionTool directories must be normalized (case duplicates removed) and each tool must provide deterministic smoke tests gated by the relevant `SMOKE_*` flags.

### Human + QA governance hooks
- Script review, per-shot approvals, and QA escalations all use ADK’s review queue (`request_human_input`). Memory events must capture reviewer assignments and QAReport URIs for audit.
- New stages plug into the WorkflowAgent graph and inherit QA/human gating declaratively; no bespoke runner work is needed beyond registering the tool and updating the workflow spec.

### Operational summary
- Single-user Colab runtime hosts WorkflowAgent definitions, tool catalog, and artifact buckets under the `local-colab` profile.
- Resume/retry is an ADK API call (`workflow_runs.resume(run_id, start_stage=...)`); no filesystem surgery required.
- Artifact access goes through `adk artifacts download` or signed URLs; local exports are optional.
- Human + QA governance remain auditable inside ADK; documentation parity across `THE_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/ORCHESTRATOR.md` is required before coding resumes.

---

## Cross-cutting implementation rules and notes

- Agents must be small, testable, and segregate policy decisions from heavy
   compute. Agents orchestrate adapters (FunctionTools); adapters are plain,
   testable callables that use `gpu_utils.model_context`.
 - Deterministic test harnesses: provide stubbed pipelines that produce
   deterministic artifacts (seed-based PNGs, predictable pHash values).
 - Duplicate detection: use perceptual hashing (pHash) and a simple LRU cache
   or an SQLite-backed `RecentIndex` to dedupe recent artifacts; `images_stage`
   should support `dedupe=True` semantics and persist recent-index state to
   SQLite for the single-user workflow. Do not rely on Redis — this project is
   explicitly single-user and uses SQLite as the canonical lightweight store.
- OOM handling: adapters should normalize OOMs to `ModelOOMError` with
   structured metadata (stage, suggested_shrink_hint, memory_snapshots).

### `gpu_utils.model_context` (canonical API & checklist)

- Signature (copy into `src/sparkle_motion/gpu_utils.py`):

   ```python
   @contextmanager
   def model_context(
         model_key: str,
         *,
         weights: str | None = None,
         offload: bool = True,
         xformers: bool = True,
         compile: bool = False,
         device_map: Mapping[str, str] | None = None,
         low_cpu_mem_usage: bool = True,
         max_memory: Mapping[str, str] | None = None,
         timeout_s: int | None = None,
   ) -> Iterator[ModelContext]:
         ...
   ```

- Parameter semantics: `model_key` is the telemetry identifier, `weights` selects the HF repo or local path, `device_map`/`max_memory` let adapters pass multi-GPU presets, and `timeout_s` enforces load deadlines. Provide defaults for SDXL (`sdxl/base-refiner`) and Wan (`wan2.1/flf2v`).
- `ModelContext` returned by the manager should expose `pipeline`, `device_map`, `allocated_devices`, and `report_memory()` so adapters can emit consistent telemetry snapshots.
- Cleanup checklist on exit (all adapters must rely on this): delete large references (`del pipeline`), `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`, `gc.collect()`, release offload helpers, and emit a final `report_memory()` snapshot into `adk_helpers.write_memory_event()` with `step="cleanup"` metadata.
   1. **Session bootstrap** – `SessionService.create_session` issues `run_id`, stores the idea payload, and pre-allocates artifact prefixes (mounted via `adk mount`).
   2. **Script planning** – ScriptAgent loads schema URIs from `configs/schema_artifacts.yaml` and publishes the validated MoviePlan plus a memory event.
   3. **WorkflowAgent orchestration** – WorkflowAgent runs the `script → images → videos → ... → qa` graph, emits manifest events, and supports resume via `start_stage` overrides.
   4. **FunctionTool executions** – Hosted tools listen on `127.0.0.1:<port>` inside Colab, accept artifact IDs, emit telemetry, and publish new ArtifactRefs.
   5. **QA & human gating** – `qa_qwen2vl` emits QAReport artifacts + policy decisions. Escalations use `event_actions.request_human_input` and block the workflow until reviewers respond.
   6. **Observability** – Operators inspect ADK timelines/metrics and local logs; telemetry export stays disabled, but `run_events.json` can be generated from ADK data when needed.
- OOM normalization: wrap CUDA `RuntimeError` messages containing "out of memory" in `ModelOOMError(model_key=..., stage='load'|'inference', attempted_device_map=..., peak_vram_mb=...)` so callers can adaptively shrink (`adaptive_shrink_factor=0.5`) or fall back to CPU.
- Telemetry: emit events at `load_start`, `load_complete`, `inference_start`, `inference_end`, and `cleanup`. Each payload should include `plan_id`, `step_id`, `device_stats`, `max_memory`, `generator_seed` (when provided), and `attempt_index` so production_agent dashboards stay consistent.
- Testing expectations: add `tests/unit/test_gpu_model_context.py` that fakes torch/nvml calls, asserts cleanup order (even on exceptions), and verifies OOM normalization; gated smoke (`SMOKE_ADK=1`) should instantiate a tiny diffusers model via this context to ensure telemetry wiring is correct.

---

## Governance & sign-off (must be completed before code-phase)

Before changing manifests, adding large wheels, or committing runtime binary
requirements you must approve a proposal. The implementation phase will not
begin until the following checklist is signed:

- [ ] You confirm Option A (per-tool agents) and the canonical tool list above.
- [ ] Resource sizing & rollout wave owners are approved (see `docs/ROLLOUT_PLAN.md`).
- [ ] Dependency proposal (`proposals/pyproject_adk.diff`) approved for review.
- [ ] Security checklist accepted (`docs/SECURITY_CHECKLIST.md`) or owner assigned.

Reply `approve-docs` to approve these document changes and permit the next
code-phase, or `revise-docs` to request edits to this plan.

---

## Immediate next steps (practical)

1. Finalize and publish canonical schema artifacts (MoviePlan, AssetRefs,
   QAReport) and ensure `configs/schema_artifacts.yaml` points to artifact URIs.
2. Implement a minimal `gpu_utils.model_context` in `src/sparkle_motion/gpu_utils.py`
   (a light, dependency-free implementation suitable for unit tests) and the
   `ModelOOMError` domain exception.
3. Create deterministic stub harnesses for `images_stage` and `videos_stage`
   unit tests to validate batching, dedupe, chunking, and OOM fallback logic.
4. Prepare `proposals/pyproject_adk.diff` that lists proposed runtime
   dependencies for review (do not apply until approved).

If you want, I can now scaffold one of these items locally (no manifest edits):
- Option A: `videos_stage` chunk-splitting utility + unit tests (deterministic)
- Option B: `tts_stage` voice registry helper + VoiceMetadata model and tests
- Option C: minimal `gpu_utils.model_context` implementation + tests

---

This document is now aligned to the authoritative architecture and the
implementation tasks; it is intended to be the single source of truth for the
planned rollout and the sign-off workflow.