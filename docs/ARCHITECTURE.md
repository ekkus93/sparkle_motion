# Sparkle Motion architecture — UPDATED

This file documents the authoritative, ADK-first architecture for Sparkle
Motion (updated 2025-11-27). It explicitly states the current runtime
counts, the per-process agent model, and the operational plan to finish
converting FunctionTools to ADK-native entrypoints. It has been aligned with
the implementation guidance in `docs/IMPLEMENTATION_TASKS.md` (2025-11-27):
that document defines per-tool contracts, the `production_agent` orchestration
semantics, explicit per-tool gating flags (SMOKE_*), and the cross-cutting
`gpu_utils.model_context` pattern that adapters must use for safe model
loads/unloads.

## Current runtime counts (authoritative)

- **ADK agents (application runtime, excluding `resources/` samples):** 7
	- Rationale: we are using Option A (per-tool agents). Each FunctionTool
		process that needs an LLM or an ADK-managed model instantiates its own
		`LlmAgent` at startup. After normalizing a case-duplicate scaffold
		(`ScriptAgent` / `script_agent`), the canonical set contains seven unique
		FunctionTools.
- **FunctionTools (application code under `function_tools/`):** 6 directories
	- Canonical tools: `script_agent`, `images_sdxl`, `videos_wan`, `tts_chatterbox`,
		`lipsync_wav2lip`, `assemble_ffmpeg`
	- Note: there is a case-duplicate pair `ScriptAgent` / `script_agent` in
		the tree; the duplicate scaffold will be removed during normalization.

Implementation tasks alignment: `docs/IMPLEMENTATION_TASKS.md` is the
authoritative per-tool TODO reference for implementers. Important clarifications
introduced there (now reflected in this architecture) include:

- Agents (policy/orchestration) vs FunctionTools (compute adapters): Agents
	include `script_agent` (plan generation) and `production_agent` (plan
	execution/orchestration). Stage adapters (`images_stage`, `tts_stage`,
	`videos_stage`) now encapsulate the FunctionTool shims that used to carry
	the `_agent` suffix, while FunctionTools such as `images_sdxl`, `videos_wan`,
	`tts_chatterbox`, `lipsync_wav2lip`, and `assemble_ffmpeg`
	remain the heavy compute adapters.
- Per-tool gating: heavy adapters and integration tests are gated by
	`SMOKE_*` flags (for example `SMOKE_ADK`, `SMOKE_TTS`, `SMOKE_LIPSYNC`,
	`SMOKE_ADAPTERS`) to avoid accidental heavy installs or model
	loads in CI/local runs.
- Model lifecycle: adapters must use the shared `gpu_utils.model_context`
	pattern to load/unload models, emit telemetry, and normalize OOMs to the
	`ModelOOMError` domain exception so higher-level agents can implement
	deterministic fallback/shrink strategies.

## Agent responsibilities (aligned with `docs/IMPLEMENTATION_TASKS.md`)

The architecture now mirrors the per-agent contracts introduced in the
Implementation Tasks document.

- **`script_agent`** – plan-only API `generate_plan(prompt: str) -> MoviePlan`.
	Uses `adk_factory.get_agent('script_agent', model_spec)` to construct the
	LLM, emits artifacts via `adk_helpers.publish_artifact()`, and records memory
	events. It must enforce JSON-only prompting, validate outputs against the
	MoviePlan schema, run content policy checks, and raise the documented error
	types (`PlanParseError`, `PlanSchemaError`, `PlanPolicyViolation`,
	`PlanResourceError`). Unit tests cover schema conformance and error cases;
	integration smoke tests run behind `SMOKE_ADK=1` with a real or fixture LLM.
- **`tts_stage`** – orchestrates synthesis via `synthesize(text, voice_config)
	-> ArtifactRef`. Responsibilities include provider selection (priority
	profiles for cost/latency/quality/balanced, `max_latency_s`,
	`max_cost_usd`, `required_features`), rate limiting, retries/backoff, and
	telemetry of score breakdowns. It manages voice metadata via
	`get_voice_metadata` / `list_available_voices`, propagates watermark flags,
	and differentiates retryable errors (`TTSRetryableError`,
	`TTSProviderUnavailable`, `TTSServerError`) from terminal ones
	(`TTSInvalidInputError`, `TTSPolicyViolation`, `TTSQuotaExceeded`). Real TTS
	invocations stay behind `SMOKE_TTS=1` and fall back to fixture providers in
	CI.
- **`images_stage`** – API `render(prompt, opts) -> list[ArtifactRef]` that
	performs text moderation, batching, rate limiting (token bucket, queueing
	plumbed through `queue_allowed`), timeout enforcement, and perceptual-dedupe
	(`RecentIndex`). The Stage 3 sunset removed the dedicated QA FunctionTool, so
	the stage now focuses on deterministic rendering, dedupe metadata, and policy
	telemetry while future QA replacements are evaluated separately. It chunks
	oversized requests using `max_images_per_call`, annotates artifacts with
	`batch_index`/`item_index`, and records memory events for traceability. Unit
	tests cover batching, dedupe, rate-limit queueing, and policy rejections.
- **`videos_stage`** – API `render_video(start_frames, end_frames, prompt,
	opts) -> ArtifactRef`. It implements chunking with parameters from the tasks
	doc (`chunk_length_frames=64`, `chunk_overlap_frames=4`,
	`min_chunk_frames=8`, adaptive retries, CPU fallback), handles multi-GPU
	device maps, and forwards progress via callback contracts / memory events.
	Unit tests validate chunk math, adaptive shrink-on-OOM, CPU fallback, and
	progress propagation; integration smokes stay behind `SMOKE_ADK=1` /
	`SMOKE_ADAPTERS=1`.
- **`production_agent`** – API `execute_plan(plan, mode='dry'|'run') ->
	list[ArtifactRef]`. `dry` mode simulates execution and returns a
	`simulate_execution_report` (resource estimates, simulated artifact URIs,
	policy warnings). `run` mode enforces gates (`SMOKE_*`,
	`adk_helpers.require_adk()`), orchestrates steps, applies bounded retries,
	and publishes final artifacts via `assemble_ffmpeg` followed by the `finalize`
	stage. Every step emits a `StepExecutionRecord` containing plan/step IDs,
	status, timings, attempt counts, device/model metadata, logs URIs, and
	policy/error annotations. Failures trigger cleanup (GPU context release,
	temp file handling) and dependent-step skips per the Implementation Tasks
	guidance. Plan intake now materializes a canonical `RunContext` that validates
	`MoviePlan` schemas, enforces `render_profile` selections (model IDs, FPS
	limits, provider locks), confirms base-image continuity (`len(base_images)
	== len(shots) + 1`), and stitches the dialogue timeline into the per-line
	TTS schedule before any heavy stage is invoked. Each subsequent stage records
	its required inputs/outputs inside the `StepExecutionRecord` (shot IDs,
	dialogue clips, artifact manifests) so notebooks can resume or inspect runs
	without rehydrating bespoke temp files. QA metadata is no longer emitted;
	future QA reintegration will add a dedicated stage on top of this flow.

### MoviePlan schema & validation recap

`MoviePlan` artifacts must include `id`, `title`, optional `description` /
`created_by`, ISO `created_at`, ordered `steps`, and optional `assemble_opts`.
Steps share `id`, `type`, `title`, `opts` and specialize into:

- `image`: `prompt`, `count`, `ImagesOpts` (width, height, seed, sampler,
	steps, cfg_scale, denoising window, prompt_2, `dedupe`, `priority`).
- `tts`: `text`, `voice`, `TTSOpts` (sample rate, format, provider hints).
- `video`: `prompt`, `start_frames`, `end_frames`, `VideoOpts` (frames,
	dimensions, steps).
- `lipsync`: `face_video`, `audio`, `out_path`, `Wav2LipOpts` (checkpoint,
	face-detector, pads, resize factor, nosmooth, gpu idx).
- `custom`: `handler` plus arbitrary `opts` passed to extensions.

Validation flow: parse JSON, enforce schema via `pydantic`/`jsonschema`, run
step-level validation, apply policy filters, and reject plans exceeding
resource caps (frames, clips, elapsed runtime). Errors map to the four
`Plan*` exception types described above.

### Rate limiting, smoke-test gating, and delivery status

- All heavy adapters expose smoke tests under `tests/smoke/...` and only run
	when their corresponding `SMOKE_*` flag is set.
- Rate limiting follows the token-bucket guidance from the Implementation
	Tasks: per-tenant/user keys, configurable refill, queue-or-fail semantics,
	and backlog guards to surface `PlanResourceError` when the queue is full.
- Rate limiter implementation status: because the runtime is explicitly
	single-user/single-job, we are **not** implementing token-bucket + queue
	mechanics yet. The spec remains documented for the future multi-tenant
	workstream, and this note serves as the authoritative deferral.
- QA automation is paused after the Stage 3 sunset; the architecture keeps the
	hooks documented in THE_PLAN so a future FunctionTool can slot back in without
	changing notebooks or production_agent APIs.
- Dedupe relies on the SQLite-backed `RecentIndex` APIs listed earlier; the
	same index powers cross-plan dedupe for `images_stage` and `videos_stage`
	(keyframes).

Note: the `resources/` directory contains many ADK sample projects and
examples. Those are vendor/sample code and are intentionally excluded from
these counts per your instruction — they are references, not application
runtime deployments.

### `_agent` naming matrix (authoritative as of 2025-12-03)

The table below documents the renaming outcome for every runtime component
that previously ended with `_agent`. Only the WorkflowAgent entrypoint
(`production_agent`) and the LlmAgent that generates MoviePlans
(`script_agent`) still carry the `_agent` suffix; every other former `_agent`
module has been renamed to a stage-specific adapter so users no longer confuse
FunctionTools with ADK agents.

| Legacy `_agent` | Current module(s) | Runtime reality | Canonical name | Notes |
| --- | --- | --- | --- | --- |
| `script_agent` | `src/sparkle_motion/script_agent.py`<br>`src/sparkle_motion/function_tools/script_agent/entrypoint.py` | ADK `LlmAgent` that produces MoviePlans and persists artifacts. | `script_agent` | Only `_agent` allowed to keep the suffix besides the WorkflowAgent; FunctionTool entrypoint merely hosts the agent for `/invoke`. |
| `production_agent` | `src/sparkle_motion/production_agent.py`<br>`src/sparkle_motion/function_tools/production_agent/entrypoint.py` | WorkflowAgent runtime orchestrator that executes validated MoviePlans. | `production_agent` | Coordinates every stage, emits `StepExecutionRecord` history, and needs to retain its public API surface. |
| `images_agent` | `src/sparkle_motion/images_stage.py` | FunctionTool shim around `function_tools/images_sdxl` plus rate-limit/dedupe helpers (no ADK agent). | `images_stage` | Renamed from `images_agent`; telemetry keys and RunRegistry rows use the `images_stage.*` prefix. |
| `videos_agent` | `src/sparkle_motion/videos_stage.py` | FunctionTool shim around `function_tools/videos_wan` with chunk/orchestration logic. | `videos_stage` | Renamed from `videos_agent`; emits Wan chunk progress + retries under `videos_stage.*` telemetry. |
| `tts_agent` | `src/sparkle_motion/tts_stage.py` | FunctionTool shim that routes to `function_tools/tts_chatterbox` adapters. | `tts_stage` | Renamed from `tts_agent`; handles provider selection, retries, and artifact stitching under the stage moniker. |

This matrix now drives the remaining P0 tasks in `docs/TODO.md`: verify that
tool registries, telemetry, and docs consistently reference the stage names so
only real ADK agents retain the `_agent` suffix.
- **WorkflowAgent as the coordinator** – stage orchestration, retries,
	resume, and hand-offs are modeled directly in ADK’s WorkflowAgent graph so
	runs never depend on a bespoke Python runner.

- **Production orchestration (`production_agent`)** – in addition to the
	WorkflowAgent stage graph, the implementation guidance introduces a
	runtime `production_agent` that executes validated `MoviePlan` objects.
	`production_agent` provides `dry` vs `run` semantics (simulate vs execute),
	enforces policy decisions, orchestrates per-step calls to agents and
	FunctionTools, and manages retries, progress events, and artifact
	publication. WorkflowAgent remains the declarative graph; `production_agent`
	is the runtime orchestrator for local/Colab execution flows.
- **Tool catalog + hosted runtimes** – each heavy stage is packaged and
	registered as an ADK FunctionTool/ToolRuntime deployment. Under Option A,
	each FunctionTool process owns its runtime agent and associated IAM and
	resource profile. We register local FastAPI/FastHTTP servers (or in-process
	callables) bound to `127.0.0.1` inside the Colab runtime, keeping metadata
	ready for future hosted swaps while remaining entirely local today.
- **Schema-first contracts** – MoviePlan, AssetRefs, and StageEvent/Checkpoint
	schemas are published as ADK artifacts so both agents and tools fetch the same
	canonical definitions at runtime. QAReport schemas remain archived for future
	use but are no longer referenced in the active pipeline.
- **Human governance hooks** – ADK’s `request_human_input` / review queue and
	policy evaluation APIs gate stages without custom JSON polling. With QA
	automation paused, these hooks now capture pause/resume acknowledgements and
	operator annotations so we can reintroduce QA without changing infrastructure.
- **Single-run friendly** – Colab notebooks and CLI flows still exist, but
	they are thin clients invoking ADK endpoints, which keeps the local runtime
	stateless.
- **Colab-first execution, API-ready** – all heavy models currently run on a
	single Google Colab A100 session (SDXL, Wan, Chatterbox TTS, etc.). Each stage
	remains a FunctionTool boundary, so we can later swap in external API
	providers (e.g., OpenAI GPT-5.1, ElevenLabs) by updating the tool metadata
	without reworking the workflow.
- **VRAM load discipline** – the A100 host cannot keep every FunctionTool’s
	weights in memory at once, so tools must page models in/out (and release CUDA
	contexts) as they acquire/release the GPU to avoid OOM kills during multi-stage
	runs.

Operational constraint: single-user, single-job pipeline

- **Single-job operation:** Sparkle Motion runs as a single-user pipeline: at
	any given time only one workflow/job is active in the system. Given the
	limited VRAM footprint available on a single A100, tools must assume they do
	not share model weights in memory with other stages. Each FunctionTool must
	load required model weights on-demand and release CUDA memory and any
	associated GPU contexts immediately after completing its stage. Implement
	explicit load/unload (or context-manager) hooks in tool entrypoints so the
	scheduler can safely page models in/out without leaking GPU resources.

	This single-job constraint simplifies concurrency control (no multi-job
	scheduling), but increases the importance of deterministic cleanup, model
	paging discipline, and careful driver-level error handling during weight
	load/unload cycles.

Note: the `videos_wan` stage (Wan 2.1 model) is currently the most VRAM/GPU
intensive stage in the workflow and therefore should be treated as the highest
risk pilot for rollout and resource-validation (see `THE_PLAN.md` rollout
sequence). Treat `videos_wan` as the primary pilot because it exercises the
largest GPU footprint and driver interactions.

Canonical model reference:

- **Model ID:** `Wan-AI/Wan2.1-I2V-14B-720P` (Hugging Face model card).
- **Provenance:** Wan-Video upstream repo: `https://github.com/Wan-Video/Wan2.1`.
- **Notes:** This model (commonly referenced as Wan 2.1 or WAN2.1) is the
	image-to-video (start-image → end-image / I2V) variant we use as the
	canonical reference for the `videos_wan` FunctionTool. It is the primary
	heavy-footprint model for pilot validation and therefore drives our VRAM,
	driver, and paging discipline guidance above.

- **Model card (Hugging Face):** `https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P`

  Citation: Wan-Video project (Wan 2.1). See the model card above for the
  canonical weights, usage notes, license, and community-provided examples.

## Layered system view (concise)

- Interaction: CLI / Colab notebook clients call ADK SessionService to create
	sessions and drive runs. These are thin clients; the ADK control plane owns
	session and artifact state.
- Reasoning: ScriptAgent (an ADK `LlmAgent`) is responsible for turning an
	idea into a MoviePlan artifact. The ScriptAgent is created per-process and
	is required for the Script FunctionTool entrypoint.
- Coordination: WorkflowAgent YAML encodes the stage graph and retry/human
	gating semantics; we deploy and run it through the ADK control plane.
- Execution: Each stage is a FunctionTool (local FastAPI/in-process server in
	`local-colab` profile). Each FunctionTool process should instantiate its
	ADK LlmAgent as needed (we currently instantiate ScriptAgent only) or
	obtain one via a shared factory (recommended next step).
- Persistence: ADK ArtifactService + MemoryService remain the sources of
	truth when `ARTIFACTS_BACKEND=adk`. When toggled to `filesystem`, the shim
	provides an ArtifactService-compatible surface backed by local disk while
	MemoryService stays on SQLite.

### Service & tool wiring

- **SessionService** – every run is created via ADK’s managed session service
	configured for the Colab-local profile. The adapter points at a SQLite
	database file (e.g., `/content/sparkle_session.db`). Session metadata, user
	info, and run state live there permanently; local folders mount as ephemeral
	caches only when a developer needs to inspect artifacts on Colab.
- **ArtifactService** – MoviePlan, AssetRefs, checkpoints, and final renders
	can now target **either** the canonical ADK ArtifactService
	(backed by Google Drive/GCS) **or** the new filesystem shim described below.
	When `ARTIFACTS_BACKEND=adk` we mount Google Drive
	(`/content/drive/MyDrive/sparkle_artifacts`) and treat it as the artifact
	root; tools receive Drive-backed handles and no bespoke file paths are
	passed outside ADK. When `ARTIFACTS_BACKEND=filesystem`, helpers talk to the
	shim and emit `artifact+fs://` URIs that resolve against a local directory
	tree while retaining the same manifest format.

- **Filesystem shim (local-only)** – single-user environments may run the
	lightweight shim that mirrors the ArtifactService contract. Payloads live
	under `ARTIFACTS_FS_ROOT` (default `./artifacts_fs/<run_id>/...`) and index
	rows live inside `ARTIFACTS_FS_INDEX` (SQLite). `/status` and `/artifacts`
	endpoints query the same manifest data regardless of backend, so UIs do not
	branch on storage. This path is the recommended way to run real models
	locally without provisioning Google Cloud resources.
- **MemoryService** – long-lived run logs, pause/resume intents, and human
	decisions are appended through ADK’s memory APIs backed by a SQLite file such
	as `/content/sparkle_memory.db` so any agent (ScriptAgent, production_agent,
	or WorkflowAgent) can query histories without parsing filesystem logs. Future
	QA stages will reuse the same channel for auditability.

 - **Duplicate detection / RecentIndex** – duplicate detection for images and
	short-lived artifacts uses perceptual hashing (pHash) and a small
	`RecentIndex` persisted to SQLite for the single-user workflow. Persisting
	canonical URIs and pHash values to a local SQLite table provides a
	lightweight, durable store that fits the Colab / single-user topology; this
	project does not use Redis for recent-index persistence.

	SQLite schema (recommended, place in `db/schema/recent_index.sql`):

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

	Recommended config / env var:
	- `SPARKLE_DB_PATH` (env): path to SQLite DB. Default for dev: `./artifacts/sparkle.db`.

	RecentIndex API (to implement under `src/sparkle_motion/utils/recent_index_sqlite.py`):
	- `get_canonical(phash: str) -> Optional[str]`
	- `add_or_get(phash: str, uri: str) -> str`  # returns canonical uri
	- `touch(phash: str, uri: str) -> None`
	- `prune(max_age_s: int, max_entries: int) -> None`

	Persistence notes:
	- Use `sqlite3.connect(SPARKLE_DB_PATH, timeout=5.0)` and set
		`PRAGMA journal_mode=WAL` if concurrent readers/writers are expected in
		development. For single-process Colab runs the default journal is fine.
	- Store JSON payloads as TEXT in `memory_events` and use `int(time.time())`
		for timestamps.
- **Tool catalog** – FunctionTools are published to the ADK catalog with
	metadata (capabilities, IAM scopes, cost hints). WorkflowAgent binds to
	tool IDs rather than importing Python modules, which keeps deployments
	consistent across revisions.

### Filesystem ArtifactService shim (design recap)

To unblock real-model runs without Google Cloud access, we layer a shim that
reimplements the subset of ArtifactService APIs our runtime depends on. This
shim runs in-process (or as a FastAPI microservice) inside the Colab session
and is selected via `ARTIFACTS_BACKEND=filesystem`.

- **Directory layout** – artifacts live under
	`${ARTIFACTS_FS_ROOT}/${run_id}/${stage}/${artifact_id}/...` with manifest
	JSON stored beside payloads. Binary blobs (images, WAVs, MP4s) keep their
	original extensions so notebook previews still work via `file://` mounts.
- **Index** – a SQLite file (`ARTIFACTS_FS_INDEX`, default
	`./artifacts_fs/index.db`) tracks `artifact_id`, `run_id`, stage, relative
	path, MIME, checksum, and created_at timestamps. `/artifacts` queries hit
	this index, mirroring ADK response schemas.
- **Manifests & URIs** – helpers emit `artifact+fs://<run_id>/<artifact_id>`
	values. UI code and `production_agent` resolve the URI by consulting the
	index + manifest data, so no caller needs to know the filesystem path.
- **API surface** – the shim exposes `POST /artifacts`, `GET /artifacts/<id>`,
	`GET /artifacts?run_id=...`, and health metadata used by `/status`. Payload
	uploads accept multipart/form-data (for large binaries) or JSON bodies (for
	manifests-only). Requests authenticate via a shared secret env var since this
	remains single-user.
- **Compatibility guarantees** – manifest JSON matches ADK’s schema, including
	finalize metadata, checksums, and download URLs (local `file://` paths in shim
	mode). `production_agent` and the Colab dashboards treat both backends
	identically because the helper API (`adk_helpers.publish_artifact()` and
	friends) normalizes responses.
- **Retention** – optional cleanup helpers prune artifacts older than `N`
	days or exceeding `MAX_BYTES`. This is a maintenance task for operators once
	runs are copied off the Colab VM or Drive.

This section codifies the architecture decisions from `THE_PLAN.md`; future
implementation docs must keep the env var names and API surface synchronized
with this contract.

## Workflow walk-through

1. **Session bootstrap** – A client (CLI/notebook/UI) calls ADK’s
	`SessionService.create_session`, which issues the `run_id`, stores the
	initial idea payload, and pre-allocates artifact prefixes in the Google Drive
	artifact root. Any local working directory simply mounts those artifacts via
	`adk mount`.
2. **Script planning (ScriptAgent)** – An ADK LlmAgent package hosts Qwen2.5-7B
or Gemini Nano and validates output against the MoviePlan schema artifact.
Prompts load schema/context directly from the artifact URIs managed in
`configs/schema_artifacts.yaml` (accessible via
`sparkle_motion.schema_registry`). The helper
`sparkle_motion.prompt_templates.build_script_agent_prompt_template()` (or the
CLI `scripts/render_script_agent_prompt.py`) assembles prompt metadata with the
canonical `artifact://sparkle-motion/schemas/movie_plan/v1` reference so ScriptAgent
deployments inherit the correct schema without copying JSON inline. The resulting
plan is saved as an ADK artifact version and written to the memory timeline for
provenance.
3. **WorkflowAgent orchestration** – The WorkflowAgent definition encodes the
	stage graph (`script -> images -> videos -> ... -> finalize`). It binds each
	step to a FunctionTool ID, specifies retry policies, and emits ADK manifest
	events automatically. Resume is a first-class capability: rerun requests
	simply invoke WorkflowAgent with a `start_stage` override.
4. **FunctionTool executions** – Each tool runs inside the Colab runtime as a
	lightweight HTTP server (or registered in-process callable) bound to
	`127.0.0.1:<port>`. Inputs reference artifact IDs, not file paths; outputs are
	new artifact versions (AssetRefs, clips, final deliveries). Tools publish
	structured telemetry (latency, cost, GPU usage) via ADK’s telemetry hooks.
5. **Final delivery** – With QA automation paused, the `finalize` stage
	publishes the `video_final` manifest row plus supporting metadata. Human
	review hooks remain wired via `request_human_input`, so when QA returns it can
	layer on top of the same stage boundary.
6. **Observability & run book** – ADK’s timeline + metrics are inspected via
	the CLI/notebook interfaces and local logs. Telemetry export is disabled, so
	operators rely on notebook logs and session metadata only. Local summaries
	(if desired) are generator outputs derived from the ADK data, not the other
	way around.

### Production run observability & controls

- `production_agent` exposes HTTP surfaces that mirror its ADK timeline so the
	Colab UI (and future CLIs) can introspect long-running jobs. `GET /status`
	requires a `run_id` and returns the latest `StepExecutionRecord` timeline,
	current stage, attempt counts, and pause/stop flags. Polling every 3–5 seconds
	keeps the on-notebook dashboard in sync without resorting to ad hoc log
	scraping.
- `GET /artifacts` accepts `run_id` (and optional `stage`/`step_id`) and emits
	structured asset manifests per stage: base-image thumbnails, dialogue WAVs,
	per-shot MP4s, and final assembly outputs. Assets always arrive as typed URIs
	(`artifact://` or `file://` in dev) so the UI can embed imagery/audio/video
	controls without guessing locations.
- The `finalize` stage must contribute a manifest entry with
	`artifact_type="video_final"`, `artifact_uri`, the `local_path` used by
	production_agent (when it saved into `/content`), and a signed `download_url`
	when the file only exists in ArtifactService. This contract guarantees that
	operators—and the Colab dashboard—can surface the final video without
	re-parsing StepExecutionRecords.
- Control endpoints `POST /pause`, `POST /resume`, and `POST /stop` wrap the
	underlying orchestration state machine. Each accepts `{"run_id": ...}`,
	propagates intent through the asyncio event gates inside `production_agent`,
	and responds once the agent acknowledges the new state. Pauses drain the
	current stage and hold before the next stage begins; stop cancels remaining
	work while persisting the artifact manifest so a `resume_from` request can
	restart safely.
- Status payloads always echo immutable run metadata (plan ID, render profile)
	so operators immediately understand which plan is running and what delivery
	profile is in effect.

### Notebook production dashboard (Colab control cell)

- The notebook embeds an `ipywidgets` (or `google.colab.widgets`) panel that
	drives the endpoints above. Buttons trigger `/invoke` to start production and
	the pause/resume/stop controls; disabled/enabled states follow the latest
	`/status` payload to avoid inconsistent intents, and all callbacks execute in
	Python so no browser scripting is required.
- A status column renders the active stage, elapsed/remaining times derived
	from `StepExecutionRecord` telemetry, and recent log lines so users always
	know “where production_agent is” without reading raw notebooks.
- Asset tabs query `/artifacts` as soon as each stage completes, rendering base
	image thumbnails, inline audio controls for line-level TTS clips, MP4 preview
	players for per-shot videos, and the final assembly via `widgets.Image`,
	`widgets.Audio`, and `IPython.display.Video`. This mirrors the stage payload
	contracts (`plan_intake`, `dialogue_audio`, `base_images`, `video`,
	`assemble`, `finalize`) documented in `NOTEBOOK_AGENT_INTEGRATION.md`.
- The dashboard also ships a dedicated “Final Video” cell: it fetches the
	`video_final` manifest entry from the `finalize` stage, embeds the MP4 inline
	via `IPython.display.Video` or a `widgets.Output` panel, and triggers
	`google.colab.files.download()` (or falls back to `adk artifacts download`)
	so users can export the deliverable from inside Colab immediately.
- Alert banners highlight retries exhausted or pause/stop acknowledgements; a
	future QA replacement will reuse the same surface for automated review
	banners.
- The terminal `finalize` stage concentrates on packaging assembly outputs and
	ensuring metadata (duration, storage hints, download URLs) exists for
	downstream clients.

#### Final video manifest contract

`production_agent` is responsible for publishing a single `/artifacts` row with
`artifact_type="video_final"` once `finalize` completes. The manifest schema is
authoritative and mirrors THE_PLAN so downstream clients never guess field
names:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `artifact_type` | Literal `"video_final"` | yes | Reject mismatched literals before responding. |
| `artifact_uri` | string | yes | ADK ArtifactService URI for long-term storage. |
| `local_path` | string | yes | Absolute Colab path for inline playback. |
| `download_url` | string | conditional | Mandatory when `storage_hint="adk"`; omit when local/filesystem. |
| `storage_hint` | enum `{"adk","local","filesystem"}` | yes | Tells clients whether to read from disk or fetch remotely. |
| `mime_type` | string | yes | Typically `video/mp4`; allows future codecs. |
| `size_bytes` | integer | yes | Enables download progress/quotas. |
| `duration_s` | float | yes | Total runtime; must match assemble metadata. |
| `frame_rate` | float | yes | Derived from shot timing metadata. |
| `resolution_px` | string | yes | `"{width}x{height}"` for layout decisions. |
| `checksum_sha256` | string | yes | 64-char lowercase hex for integrity checks. |
| `run_id` | string | yes | Links manifest to ADK session. |
| `stage_id` | string | yes | Should be `"finalize"`; future stages can add their own rows. |
| `created_at` | ISO 8601 string | optional | Timestamp when manifest emitted (present when backend provides it). |
| `playback_ready` | boolean | yes | Signals whether `local_path` already exists. |
| `notes` | string | optional | Room for render profile / retry counts. |

Validation rules:

1. `download_url` MUST be present any time `storage_hint` is `"adk"`.
2. `checksum_sha256` is validated pre-response; production_agent refuses to
	emit the manifest until the digest matches the on-disk file.
3. `duration_s`, `frame_rate`, and `size_bytes` are positive numbers; negative
	or zero values fail the stage immediately.
4. Missing `video_final` entries are treated as fatal by clients, which surface
	“finalize incomplete” guidance instead of attempting playback.

## Data contracts & storage layout

- **MoviePlan** – schema published as an ADK artifact bundle. Plans are stored
	via ArtifactService (`artifact://sparkle-motion/movie_plan/<run_id>/<ver>`)
	and referenced by ID in WorkflowAgent state.
- **Schema registry** – `configs/schema_artifacts.yaml` plus
	`sparkle_motion.schema_registry` expose the canonical artifact URIs for
	MoviePlan, AssetRefs, QAReport, StageEvent, and Checkpoint along with local
	fallback paths. Prompt templates and WorkflowAgent tooling must read from this
	source so every environment points at the same schema versions. A concise
	reference table now lives in `docs/SCHEMA_ARTIFACTS.md`; update both the YAML
	and that document whenever URIs or versions change.
- **AssetRefs** – stored as structured JSON artifacts plus derived parquet
	rows for analytics. Tools receive signed URLs or stream handles from ADK.
- **QAReport** – schemas remain archived under ADK config bundles for posterity,
	but the runtime no longer emits QAReport artifacts while QA automation is
	paused. Future reinstatements can revive the same schema IDs without changing
	callers.
- **Checkpoints / manifests** – ADK automatically records stage state
	transitions; optional local checkpoint JSONs become a developer convenience,
	not the source of truth.
- **Memory log** – ADK memory timelines contain every stage success/failure and
	human decision. Historical QA actions remain readable for audit but no new
	entries are produced until QA tooling returns. Any agent simply queries
	`memory_service.list` with the run’s session ID.

## Enforced ADK usage

- **Enforced ADK usage** — In-application code (under `src/` and `function_tools/`)
	uses the Google ADK SDK as the primary runtime. Under Option A, every
	FunctionTool that needs an LLM must require the `google.adk` SDK at process
	startup and construct its own `LlmAgent`. Any missing SDK or failed agent
	construction will raise a RuntimeError rather than falling back.
- **Hosted tools** – all GPU/CPU-heavy work happens inside ADK-managed tool
	deployments with per-tool IAM scopes, logging, and rollout controls. In the
	Colab-local profile “hosted” simply means a local Python server/process or
	in-process callable, still registered through the ToolRegistry.
- **Centralized config** – schemas, QA policies, adapter metadata, and
	StageDefaults are distributed as ADK config artifacts so every environment
	reads the same contract.
- **Observability** – rely on ADK telemetry streams (timeline, traces,
	metrics) surfaced inside the CLI/notebook experience. Telemetry export stays
	disabled to keep everything local; optional `run_events.json` files are still
	generated from ADK data for debugging.
- **Determinism** – WorkflowAgent enforces seed propagation and records them
	in session metadata so reruns and audits reference the same inputs. Replay
	relies on `production_agent` writing one `line_artifacts` record per dialogue
	line so downstream lipsync stages (and operators) can chase the exact WAV that
	was synthesized, and QA tooling can plug back in later without changing
	runtime contracts.

## Short-term operational decisions (what changed)

- ADK is required at runtime for any FunctionTool that instantiates agents.
	No silent fallbacks.
- Under Option A, each FunctionTool that requires a model instantiates its own
	agent at startup (per-tool agents). To make this manageable we will add a
	small `src/sparkle_motion/adk_factory.py` that supports `mode="per-tool"`
	(default) and provides guarded SDK probing, credential checks, and helper
	APIs. The factory is a helper for consistent behavior — it does *not* force
	the shared-singleton pattern.
- FunctionTool entrypoints will be normalized (one canonical directory name
	per tool) as a small follow-up to remove duplicate scaffolds (e.g.,
	`ScriptAgent` vs `script_agent`).

	must adopt a guarded `model_context` context manager that standardizes model
	load/unload, emits memory telemetry at key points (load_start/load_complete/
	inference_start/inference_end/cleanup), and normalizes OOMs to a
	`ModelOOMError` domain exception for consistent fallback strategies.
	dependencies or system binaries (for example `torch`, `diffusers`, or
	`ffmpeg`) must be prepared as `proposals/pyproject_adk.diff` and approved
	before editing `pyproject.toml` or CI image definitions.

## ADK helper modules (spec reference)

To keep THE_PLAN, ARCHITECTURE, and IMPLEMENTATION_TASKS in lockstep, this section mirrors the helper spec introduced in `docs/THE_PLAN.md` and establishes the canonical interfaces for implementers.

### `src/sparkle_motion/adk_factory.py`
- **Purpose**: enforce "ADK required" semantics, centralize agent construction, and expose lifecycle hooks for Option A per-tool agents (with a documented shared-mode escape hatch).
- **Public API**:
	- `require_adk(*, allow_fixture: bool = False) -> None` — validates SDK import + credentials; raises `MissingAdkSdkError` unless fixture mode explicitly allowed.
	- `get_agent(tool_name: str, model_spec: ModelSpec, mode: Literal['per-tool','shared']='per-tool') -> LlmAgent` — constructs/returns the agent handle for the caller while recording provenance; raises `AdkAgentCreationError` on failure.
	- `create_agent(config: AgentConfig) -> LlmAgent` — low-level helper for bespoke orchestration layers/tests that need direct control over construction parameters.
	- `close_agent(tool_name: str) -> None` — disposes the per-tool agent (and removes it from the registry) when FunctionTools shut down or Colab resets.
	- `shutdown() -> None` — best-effort cleanup to close every tracked agent; used by notebooks/tests.
- **Failure behavior**: all helpers raise typed exceptions (`MissingAdkSdkError`, `AdkAgentCreationError`, `AdkAgentLifecycleError`) carrying `tool_name`, `model_spec`, and the underlying SDK exception. Fixture bypasses must emit a warning-level `adk_helpers.write_memory_event()` so telemetry shows that a stub path was used.
- **State**: maintains an in-memory registry (`_agents: dict[str, LlmAgentHandle]`) containing metadata (`created_at`, `last_used_at`, `mode`). No persistence; callers reconstruct agents on process restart.

### `src/sparkle_motion/adk_helpers.py`
- **Purpose**: shared façade for ArtifactService publishing, MemoryService writes, human-input requests, and schema registry loading so every FunctionTool and agent emits telemetry the same way.
- **Public API (minimum set)**:
	- `publish_artifact(*, local_path: Path, artifact_type: str, metadata: dict[str, Any], run_id: str | None = None) -> ArtifactRef` — uploads to ADK ArtifactService (or file:// fallback) and returns canonical URIs, raising `ArtifactPublishError` on failure.
	- `publish_local(*, payload: bytes | str, suffix: str, metadata: dict[str, Any] | None = None) -> ArtifactRef` — deterministic helper for fixture/unit tests that stores data under `runs/<run_id>/` and marks `metadata['fixture']=True`.
	- `write_memory_event(run_id: str, event_type: str, payload: Mapping[str, Any], *, ts: datetime | None = None) -> None` — appends structured events to MemoryService (or SQLite fallback). Raises `MemoryWriteError` instead of swallowing failures.
	- `request_human_input(*, run_id: str, reason: str, artifact_uri: str | None, metadata: dict[str, Any]) -> str` — wraps ADK’s review queue, returning a task ID or raising `HumanInputRequestError`.
	- `ensure_schema_artifacts(schema_config_path: Path) -> SchemaRegistry` — loads and validates `configs/schema_artifacts.yaml`, exposing accessors for ScriptAgent/production_agent.
- **Failure behavior**: helper-layer errors (`ArtifactPublishError`, `MemoryWriteError`, `HumanInputRequestError`, `SchemaRegistryError`) must surface machine-readable context so agents can log them. Any fallback to local storage must log via `write_memory_event` that artifacts were saved outside ADK.
- **Testing hooks**: expose `set_backend(overrides: HelperBackend) -> ContextManager` so unit tests inject in-memory fakes; IMPLEMENTATION_TASKS references these hooks for `tests/unit/test_adk_helpers.py`.

For details on the next steps and the todo list, see the updated
`resources/THE_PLAN.md` and `resources/TODO.md` files in this repo.


## Extensibility & human-in-the-loop hooks

- **Script review** – ScriptAgent emits a `request_human_input` event with the
	MoviePlan artifact link. Reviewers respond through ADK’s review console, and
	WorkflowAgent pauses until approval arrives.
- **Image/clip approvals** – Image/video stages can emit human-review requests
	per shot, attaching thumbnails via ArtifactService. Operators approve/reject
	from the ADK UI; their decisions are persisted via MemoryService and drive
	stage retries automatically.
- **QA escalation (paused)** – The retired QA FunctionTool previously wrote
	policy outcomes and, on `regenerate` / `escalate`, triggered automatic retries
	inside WorkflowAgent or raised blocking human tasks. The finalize stage now
	relies solely on human review via `request_human_input`; the escalation hooks
	remain documented so future QA replacements can drop in without new plumbing.
- **New stages** – add a FunctionTool package, publish it to ADK, and update
	the WorkflowAgent graph. No orchestrator code change is required, and gating
	logic (QA or human) can be declared declaratively in the workflow spec.

## Operational summary

- **Environment topology** – the single-user Colab runtime hosts the
	WorkflowAgent definition, tool catalog, and artifact buckets using the
	local-colab profile. Developers run `adk workflows run` to kick off sessions;
	Colab notebooks authenticate with service accounts and call the same APIs.
- **Resume/retry** – restarting a stage is an ADK API call
	(`workflow_runs.resume(run_id, start_stage=...)`). No filesystem surgery is
	required.
- **Artifact access** – operators fetch outputs via `adk artifacts download`
	or through signed URLs exposed in the console. Local `runs/<run_id>` folders
	are optional scratch exports.
- **Human governance** – ADK’s review console and policy engine guard the
	critical checkpoints; approvals are auditable within the platform, and the
	same surfaces will host automated QA once it returns.
- **Documentation parity** – `THE_PLAN.md`, `docs/ORCHESTRATOR.md`, and this file
	all describe the ADK-native deployment so newcomers learn the managed story
	before touching any local harness.

This architecture should continue evolving in lockstep with the WorkflowAgent
and tool catalog definitions so ADK remains the unquestioned source of truth.

## Recommended improvements

- **Centralize ADK helper utilities:** create a small helper module (for
	example `src/sparkle_motion/adk_helpers.py`) that performs the guarded
	`google.adk` import, probes the SDK surface, and exposes typed helpers such
	as `get_adk_module()`, `get_artifact_service()` and `register_tool_via_sdk()`.
	Centralizing probing and candidate-method logic reduces duplicated code and
	makes failures easier to diagnose.
- **Document supported SDK surfaces & versions:** add a concise `docs/ADK_USAGE.md`
	describing which SDK entrypoints the repo expects (e.g., `ToolRegistry`,
	`ArtifactService`, `google.genai.types.Part`) and the tested/target SDK
	versions. Document environment variables the scripts honor (for example
	`ADK_ARTIFACTS_GCS_BUCKET`, `ADK_PROJECT`, `ADK_ARTIFACTS_ROOT`).
- **Add an explicit `--require-sdk` flag for scripts:** where appropriate,
	allow scripts to fail fast when the real SDK (and credentials) are required
	(useful for CI or integration runs). Keep the current SDK-first / CLI-fallback
	behavior as the default for Colab and local developer convenience.
- **Consolidate CLI invocation helpers:** wrap `adk` CLI invocations in a
	shared helper that normalizes stdout/stderr parsing (JSON extraction, URI
	regex), retry/backoff, and clear error messages. This avoids inconsistent
	CLI parsing across multiple scripts.
- **Add an opt-in integration smoke test:** include a small integration test
	that runs only when `SMOKE_ADK=1` (or similar env var) is set. This test
	should validate end-to-end publishing/registration flows against a real
	ADK environment and be opt-in so local/CI runs are unaffected.
- **Keep local shim for unit tests only:** the lightweight `src/google` shim is
	useful for unit tests; ensure docs make clear it is *only* for tests and that
	the real ADK is required for integration/production runs.

## Per-Tool Implementation Details (authoritative mapping)

This section maps each canonical FunctionTool to the recommended model/tool,
the integration approach into the codebase, required runtime environment
variables, and short-term engineering tasks needed to convert scaffolds into
production-capable FunctionTools. These are implementation-level notes that
drive the immediate workstream and the TODO list.

- **script_agent**
	- Model/Tool: ADK LlmAgent wrapping a mid-sized instruction-following LLM
		(canonical choices: Qwen-2.5/7B or Gemini Nano for on-device/colab; a
		hosted alternative is OpenAI/GPT-5-style endpoint or an ADK-hosted model).
	- Integration: instantiate via `adk_factory.get_agent("script_agent", model_spec=...)`.
		Implement a `generate_plan(prompt)` helper that calls `agent.generate()` or
		`agent.run()` (feature-detect) and validates the result against the
		`MoviePlan` schema before publishing as an artifact and appending MemoryService
		metadata.
	- Env vars: `SCRIPT_AGENT_MODEL`, `SCRIPT_AGENT_SEED` (optional), ADK creds.
	- Dependencies (proposal): none in-code beyond `google-adk` for runtime;
		if using a local HF model add `transformers`/`accelerate`/`safetensors`.
	- Risks: hallucination vs schema; mitigate with schema validation and a
		small prompt-synthesis validator + unit tests.

- **images_sdxl**
	- Model/Tool: Stable Diffusion XL (SDXL) model family (via `diffusers` locally
		or an ADK-hosted image-generation model). Canonical default: `stable-diffusion-xl`.
	- Integration: implement `render_images(prompt, cfg)` that wraps
		`StableDiffusionXLPipeline` (base+refiner flow) inside
		`gpu_utils.model_context('sdxl', weights=..., offload, xformers)` with
		`torch_dtype=torch.float16`, `variant='fp16'`, `use_safetensors=True`, and
		optional refiner passes. Honor `prompt_2`/`negative_prompt_2`, micro-
		conditioning fields, deterministic sampling via `torch.Generator(device)
		.manual_seed(seed)`, and publish PNG artifacts with metadata (seed,
		model_id, device, sampler, steps, width/height, guidance scale, timings).
	- Env vars: `IMAGES_SDXL_MODEL`, `IMAGES_SDXL_SEED`, `ARTIFACTS_DIR`, plus
		optional overrides for refiner ids and sampler defaults.
	- Dependencies (proposal): `torch>=2.0` (cu118/cu120), `diffusers`,
		`transformers`, `accelerate`, `safetensors`, optional `xformers` /
		`bitsandbytes`, `huggingface_hub`. Changes must be staged via
		`proposals/pyproject_adk.diff` before editing manifests.
	- Risks: VRAM / CUDA cleanup. Mitigate via `model_context`, explicit
		pipeline deletion, `torch.cuda.empty_cache()`, and tests that simulate OOM
		and verify cleanup. Real runs gated by `SMOKE_ADK=1`; unit tests use the
		deterministic stub guidance from the Implementation Tasks doc.

- **videos_wan**
	- Model/Tool: Wan-AI Wan2.1 FLF2V family (e.g.,
		`Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`). Primary pilot due to GPU load.
	- Integration: load `WanImageToVideoPipeline` with shared `CLIPVisionModel`
		and `AutoencoderKLWan`, enable balanced/sequential device maps per host,
		and wrap inference inside `gpu_utils.model_context('wan2.1/flf2v',
		weights=MODEL_ID, offload=True, max_memory=..., low_cpu_mem_usage=True)`.
		Support scheduler swaps (`DPMSolverMultistepScheduler`), per-chunk seeds,
		keyframe inputs, and callback wiring for progress updates. Save frames via
		`diffusers.utils.export_to_video()` (or ffmpeg fallback) and publish MP4s
		with metadata (chunk stats, model_id, device_map, inference time, peak
		VRAM, cpu_fallback flag).
	- Env vars: `VIDEOS_WAN_MODEL`, `VIDEOS_WAN_SEED`, `ADK_MEMORY_SQLITE` for session persistence, `ARTIFACTS_DIR`.
	- Dependencies (proposal): `torch>=2.0`, `diffusers` with Wan support,
		`transformers`, `accelerate`, `safetensors`, `imageio`, `imageio-ffmpeg`,
		Wan repo utilities. Proposals must document CUDA toolkits and wheel sizes.
	- Risks: VRAM/driver reliability. Follow the Implementation Tasks fallback
		sequence (adaptive chunk shrink, device swap, CPU fallback, fail). Add
		telemetry at load/inference/cleanup and gate integration smokes under
		`SMOKE_ADK=1`/`SMOKE_ADAPTERS=1`.

- **tts_chatterbox**
	- Model/Tool: Resemble AI's open-source Chatterbox TTS (English +
		multilingual variants, `chatterbox-tts` PyPI /
		`https://github.com/resemble-ai/chatterbox`). Outputs are watermarked;
		metadata must capture `watermarked: bool`.
	- Integration: implement `tts_chatterbox` FunctionTool entrypoint that loads
		`ChatterboxTTS` / `ChatterboxMultilingualTTS` via
		`ChatterboxTTS.from_pretrained(device='cuda'|'cpu')` inside
		`gpu_utils.model_context('tts/chatterbox', ...)`, supports optional
		`audio_prompt_path`, `language_id`, `cfg_weight`, `exaggeration`, and
		optional fixture stubs when `SMOKE_TTS` is unset. Publish WAV artifacts via
		`adk_helpers.publish_artifact()` including metadata (duration, sample_rate,
		voice_id, provider voice id, model_id, device, synth_time_s,
		watermarked flag, selected provider score breakdown). `production_agent`
		synthesizes one clip per script line and records the outputs as
		`line_artifacts` so lipsync, QA, and assemble stages can chase every
		utterance end-to-end. Fixture mode (default when `SMOKE_TTS`/`SMOKE_ADAPTERS`
		are unset) must continue generating deterministic per-line WAVs and
		metadata for local replay.
	- Env vars: `TTS_CHATTERBOX_MODEL`, `TTS_CHATTERBOX_DEVICE`,
		`TTS_PRIORITY_PROFILE`, `TTS_PROVIDER_ALLOWLIST`, `SMOKE_TTS`.
	- Dependencies (proposal): `chatterbox-tts`, `torch`, `torchaudio`, and any
		vendor-specific extras. Manifest edits require an approved
		`proposals/pyproject_adk.diff` describing wheel sizes and CUDA variants.
	- Risks: provider licensing / cost. Mitigate via the `tts_stage` selection
		logic, fallback providers, and the error taxonomy defined earlier.

- **lipsync_wav2lip**
	- Model/Tool: Wav2Lip (upstream `https://github.com/Rudrabha/Wav2Lip`), with
		configurable checkpoints (Wav2Lip or Wav2Lip+GAN) and S3FD face detector.
	- Integration: expose `run_wav2lip(face_video, audio, out_path, opts)` that
		either imports the Python inference helper or shells out to a pinned repo
		commit via a safe `run_command` wrapper. Options include `checkpoint_path`,
		`face_det_checkpoint`, `pads`, `resize_factor`, `nosmooth`, `crop`, `fps`,
		`gpu`, `verbose`. Adapter must validate weights, manage temp dirs, ensure
		`ffmpeg` availability, and publish MP4 artifacts with metadata (model
		checkpoint, detector, opts snapshot, runtime, logs URIs).
	- Env vars: `LIPSYNC_WAV2LIP_MODEL`, `LIPSYNC_FACE_DET_PATH`, `FFMPEG_PATH`,
		`SMOKE_LIPSYNC`.
	- Dependencies (proposal): `torch`, `numpy`, `scipy`, `opencv-python`,
		`face-alignment`, upstream Wav2Lip modules. Document binary requirements
		(`ffmpeg`). Tests mock subprocess/model calls to avoid heavy loads.

- **assemble_ffmpeg**
	- Model/Tool: deterministic `ffmpeg` pipeline invoked via a hardened
		`run_command(cmd, cwd, timeout_s, retries)` helper that captures
		stdout/stderr, enforces timeouts, and tears down process groups.
	- Integration: provide `assemble_clips(movie_plan, clips, audio, out_path,
		opts)` that builds concat/filter graphs for overlays, transitions, audio
		mixes, and subtitles. Enforce canonical encode flags
		(`-c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p -movflags
		+faststart`, `-c:a aac -b:a 192k`) unless overridden explicitly, and write
		logs/commands into artifact metadata for reproducibility. Quarantine
		temp outputs when `debug=True`.
	- Env vars: `FFMPEG_PATH`, `SMOKE_ASSEMBLE`, `ARTIFACTS_DIR`.
	- Dependencies: system `ffmpeg` (document install vector) plus optional
		`imageio-ffmpeg` for verification. Unit tests focus on command generation
		and error handling; smoke tests (gated) assemble short synthetic clips.

- **Manual review placeholder**
	- Context: the Stage 3 sunset removed the automated QA FunctionTool.
		Operators now review the assembled outputs (base images through
		`video_final`) using the notebook preview helpers and log their decision via
		`adk_helpers.write_memory_event(event_type="qa_manual_review", ...)` so the
		run history still shows a clear approval trail.
	- Integration: notebook control panels highlight manual-review steps,
		encourage entering reviewer notes, and expose `request_human_input` for any
		runs that require escalation. Downstream automation that previously consumed
		`QAReport` artifacts now relies on these memory events and reviewer note
		attachments until a new FunctionTool is introduced.
	- Future work: when QA automation returns it will be layered on top of the
		`finalize` stage instead of reviving the old FunctionTool. Env vars and
		SMOKE flags will be reintroduced as part of that rollout.

### Implementation notes & process

- These per-tool recommendations intentionally separate runtime **behavior**
	(what the tool must do at a high level) from the **packaging/installation**
	decisions (which may require heavy wheels or service credentials). When a
	local heavy dependency is required (for example Wan or SDXL), prefer a
	containerized or optional-install plan and keep unit tests runnable in
	`ADK_USE_FIXTURE=1` mode.

- Any change that adds runtime dependencies (new Python packages, system
	binaries, or large model weights) must be proposal-only until you approve
	the dependency diff; I will prepare `proposals/pyproject_adk.diff` that
	lists exact package names and versions for review.

- For each tool the immediate engineering tasks are:
	1. Implement load/unload helpers following `gpu_utils.model_context`.
	2. Implement a small `run_*` adapter (e.g., `run_wav2lip`, `render_images`)
		 that performs the work and writes canonical artifact metadata to disk.
	3. Validate outputs against the appropriate schema(s) and publish via
		 `adk_helpers.publish_with_sdk` (or CLI/local file fallback in dev).
	4. Add per-tool smoke tests (unit + optional ADK integration gated by env).

These additions are now the authoritative per-tool mapping and should be used
to drive the concrete TODOs and implementation sprints.
