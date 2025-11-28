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
- **FunctionTools (application code under `function_tools/`):** 7 directories
	- Canonical tools: `script_agent`, `images_sdxl`, `videos_wan`, `tts_chatterbox`,
		`lipsync_wav2lip`, `assemble_ffmpeg`, `qa_qwen2vl`
	- Note: there is a case-duplicate pair `ScriptAgent` / `script_agent` in
		the tree; the duplicate scaffold will be removed during normalization.

Implementation tasks alignment: `docs/IMPLEMENTATION_TASKS.md` is the
authoritative per-tool TODO reference for implementers. Important clarifications
introduced there (now reflected in this architecture) include:

- Agents (policy/orchestration) vs FunctionTools (compute adapters): Agents
	include `script_agent` (plan generation), `production_agent` (plan
	execution/orchestration), `images_agent`, `tts_agent`, and `videos_agent`.
	FunctionTools remain the heavy compute adapters such as `images_sdxl`,
	`videos_wan`, `tts_chatterbox`, `lipsync_wav2lip`, `qa_qwen2vl`, and
	`assemble_ffmpeg`.
- Per-tool gating: heavy adapters and integration tests are gated by
	`SMOKE_*` flags (for example `SMOKE_ADK`, `SMOKE_TTS`, `SMOKE_LIPSYNC`,
	`SMOKE_QA`, `SMOKE_ADAPTERS`) to avoid accidental heavy installs or model
	loads in CI/local runs.
- Model lifecycle: adapters must use the shared `gpu_utils.model_context`
	pattern to load/unload models, emit telemetry, and normalize OOMs to the
	`ModelOOMError` domain exception so higher-level agents can implement
	deterministic fallback/shrink strategies.

Note: the `resources/` directory contains many ADK sample projects and
examples. Those are vendor/sample code and are intentionally excluded from
these counts per your instruction — they are references, not application
runtime deployments.
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
- **Schema-first contracts** – MoviePlan, AssetRefs, QAReport, and Checkpoint
	schemas are published as ADK artifacts so both agents and tools fetch the
	same canonical definitions at runtime.
- **Human + QA hooks baked in** – ADK’s `request_human_input` / review queue
	and policy evaluation APIs gate stages without custom JSON polling.
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
- Persistence: ADK ArtifactService + MemoryService are the sources of truth;
	local files and caches are developer conveniences only.

### Service & tool wiring

- **SessionService** – every run is created via ADK’s managed session service
	configured for the Colab-local profile. The adapter points at a SQLite
	database file (e.g., `/content/sparkle_session.db`). Session metadata, user
	info, and run state live there permanently; local folders mount as ephemeral
	caches only when a developer needs to inspect artifacts on Colab.
- **ArtifactService** – MoviePlan, AssetRefs, QA reports, checkpoints, and
	final renders are stored in ADK’s artifact buckets. We mount Google Drive
	(`/content/drive/MyDrive/sparkle_artifacts`) and treat it as the artifact
	root. Tools receive Drive-backed handles; no bespoke file paths are passed
	outside ADK.

- Local-only artifacts: for isolated or single-user environments we accept an
	alternative artifact root under `artifacts/` (for example
	`artifacts/schemas/`). In that mode tools and operators may reference
	artifacts using `file://` URIs or repo-relative paths. This is a
	developer-only fallback for environments without ADK credentials and is not
	a substitute for publishing artifacts into the ADK control plane for shared
	deployments.
- **MemoryService** – long-lived run logs, QA outcomes, and human decisions
	are appended through ADK’s memory APIs backed by a SQLite file such as
	`/content/sparkle_memory.db` so any agent (ScriptAgent, QA Agent, future
	WorkflowAgent) can query histories without parsing filesystem logs.

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
	stage graph (`script -> images -> videos -> ... -> qa`). It binds each step
	to a FunctionTool ID, specifies retry policies, and emits ADK manifest events
	automatically. Resume is a first-class capability: rerun requests simply
	invoke WorkflowAgent with a `start_stage` override.
4. **FunctionTool executions** – Each tool runs inside the Colab runtime as a
	lightweight HTTP server (or registered in-process callable) bound to
	`127.0.0.1:<port>`. Inputs reference artifact IDs, not file paths; outputs are
	new artifact versions (AssetRefs, clips, QA reports).
	Tools publish structured telemetry (latency, cost, GPU usage) via ADK’s
	telemetry hooks.
5. **QA & human gating** – The QA tool produces `QAReport` artifacts plus
	policy decisions. When escalation or manual edits are required, it triggers
	`event_actions.request_human_input`, which pushes the run onto ADK’s review
	queue. Approvals/rejects are captured in MemoryService and unblock the
	WorkflowAgent when resolved.
6. **Observability & run book** – ADK’s timeline + metrics are inspected via
	the CLI/notebook interfaces and local logs. Telemetry export is disabled, so
	operators rely on notebook logs and session metadata only. Local summaries
	(if desired) are generator outputs derived from the ADK data, not the other
	way around.

## Data contracts & storage layout

- **MoviePlan** – schema published as an ADK artifact bundle. Plans are stored
	via ArtifactService (`artifact://sparkle-motion/movie_plan/<run_id>/<ver>`)
	and referenced by ID in WorkflowAgent state.
- **Schema registry** – `configs/schema_artifacts.yaml` plus
	`sparkle_motion.schema_registry` expose the canonical artifact URIs for
	MoviePlan, AssetRefs, QAReport, StageEvent, and Checkpoint along with local
	fallback paths. Prompt templates and WorkflowAgent tooling must read from this
	source so every environment points at the same schema versions.
- **AssetRefs** – stored as structured JSON artifacts plus derived parquet
	rows for analytics. Tools receive signed URLs or stream handles from ADK.
- **QAReport** – emitted as an artifact + memory event; policy schemas live in
	ADK config bundles so updates roll out atomically.
- **Checkpoints / manifests** – ADK automatically records stage state
	transitions; optional local checkpoint JSONs become a developer convenience,
	not the source of truth.
- **Memory log** – ADK memory timelines contain every stage success/failure,
	QA action, and human decision. Any agent simply queries `memory_service.list`
	with the run’s session ID.

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
	in session metadata so reruns and audits reference the same inputs.

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

- `gpu_utils.model_context` requirement: per the Implementation Tasks, adapters
	must adopt a guarded `model_context` context manager that standardizes model
	load/unload, emits memory telemetry at key points (load_start/load_complete/
	inference_start/inference_end/cleanup), and normalizes OOMs to a
	`ModelOOMError` domain exception for consistent fallback strategies.
- Proposals required for manifest changes: any proposal that adds runtime
	dependencies or system binaries (for example `torch`, `diffusers`, or
	`ffmpeg`) must be prepared as `proposals/pyproject_adk.diff` and approved
	before editing `pyproject.toml` or CI image definitions.

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
- **QA escalation** – QA FunctionTool writes policy outcomes and, on
	`regenerate/escalate`, either triggers auto-regeneration inside WorkflowAgent
	or raises a blocking human task.
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
- **Human + QA governance** – ADK’s review console and policy engine guard the
	critical checkpoints; approvals are auditable within the platform.
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
	- Integration: implement `render_images(prompt, cfg)` helper that loads SDXL
		in `gpu_utils.model_context(...)` and invokes the provider API (`pipeline(...)`
		for `diffusers`, or `agent.generate_image(...)` when using ADK agent).
	- Env vars: `IMAGES_SDXL_MODEL`, `IMAGES_SDXL_SEED`, `ARTIFACTS_DIR`.
	- Dependencies (proposal): `diffusers`, `transformers`, `accelerate`, `safetensors` if local; otherwise none beyond `google-adk`.
	- Risks: VRAM; use small-batch inference, offload schedulers, and test with
		`MODEL_LOAD_DELAY` and `DETERMINISTIC` flags. Provide a fallback `--negative` memory-friendly sampler option.

- **videos_wan**
	- Model/Tool: Wan-AI Wan2.1-I2V-14B-720P (Hugging Face / Wan-Video). This is
		the primary heavy-footprint model and the rollout pilot.
	- Integration: provide `load_wan_model()` and `run_wan_inference(start_frames, end_frames, prompt)` functions.
		Use `gpu_utils.model_context(_loader, name="videos_wan")` to page weights and
		call the model's generation API. Produce MP4 artifacts and upload via
		`adk_helpers.publish_with_sdk` or CLI.
	- Env vars: `VIDEOS_WAN_MODEL`, `VIDEOS_WAN_SEED`, `ADK_MEMORY_SQLITE` for session persistence, `ARTIFACTS_DIR`.
	- Dependencies (proposal): HF runtime deps as required by the Wan repo (likely `transformers`, `accelerate`, custom Wan-Video code). These must be added only after approval due to large wheels and licensing.
	- Risks: VRAM/driver reliability; add intensive integration tests, GPU smoke tests, and a strict model load/unload discipline. Pilot this tool first.

- **tts_chatterbox**
	- Model/Tool: two viable patterns — (A) ADK LlmAgent that exposes a TTS
		`synthesize()` method (if available), or (B) integrate a dedicated TTS
		engine such as ElevenLabs (cloud) or an open-source fallback (Coqui TTS,
		Bark) for local runs.
	- Integration: implement `synthesize_speech(text, voice_config)` that uses
		the agent API when `ADK_USE_FIXTURE=0` + SDK present; otherwise call
		external TTS SDK/HTTP endpoint. Persist WAV and publish via ADK artifacts.
	- Env vars: `TTS_CHATTERBOX_MODEL`, `TTS_CHATTERBOX_VOICE`, `TTS_API_KEY` (when using hosted).
	- Dependencies (proposal): optional `coqui-ai` or provider SDKs; prefer
		calling provider APIs via thin adapters to avoid heavy local installs.
	- Risks: latency and licensing for hosted voices; include a license/cost
		review before enabling any paid provider in CI runs.

- **lipsync_wav2lip**
	- Model/Tool: Wav2Lip (open-source) pipeline that performs audio-driven
		lip-synchronization using a face/video input and a target audio track.
	- Integration: ship a light adapter `run_wav2lip(video_path, audio_path, out_path)`
		that calls a Python API (preferred) or a subprocess helper pointing at the
		Wav2Lip repo. Ensure the tool can run in a GPU-backed `gpu_utils.model_context`.
	- Env vars: `LIPSYNC_WAV2LIP_MODEL` (path or name), `ARTIFACTS_DIR`.
	- Dependencies (proposal): `torch`, `opencv-python`, and Wav2Lip sources.
	- Risks: extra binary dependencies (ffmpeg/opencv). Use the `assemble_ffmpeg`
		step to normalize final packaging.

- **assemble_ffmpeg**
	- Model/Tool: `ffmpeg` binary invoked via a safe subprocess wrapper.
	- Integration: implement deterministic assembly pipelines defined by the
		`MoviePlan` (concat, overlay, audio mix). Use JSON-driven command templates
		and verify output integrity before publishing artifacts.
	- Env vars: `FFMPEG_PATH` (optional), `ARTIFACTS_DIR`.
	- Dependencies: system `ffmpeg` (document installation), `python-ffmpeg` or
		simple subprocess usage. No ADK SDK-specific dependency required beyond artifact publishing.
	- Risks: cross-platform flags and codec licensing; lock a canonical ffmpeg
		invocation and add acceptance tests that validate containerized runs.

- **qa_qwen2vl**
	- Model/Tool: Qwen-2-VL (vision-language model) or an ADK LlmAgent that
		exposes multimodal capabilities. Use it to inspect frames and emit a
		`QAReport` artifact and policy decision.
	- Integration: implement `inspect_frames(frames, prompts)` that calls the
		agent (feature-detect `agent.analyze` / `agent.generate`) or invokes HF
		adapter for Qwen-2-VL. Publish results and optionally call
		`event_actions.request_human_input` when escalation is needed.
	- Env vars: `QA_QWEN2VL_MODEL`, `QA_QWEN2VL_SEED`.
	- Dependencies (proposal): `transformers`/vision tokenizers or ADK SDK only.
	- Risks: multimodal model licensing and performance; gate by opt-in smoke test.

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
