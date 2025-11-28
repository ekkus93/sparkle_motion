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

2. Normalize `function_tools/` directory names
   - Remove or merge duplicate scaffolds (e.g., `ScriptAgent` vs `script_agent`)
   - Ensure each tool exposes a single canonical `entrypoint.py` and HTTP
     contract.

3. Refactor ScriptAgent to follow Option A (manifest-only) and use `adk_factory`
    - Behavior change (docs-only at this stage): ScriptAgent's domain is
       script/manifest generation. It emits a `MoviePlan` manifest/artifact and
       does not itself generate images/tts/video. ScriptAgent will instantiate
       its own per-process `LlmAgent` (via `adk_factory.get_agent(..., mode="per-tool")`)
       when the runtime refactor is approved and implemented.
    - Keep behavior: agent instantiation remains eager and fatal if it fails
       at process startup (fail-loud). No code changes until docs sign-off.

4. Wire other FunctionTools to instantiate their own agents (Option A rollout)
    - For each tool: the process will construct its own `LlmAgent` at startup
       using `adk_factory.get_agent(tool_name, model_spec, mode="per-tool")`.
    - Add a per-tool smoke test asserting the process fails-loud when the ADK
       SDK is missing. When SDK is present, tools should attach the agent to
       `app.state.agent` (or equivalent) for testability and telemetry.
   - Rollout plan: pilot `videos_wan` first (Wan 2.1 video model — the heaviest VRAM/GPU consumer), validate
       resource sizing and fail-loud behavior, then roll remaining tools in
       controlled waves (see rollout docs in TODO). No code changes until docs
       sign-off.

5. Propose dependency update (for your approval)
   - Prepare `pyproject.toml` diff to add `google-adk` / `google-genai` (exact
     package names and versions will be proposed). Do not apply without
     explicit approval.

6. Tests & gating (opt-in integration)
   - Add integration smoke tests gated by `SMOKE_ADK=1` (opt-in). These are
     for real ADK runs only and must not block normal local tests.
   - Unit tests remain runnable without SDK by mocking or using test shims.
   - Per-tool smoke tests assert fail-loud behavior when the SDK is absent.

---

Operational constraints

- All application code must treat ADK agent construction as required; any
  fallback behavior must be restricted to scripts or tools purposely designed
  to run in environments without the SDK (and those scripts must be flagged
  with `--require-sdk` when they need it).
- `resources/` samples remain for reference only and will not be modified as
  part of this plan unless you explicitly request it.

Legacy runner policy: the legacy runner and its adapters are removed and
quarantined; there will be no compatibility adapter or preserved legacy
runner branch kept for continued use. The project will not retain or support
the legacy runner — it is permanently retired.

---

If you approve these doc changes, sign the checklist below and I will
proceed with the implementation work (code-only after docs sign-off). I will
NOT write runtime code until you explicitly approve the sign-off.

Sign-off checklist (must be approved before code changes)
- [ ] You confirm Option A (per-tool agents) and the canonical tool list above.
- [ ] Resource sizing & rollout wave owners are approved (see `docs/ROLLOUT_PLAN.md`).
- [ ] Dependency proposal (`proposals/pyproject_adk.diff`) approved for review.
- [ ] Security checklist accepted (`docs/SECURITY_CHECKLIST.md`) or owner assigned.

Reply `approve-docs` to approve these document changes and permit the next
code-phase, or `revise-docs` to request edits to this plan.
# The Plan

Build Sparkle Motion as an **ADK-native video workflow** from day one. Every
session, artifact, tool, and coordination step runs on managed ADK services;
Colab remains only a thin operator console.

> Current compute stance: all heavy models (ScriptAgent LLM, SDXL, Wan,
> Chatterbox TTS, Wav2Lip, etc.) run locally inside a single Google Colab A100
> runtime. Since that GPU cannot host all weights simultaneously, stages load
> their model into VRAM, finish their work, and release CUDA memory before the
> next tool acquires the GPU. Each stage stays encapsulated as a FunctionTool
> boundary so we can later point it at managed APIs (e.g., OpenAI GPT-5.1,
> ElevenLabs) without rewriting the workflow.

## Objectives

1. **WorkflowAgent-first orchestration** – encode the entire stage graph in a
   WorkflowAgent definition (YAML/JSON) deployed via `adk workflows deploy`.
2. **Hosted FunctionTool catalog** – package each heavy model/tool as an ADK
   FunctionTool/ToolRuntime deployment with IAM, telemetry, and versioning.
3. **Schema-first contracts** – publish MoviePlan, AssetRefs, QAReport, and
   Checkpoint schemas as ADK artifacts that agents/tools import at runtime.
4. **Managed services** – rely on ADK SessionService, ArtifactService,
   MemoryService, and observability pipelines configured exclusively for the
   **local-colab** profile (SQLite files, mounted Google Drive path,
   filesystem secrets, telemetry disabled). Local mirrors are optional caches,
   never the source of truth.
5. **Human + QA governance baked in** – stages trigger ADK’s review APIs and
   policy engine rather than bespoke JSON polling.
6. **Single-run friendly UX** – CLI + Colab flows authenticate into ADK and
   call the same workflow APIs, keeping the local machine stateless.

## Workstreams & Deliverables

| # | Workstream | Key Deliverables |
|---|------------|------------------|
| 1 | ADK project bootstrap | Local-colab ADK profile (SQLite session/memory DBs, Drive artifact root, secrets template, WorkflowAgent skeleton). |
| 2 | Schema + config publishing | MoviePlan/AssetRefs/QAReport schemas, QA policy bundle, Stage defaults – all as ADK artifacts. |
| 3 | Tool packaging | SDXL, Wan, Chatterbox TTS, Wav2Lip, Assemble (ffmpeg), QA (Qwen2-VL) packaged as FunctionTools with metadata + retries. |
| 4 | WorkflowAgent definition | Graph describing `script -> images -> videos -> tts -> lipsync -> assemble -> qa`, retry policies, human gates, telemetry wiring. |
| 5 | Human-in-loop + QA gating | `request_human_input` hooks at script/images, QA auto-regenerate + escalation controls, MemoryService logging. |
| 6 | Operator experience | `adk workflows run`, Colab notebook calling the workflow, run dashboards referencing ADK telemetry exports. |

## Implementation Phases

### Phase 1 – Platform foundation
- Provision the **local-colab** profile by creating SQLite databases (session +
   memory) under `/content/`, mounting Google Drive for artifacts, and writing a
   local `.env` for secrets via the bootstrap manifest/script.
- Define service accounts + IAM for WorkflowAgent, tools, and operators, then
   store their credentials inside the local secrets template.
- Publish schema artifacts (MoviePlan, AssetRefs, QAReport) + QA policy bundle.
- Record their artifact URIs in `configs/schema_artifacts.yaml` so every
   consumer points to the same `artifact://sparkle-motion/schemas/.../v1`
   locations, with `sparkle_motion.schema_registry` providing runtime access.
   If ADK credentials are unavailable (for example on an isolated server), an operator may perform a local-only publish: copy schema files into `artifacts/schemas/` and point `configs/schema_artifacts.yaml` to local `file://` URIs so local tools and the runner can read the canonical schemas.

### Phase 2 – Tool packaging
- ScriptAgent LlmAgent (Qwen2.5-7B or Gemini Nano) served via ADK model
  adapter; outputs validated against MoviePlan schema.
- FunctionTool deployments run as FastAPI/Flask servers (or in-process
   callables) bound to `http://127.0.0.1:<port>` with no auth so every stage
   lives inside the Colab runtime. Same catalog: SDXL, Wan FLF2V, Chatterbox
   TTS, Wav2Lip, Assemble (containerized ffmpeg), QA (Qwen2-VL).
- Each tool exports capability metadata (inputs/outputs, GPU needs, cost).

### Phase 3 – WorkflowAgent definition
- Encode stage graph in WorkflowAgent YAML: tool bindings, concurrency,
  retry/backoff (3 attempts, exponential + jitter), auto-resume semantics.
- Use ADK EventActions for logging + artifact registration; rely on ADK
  timeline/manifest rather than local checkpoint JSON.
- Implement `workflow_runs.resume(..., start_stage=...)` path for manual or
  QA-driven regenerations.

### Phase 4 – Human + QA controls
- Script and image stages emit `request_human_input` events with ArtifactService
  links; WorkflowAgent pauses until reviewers approve in the ADK console.
- QA FunctionTool writes QAReport artifacts + policy decisions; WorkflowAgent
  automatically requeues regenerate stages or escalates to humans per policy.
- MemoryService logs every decision, including reviewer notes, latency, seeds.

### Phase 5 – Operator tooling
- CLI wrapper + Colab notebook authenticate via service account, call `adk
  workflows run`, watch progress via ADK session APIs, download artifacts via
  `adk artifacts download`.
- Observability relies on ADK session timelines and notebook logs; telemetry
   export stays disabled in the local-colab profile.

## Stage-by-stage plan (hosted tools)

1. **ScriptAgent (ADK LlmAgent)**
   - Input: idea + target duration; uses MoviePlan schema to constrain output.
   - Output: MoviePlan artifact version + memory entry.
2. **Images (SDXL FunctionTool)**
   - Inputs: MoviePlan prompts; outputs start/end PNG artifacts per shot.
3. **Videos (Wan FunctionTool)**
   - Inputs: start/end frames, motion prompt; outputs raw MP4 artifacts.
4. **TTS (Chatterbox FunctionTool)**
   - Inputs: dialogue lines; outputs WAV artifacts + metadata (speaker, seed).
5. **Lipsync (Wav2Lip FunctionTool)**
   - Inputs: raw clip + dialogue audio; outputs lipsynced MP4 artifacts.
6. **Assemble (ffmpeg FunctionTool)**
   - Inputs: final per-shot clips + audio; outputs final movie artifact.
7. **QA (Qwen2-VL FunctionTool)**
   - Inputs: sampled frames + shot descriptions; outputs QAReport artifact and
     policy decision, potentially triggering auto-regenerate/human review.

## Data & operations

- **Artifacts** – referenced via ADK URIs (e.g.,
  `artifact://sparkle-motion/asset_refs/<run>/<version>`). Local `runs/`
  exports are optional developer pulls.
- **Seeds/determinism** – WorkflowAgent stores seeds in session metadata and
  enforces propagation to downstream tools for reproducible reruns.
- **Retries/resume** – handled by WorkflowAgent policies; manual reruns use
  `workflow_runs.resume` APIs, no manual file edits.
- **Human review** – ADK review console is the control tower; JSON polling is
  gone. Reviewer decisions stream into MemoryService and ADK timeline.
- **Observability** – rely on ADK metrics/timeline surfaced directly inside
   the CLI/notebook experience. Telemetry export remains disabled and we lean on
   notebook logs. Optional `run_events.json` can be generated *from* ADK data
   for convenience, not as source of truth.

## Immediate next steps

1. Finalize WorkflowAgent schema + repo location; scaffold deployment command.
2. Containerize each FunctionTool (Dockerfiles + IaC) and register in ADK
   catalog with metadata + retry hints.
   Note: containerization is optional and targeted at hosted deployments only.
   The `local-colab` profile cannot run Docker — use in-process runners or
   local FastAPI servers for development and testing in Colab.
3. Publish schema + QA policy artifacts; update ScriptAgent prompt templates to
   fetch schema from ADK rather than embedding copies. Consumers call
   `schema_registry.get_schema_uri("movie_plan")` (or
   `get_schema_path(...)` for local fallbacks) instead of hard-coding paths.
   The helper `sparkle_motion.prompt_templates.build_script_agent_prompt_template()`
   and the CLI `scripts/render_script_agent_prompt.py` already encode this wiring
   so the same JSON payload can be pushed to ADK without manual edits.
4. Wire Colab notebook + CLI to call WorkflowAgent and download artifacts via
   ADK APIs, removing residual dependencies on local runner semantics.
5. Update `resources/ORCHESTRATOR.md` / `docs/ORCHESTRATOR.md` to reference the
   WorkflowAgent contract wherever they previously described the Python runner.

## Per-Tool Implementation Tasks (detailed)

Below are concrete per-tool tasks and recommended model/tool choices to make
the FunctionTools fully implemented (not scaffolds). These items drive the
implementation sprint once you approve dependency proposals and the rollout
plan.

### script_agent
- Model: ADK `LlmAgent` wrapping Qwen-2.5/7B or Gemini Nano (on-device), or
   an ADK-hosted LLM for hosted deployments.
- Tasks: Implement `generate_plan(prompt)` that calls `agent.generate()` (or
   a feature-detected equivalent), validate output against `MoviePlan` schema,
   publish artifact via `adk_helpers`, and write MemoryService metadata. Add
   smoke tests for schema conformance and SDK fail-loud behavior.

### images_sdxl
- Model: Stable Diffusion XL (SDXL) via `diffusers` locally or ADK-hosted
   image-generation model.
- Tasks: Implement `render_images(prompt, options)` using
   `gpu_utils.model_context` and provider API (`diffusers` pipeline or ADK
   agent image call). Persist images, publish artifacts, and add
   deterministic-sampling and memory-friendly modes.

### videos_wan (pilot)
- Model: Wan-AI Wan2.1-I2V-14B-720P (Hugging Face / Wan-Video).
- Tasks: Implement `load_wan_model()` and `run_wan_inference(...)` inside
   `gpu_utils.model_context`; produce MP4 artifacts, validate codecs/size, and
   add GPU smoke tests for model load/unload. Pilot this tool first due to
   VRAM and driver risk.

### tts_chatterbox
- Model: ADK-enabled TTS (if ADK agent exposes TTS) or hosted providers
   (ElevenLabs) with open-source fallbacks (Coqui/Bark) for local dev.
- Tasks: Implement `synthesize_speech(text, voice_config)`, persist WAV,
   publish artifact, and emit metadata (duration, sample rate). Include
   license/cost review for hosted providers.

### lipsync_wav2lip
- Model: Wav2Lip pipeline (open-source) with GPU acceleration.
- Tasks: Provide `run_wav2lip(video_path, audio_path, out_path)` adapter
   (Python API preferred; subprocess wrapper permitted), ensure ffmpeg/opencv
   availability, and integrate with `assemble_ffmpeg` for final packaging.

### assemble_ffmpeg
- Tool: system `ffmpeg` invoked via a safe subprocess wrapper.
- Tasks: Implement deterministic assembly (concat, overlay, audio mix)
   driven by `MoviePlan`; verify output integrity and add acceptance tests
   for codecs/containers.

### qa_qwen2vl
- Model: Qwen-2-VL or ADK LlmAgent with multimodal capability.
- Tasks: Implement `inspect_frames(frames, prompts)` to produce structured
   `QAReport` artifacts and policy decisions; integrate `request_human_input`
   for escalations.

### Dependencies & approval
- Any proposed runtime dependency (large Python wheels, system binaries,
   or model weights) will be prepared as `proposals/pyproject_adk.diff` and
   presented for your review. Per repo policy, do not add those dependencies
   to manifests until you approve the proposal.

### Tests & rollout
- Add per-tool unit + smoke tests. Integration tests requiring real
   credentials or heavy weights should be gated by `SMOKE_ADK=1` and
   `CI_SMOKE_OPTS`. Pilot order: `videos_wan` → `images_sdxl` → `tts_chatterbox`
   → `lipsync_wav2lip` → `assemble_ffmpeg` → `qa_qwen2vl` → `script_agent`.

## Recommended improvements

- **Create a centralized ADK helper module:** add `src/sparkle_motion/adk_helpers.py`
   to encapsulate guarded `google.adk` imports, SDK probing, and typed helper
   functions (`get_adk_module()`, `get_artifact_service()`, `publish_schema()`,
   `register_tool_via_sdk()`). This reduces duplicated probe logic across scripts
   and makes SDK-related errors easier to diagnose.
- **Document SDK expectations:** add `docs/ADK_USAGE.md` listing the SDK
   entrypoints and versions the repo targets, the expected shapes (ToolRegistry,
   ArtifactService, `google.genai.types.Part`), and the environment variables
   scripts honor (e.g., `ADK_ARTIFACTS_GCS_BUCKET`, `ADK_PROJECT`).
- **Add `--require-sdk` opt-in for scripts:** where a script must have the
   SDK and credentials (integration/CI), support a `--require-sdk` flag that
   fails fast when the SDK is absent rather than silently falling back to CLI.
- **Consolidate `adk` CLI helpers:** provide a single helper for calling the
   `adk` CLI that normalizes JSON/stdout parsing, retries, and error messages.
- **Opt-in integration smoke test:** add a small integration test gated by
   `SMOKE_ADK=1` to validate publishing/registration flows against a real ADK
   environment. Make it opt-in so it doesn't run in normal local/CI test runs.

With these steps, Sparkle Motion operates exactly as ADK intended: ADK services
own session state, artifacts, memory, tool execution, and human gating; local
code simply becomes a client riding on top of the platform.