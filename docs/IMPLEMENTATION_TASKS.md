# Implementation Tasks — Per-Tool (issue-style)

This document contains issue-style, actionable TODOs for converting FunctionTool
scaffolds into production-capable, ADK-integrated implementations. These are
meant to be reference tasks for engineering sprints; each item should be
implemented behind feature branches and reviewed before changing manifests.

Guidelines:
- All runtime dependency changes must be proposed via `proposals/pyproject_adk.diff` and approved before editing `pyproject.toml`.
- Integration tests requiring real ADK credentials or heavy weights are gated by `SMOKE_ADK=1`.
- Entrypoints that instantiate agents must `require_adk()` and fail loudly if SDK or credentials are missing.

## Summary — Agents & FunctionTools

- **Agents (decision/orchestration layers)** — currently declared agents:
  - `script_agent` : LLM-based plan generator (`generate_plan(prompt) -> MoviePlan`).
  - `production_agent` : Production/director orchestration agent (executes MoviePlans, calls media agents and assemblers).
  - `tts_agent` : TTS decision layer (provider selection, policy, retries) — invokes `tts_chatterbox` FunctionTool.
  - `images_agent` : Image orchestration layer (policy, provider selection, batching) — calls `images_sdxl` FunctionTool.
  - `videos_agent` : Video orchestration layer (chunking, provider selection, orchestration) — calls `videos_wan` FunctionTool.

- **FunctionTools / Adapters (compute-bound)** — current adapters/function tools:
  - `images_sdxl` (DiffusersAdapter) — SDXL image renderer (`render_images(prompt, opts)`).
  - `tts_chatterbox` — Chatterbox TTS FunctionTool (synthesis, wav export).
  - `videos_wan` (WanAdapter) — heavy video inference pipeline.
  - `lipsync_wav2lip` — wav2lip lipsync adapter.
  - `qa_qwen2vl` — frame QA / inspection adapter.
  - `assemble_ffmpeg` — deterministic ffmpeg assembly helper.

Note: the document intentionally separates Agents (policy & orchestration) from FunctionTools (heavy compute). Agents may call one or more FunctionTools; FunctionTools should be plain and testable (stubs) in dev, and can later be wrapped into ADK-aware adapters when ready to publish.

  **Agent → FunctionTool relationships (1-to-many)**

  Below are the canonical 1-to-many relationships showing which FunctionTools each Agent may invoke. This is a guide for implementers — agents enforce policy, orchestration and retries, while FunctionTools perform heavy compute. An Agent may call other Agents as part of orchestration (e.g., `script_agent` invoking `images_agent`) and may therefore be indirectly connected to additional FunctionTools.

  - **`script_agent`**: produces a `MoviePlan` only — public API `generate_plan(prompt) -> MoviePlan`. It is intentionally pure (planning, prompts, and schema validation) and should not perform heavy orchestration or FunctionTool calls.
  - **`production_agent`**: `images_agent`, `tts_agent`, `videos_agent`, `assemble_ffmpeg`, `qa_qwen2vl` — responsible for executing a `MoviePlan`, orchestrating per-media agents and FunctionTools, enforcing policy and gating (e.g., `SMOKE_ADK`), and publishing final artifacts. This keeps planning separate from heavy compute and resource management.
  - **`tts_agent`**: `tts_chatterbox` (primary), local/fixture TTS stubs (dev) — chooses provider, enforces policy, and calls TTS FunctionTools for synthesis and WAV artifact publishing.
  - **`images_agent`**: `images_sdxl`, `qa_qwen2vl`, `assemble_ffmpeg` (helper) — performs content checks, batching and calls the `DiffusersAdapter`/`images_sdxl` renderer; may call QA or assembly helpers as needed.
  - **`videos_agent`**: `videos_wan`, `lipsync_wav2lip`, `images_sdxl` (keyframes), `qa_qwen2vl`, `assemble_ffmpeg` — orchestrates chunked video rendering, optional lipsync, frame-level image generation, QA, and final assembly.

  Notes:
  - Relationship type: 1 Agent → many FunctionTools (and sometimes other Agents).
  - Agents should treat FunctionTools as thin, testable adapters and not embed heavy model logic themselves.
  - When adding or changing mappings, update this section so implementers know which adapters to stub/implement and which manifests/proposals may be impacted.

------------------------------------------------------------

## Agent Tasks

These are decision/orchestration layers that select providers, enforce policy, and orchestrate FunctionTool invocations. Implement agents as small, testable services that call into adapters (FunctionTools) for heavy compute.

### script_agent
- Task: Implement `generate_plan(prompt: str) -> MoviePlan`
This agent is plan-only: it generates and validates a `MoviePlan` (the script) and should not execute the plan or call FunctionTools directly. Execution, rendering, and synthesis are delegated to the `production_agent`, which handles orchestration, policy enforcement, gating (e.g., `SMOKE_ADK`), and calls to per-media agents and FunctionTools.
  - Use `adk_factory.get_agent('script_agent', model_spec)` to construct the LlmAgent.
  - Publish plan artifact via `adk_helpers.publish_artifact()` and write a memory event.
  - Tests: unit test for schema conformance; gated integration smoke that exercises a small LLM (fixture-mode or real SDK).
  - Estimate: 2–3 days (including prompts+validation)

  **MoviePlan schema (recommended)**

  - `MoviePlan` (object)
    - `id: str` — unique plan identifier (UUID recommended)
    - `title: str` — short human-readable title
    - `description: Optional[str]` — optional longer description
    - `created_by: Optional[str]` — user id or agent id that created the plan
    - `created_at: str` — ISO 8601 timestamp
    - `steps: List[Step]` — ordered list of steps to execute
    - `assemble_opts: Optional[dict]` — final assembly/options (codec, crf, fps)

  - `Step` (union by `type`)
    - Common fields: `id: str`, `type: str`, `title: str`, `opts: dict`
    - `type == "image"`:
      - `prompt: str` — text prompt to render
      - `count: int` — number of images to produce (default 1)
      - `opts: ImagesOpts` — passed to `images_agent`/`images_sdxl`
    - `type == "tts"`:
      - `text: str` — text to synthesize
      - `voice: dict` — voice config (id, style)
      - `opts: TTSOpts` — sample rate, format
    - `type == "video"`:
      - `prompt: str` — prompt for video render
      - `start_frames: list[str|Path]` — reference frames or keyframes
      - `end_frames: list[str|Path]` — optional end frames
      - `opts: VideoOpts` — number of frames, height/width, steps
    - `type == "lipsync"`:
      - `face_video: str|Path`, `audio: str|Path`, `out_path: str|Path`
      - `opts: Wav2LipOpts`
    - `type == "custom"`:
      - `handler: str` — extension point; `opts` contains handler args

  **Example MoviePlan (JSON)**

  ```json
  {
    "id": "0f8fad5b-d9cb-469f-a165-70867728950e",
    "title": "Demo: Intro Clip",
    "created_at": "2025-11-27T12:00:00Z",
    "steps": [
      {"id":"s1","type":"image","title":"title_card","prompt":"A cinematic title card, sunrise","opts":{"width":1280,"height":720,"seed":42}},
      {"id":"s2","type":"tts","title":"narration","text":"Welcome to the demo.","voice":{"id":"emma"}},
      {"id":"s3","type":"video","title":"main_shot","prompt":"A calm ocean at dawn","start_frames":[],"end_frames":[],"opts":{"num_frames":32}}
    ],
    "assemble_opts": {"codec":"libx264","crf":18,"fps":30}
  }
  ```

  **Example MoviePlan (YAML)**

  ```yaml
  id: 0f8fad5b-d9cb-469f-a165-70867728950e
  title: Demo: Intro Clip
  created_at: 2025-11-27T12:00:00Z
  steps:
    - id: s1
      type: image
      title: title_card
      prompt: "A cinematic title card, sunrise"
      opts:
        width: 1280
        height: 720
        seed: 42
    - id: s2
      type: tts
      title: narration
      text: "Welcome to the demo."
      voice:
        id: emma
    - id: s3
      type: video
      title: main_shot
      prompt: "A calm ocean at dawn"
      opts:
        num_frames: 32
  assemble_opts:
    codec: libx264
    crf: 18
    fps: 30
  ```

  **LLM prompt template & sampling guidance**

  - Template: provide a short, structured system prompt and a user instruction that requests a JSON MoviePlan only. Always include an explicit output schema and a final validation checklist. Example system+user bundle:

  ```text
  System: You are a plan author. Produce valid JSON following the MoviePlan schema exactly. Do not include commentary.
  User: Create a MoviePlan for: <user prompt>
  Constraints: max 10 steps; steps must be one of image|tts|video|lipsync|custom; include assemble_opts.
  Output: JSON only, matching the schema.
  ```

  - Sampling settings (recommended): `temperature=0.0..0.3` for deterministic structure; `top_p=0.8` where creative variation desired. Use `max_tokens` adequate for plan size (e.g., 1024).

  - Few-shot: include 1–2 example plans as few-shot context when the prompt is ambiguous about style or length.

  **Safety & hallucination mitigations**

  - Enforce strict output-only mode in the LLM instruction (JSON-only). Post-parse: run schema validation and a content-policy pass (check for disallowed content in prompts/text fields). If disallowed content is detected, return a `PolicyViolation` result instead of a plan.
  - Limit fields that can contain free text (e.g., `prompt`) to a configurable max length; normalize/escape user-provided text.
  - Where factual claims are required (e.g., `created_by` metadata), mark as optional and do not rely on LLM for authoritative IDs.

  **Validation rules & error types**

  - Validation steps (brief):
    1. JSON parse success
    2. Conformance to `MoviePlan` schema (field presence/types)
    3. Step-level validation: allowed `type`, valid `opts` types, numeric ranges
    4. Policy checks: no disallowed content in `prompt`/`text` fields
    5. Resource estimates: reject plans exceeding configured resource caps (e.g., total frames > 10k)

  - Error types (exceptions / structured error objects):
    - `PlanParseError` — invalid JSON / unparseable output
    - `PlanSchemaError` — missing or invalid schema fields
    - `PlanPolicyViolation` — disallowed content detected
    - `PlanResourceError` — estimated resource usage exceeds allowed budgets

  **Unit tests (suggested)**

  - Positive test: given a simple prompt, `generate_plan()` returns a dict matching `MoviePlan` schema; assert presence of `id`, `steps` and that each step has required fields.

  - Negative tests:
    - Malformed output: LLM returns non-JSON → assert `PlanParseError` raised
    - Schema mismatch: required field missing (e.g., no `steps`) → assert `PlanSchemaError`
    - Policy violation: prompt contains disallowed terms and LLM returns a plan → assert `PlanPolicyViolation` and that no artifacts are scheduled
    - Resource overage: LLM plans > configured frame budget → assert `PlanResourceError`

  - Test harness notes: use a lightweight LLM stub / fixture that returns canned outputs for deterministic testing. For integration smoke, run with a real LLM behind `SMOKE_ADK=1` and verify `generate_plan` returns a valid plan and the artifact publish call is invoked (mock `adk_helpers.publish_artifact` when appropriate).

  **Implementation tips**

  - Always run a strict schema validator (e.g., `pydantic` or `jsonschema`) on the parsed LLM output before accepting the plan.
  - When assembling few-shot examples include both a successful and a rejected example to guide structure and policy boundaries.
  - Persist the raw LLM output alongside the validated plan for auditing and debugging.


### tts_agent
- Task: Implement `tts_agent` decision layer and `tts_chatterbox` FunctionTool adapter
  - `tts_agent` (decision layer): responsible for provider selection, policy checks, retries/backoff, rate limiting, and orchestrating FunctionTool invocations. Public API: `synthesize(text: str, voice_config: dict) -> ArtifactRef` (returns a WAV artifact reference and metadata).
  - `tts_chatterbox` (FunctionTool adapter): compute-bound adapter that performs heavy TTS work (model load, synthesis, wav export). Implement as an ADK FunctionTool so the agent can call it; keep model lifecycle inside a guarded context manager to ensure safe load/unload.
  - Fallbacks: prefer ADK-managed TTS providers when available; fall back to local Coqui TTS for developer workflows. Entrypoints that instantiate agents must call `adk_helpers.require_adk()` or use `adk_helpers.probe_sdk(non_fatal=True)` where appropriate.
  - Metadata: publish duration, sample_rate, voice_id, format, seed (if applicable), and runtime info (model id, inference device, synth time). Publish WAV artifacts via `adk_helpers.publish_artifact()`.
  - Per-line synthesis: `production_agent` calls `tts_agent.synthesize()` once per dialogue line and records each WAV in `line_artifacts` alongside metadata (voice_id, provider voice id, duration_s, watermarked flag) so lipsync, QA, and assemble stages can trace every clip. Fixture mode (when `SMOKE_TTS`/`SMOKE_ADAPTERS` are unset) must still emit deterministic per-line artifacts.
  - Tests: unit tests for decision logic and metadata/format validation; integration smoke tests gated by `SMOKE_TTS=1` that exercise a small real or fixture TTS provider.
  - Estimate: 2–4 days
  - Notes: runtime dependency or manifest changes must follow the `proposals/pyproject_adk.diff` process and require explicit approval before editing `pyproject.toml` or pushing runtime changes.

  Provider selection rules (cost / latency / quality priorities)

  - Objective: choose a TTS provider (ADK-managed, local vendor like Chatterbox, or fixture stub) based on configured priorities and runtime signals.
  - Inputs considered for selection:
    - `priority_profile`: one of `cost`, `latency`, `quality`, or `balanced` (default `balanced`).
    - `max_latency_s`: caller hint for acceptable end-to-end latency.
    - `max_cost_usd`: soft cap per-synthesis request (if billing metadata available).
    - `required_features`: list of required capabilities (e.g., `voice_clone`, `multilingual`, `emotional_range`).
    - `smoke_flags`: `SMOKE_TTS`, `SMOKE_ADAPTERS` to gate heavy providers in dev/CI.
  - Selection algorithm (recommended):
    1. If `SMOKE_TTS`/`SMOKE_ADAPTERS` disabled, prefer local stub or fixture provider.
    2. Filter candidate providers by `required_features` — discard providers that lack required features.
    3. Rank candidates by profile:
       - `cost`: prefer cheapest provider whose `estimated_cost` <= `max_cost_usd` (if provided), then latency.
       - `latency`: prefer provider with lowest estimated latency (local/edge-first), then cost.
       - `quality`: prefer provider with highest historical mean MOS / model quality score, with cost/latency as tiebreakers.
       - `balanced`: weighted scoring of quality (0.5), latency (0.25), cost (0.25).
    4. If top candidate reports transient overload/unavailable, fall back to next candidate. If no provider meets constraints, return `ProviderSelectionError` with suggested relaxations.
  - Telemetry: record `selected_provider`, `score_breakdown`, `estimated_cost_usd`, `estimated_latency_s`, and `reason` in selection metadata.

  Provider catalog (`configs/tts_providers.yaml`)

  - Structure:
    - `version`: config schema version (current `v1`).
    - `priority_profiles`: named scoring weights for `quality|latency|cost`; `tts_agent` loads these to compute a weighted score.
    - `providers`: keyed by provider id; each entry specifies `display_name`, `tier` (`tier1|tier2|fixture`), adapter id, `fixture_alias`, defaults (voice, latency, cost, `quality_score`), capabilities (`features`, `languages`), per-provider `rate_limits` and `retry_policy`, plus a `watermarking` flag consumed by metadata tests.
    - `voices`: logical voices exposed to MoviePlans; each voice includes description, default audio settings, and a `provider_preferences` list mapping to provider-specific voice ids so the agent can fall back cleanly.
    - `rate_caps`: tier-wide caps (`daily_requests`, `concurrent_jobs`) referenced by the agent when honoring plan-level resource hints.
  - Current provider ids:
    - `chatterbox-pro` (tier1): Resemble/Chatterbox production path, watermark-on, supports `voice_clone|multilingual|emotional_range` with moderate latency/cost.
    - `adk-edge` (tier2): ADK-managed edge deployment (lower latency, watermark off) for multilingual + style control use cases.
    - `fixture-local` (fixture): deterministic WAV stub used whenever `SMOKE_TTS`/`SMOKE_ADAPTERS` are disabled; zero cost/high throughput guardrail.
  - Fixture aliases are consumed by tests and the FunctionTool entrypoint so local runs can force a deterministic provider (`fixture-chatterbox` / `fixture-tts`).
  - When adding providers, update both `providers` and `voices` sections so MoviePlan `voice_id`s stay portable; docs must describe new tiers/features before code depends on them.

  Voice mapping API & metadata

  - Purpose: provide a clear mapping from `voice_id` used in `MoviePlan`/step opts to provider-specific voice configurations and metadata.
  - Public helpers (suggested):
    - `get_voice_metadata(voice_id: str) -> VoiceMetadata` — returns a `VoiceMetadata` dataclass with fields below.
    - `list_available_voices(provider: Optional[str]=None) -> List[VoiceMetadata]` — list voices across providers or for a specific provider.

  - `VoiceMetadata` fields (example):
    - `voice_id: str` — logical id used in plans (e.g., `emma`).
    - `provider: str` — provider id/adapter name (e.g., `chatterbox`, `adk_tts`) or `stub`.
    - `display_name: str` — human-friendly name.
    - `language_codes: List[str]` — supported language tags (e.g., `['en-US']`).
    - `features: List[str]` — capabilities like `voice_clone`, `style_control`, `spanish_support`.
    - `sample_rate: int` — native sample rate.
    - `bit_depth: int` — e.g., 16, 24.
    - `watermarked: bool` — whether the provider applies watermarking by default.
    - `cost_estimate_usd_per_1s: float` — optional per-second cost estimate for decisioning.
    - `quality_score: float` — historical/benchmark quality metric (0..1) used in selection.

  - Mapping behavior:
    - Allow aliasing: a single logical `voice_id` can map to multiple provider voices with a provider preference order.
    - When publishing metadata, include both `voice_id` and resolved `provider_voice_id` to aid auditing.

  Error semantics (retryable vs non-retryable)

  - Objective: classify failure modes so `tts_agent` can decide whether to retry, switch provider, or fail fast.
  - Suggested error taxonomy:
    - `TTSRetryableError` — transient errors that should be retried (network timeouts, rate-limit 429, transient backend errors 5xx). Include `retry_after_s` if provider supplies it.
    - `TTSProviderUnavailable` — provider-side unavailability; treat as `retryable` for short backoff but also trigger provider-failover logic.
    - `TTSQuotaExceeded` — billing/quota errors; do not retry on same provider, attempt fallback provider if available; surface `PlanResourceError` if no fallback.
    - `TTSInvalidInputError` — non-retryable: invalid audio params, unsupported characters, voice_id unknown. Surface to caller as `BadRequest` / `PlanSchemaError` and do not retry.
    - `TTSPolicyViolation` — non-retryable: content moderation blocked; fail with `PlanPolicyViolation` and do not publish artifact.
    - `TTSServerError` — generic server error; treat as retryable up to `max_retries` then escalate.

  - Retry policy (recommended):
    - Default `max_retries`: 3 (with exponential backoff and jitter).
    - For `TTSQuotaExceeded` or `TTSInvalidInputError` do not auto-retry; perform provider failover only if alternative providers exist and meet `required_features`.
    - For `TTSRetryableError` and `TTSServerError` perform provider-local retries, then provider failover.

  Watermark handling & metadata tests

  - Providers may watermark TTS outputs. Agent must detect or be informed of watermarking behavior and record `watermarked: bool` in published metadata.
  - Test cases to add (unit):
    1. `test_tts_metadata_fields`: ensure `synthesize()` returns metadata containing `artifact_uri`, `duration_s`, `sample_rate`, `voice_id`, `provider`, `model_id`, `device`, `synth_time_s`, and `watermarked` (bool).
    2. `test_watermark_flag_propagated`: mock `tts_chatterbox` to return a `watermarked` flag and assert `tts_agent` includes it in published artifact metadata and telemetry.
    3. `test_retryable_vs_nonretryable`: simulate provider returning a `429` then `200` and assert retries occur; simulate `InvalidInput` and assert no retries and immediate `TTSInvalidInputError`.
    4. `test_provider_failover_on_quota`: simulate primary provider returning `TTSQuotaExceeded` and second provider succeeding; assert `tts_agent` switches providers and returns successful artifact.

  Test harness notes:
  - Use a fixture provider registry that exposes fake providers with programmable behavior (latency, cost, failure modes) to test selection and failover without real dependencies.
  - Gate heavy integration tests with `SMOKE_TTS=1`; integration smoke should verify artifact publish and watermark metadata when using a real provider.


#### Chatterbox upstream (Resemble AI)

Reference: Resemble AI's open-source Chatterbox TTS repository and model distribution. Use these links as the canonical upstream source when implementing the `tts_chatterbox` FunctionTool adapter:

- GitHub repo: `https://github.com/resemble-ai/chatterbox`
- Hugging Face model: `https://huggingface.co/ResembleAI/chatterbox`
- PyPI package: `https://pypi.org/project/chatterbox-tts/`
- Demo/Gradio pages: `https://resemble-ai.github.io/chatterbox_demopage/` and HF Spaces linked from the repo

Key install & quickstart notes (extracted from upstream README):

- Supported / tested: Python 3.11 on Debian; dependencies pinned in upstream `pyproject.toml`.
- Quick install (pip):

  ```bash
  pip install chatterbox-tts
  ```

- From-source (dev):

  ```bash
  conda create -yn chatterbox python=3.11
  conda activate chatterbox
  git clone https://github.com/resemble-ai/chatterbox.git
  cd chatterbox
  pip install -e .
  ```

Canonical usage examples (important for adapter wiring):

- English (single-language) quick example:

  ```python
  import torchaudio as ta
  from chatterbox.tts import ChatterboxTTS

  model = ChatterboxTTS.from_pretrained(device="cuda")
  text = "Hello world"
  wav = model.generate(text)
  ta.save("test-english.wav", wav, model.sr)
  ```

- Multilingual quick example (multilingual model):

  ```python
  import torchaudio as ta
  from chatterbox.mtl_tts import ChatterboxMultilingualTTS

  model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
  wav = model.generate("Bonjour tout le monde", language_id="fr")
  ta.save("test-fr.wav", wav, model.sr)
  ```

- Voice/Reference audio prompt support (voice cloning / voice transfer):

  ```python
  AUDIO_PROMPT_PATH = "ref.wav"
  wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
  ```

Upstream behavioral / tuning notes relevant to implementation:

- Model variants: Chatterbox (English-focused) and Chatterbox-Multilingual (23 languages). Both provide `.from_pretrained()` entrypoints.
- Useful knobs: `cfg_weight` (guides adherence to reference), `exaggeration` (controls emotion/intensity). Typical defaults: `exaggeration=0.5`, `cfg_weight=0.5`.
- If using reference clips from a different language, prefer `cfg_weight=0` to avoid accent carryover.
- Outputs are watermarked via Resemble's Perth watermarking; include watermark-awareness in metadata and policy notes.
- Upstream provides example scripts: `example_tts.py`, `example_vc.py`, `gradio_tts_app.py` — consult for runtime args and best-practice invocation patterns.

Implementation notes for `tts_chatterbox` FunctionTool adapter (from gathered upstream docs):

- Packaging & dependencies:
  - Upstream targets Python 3.11 and pins versions in `pyproject.toml`. Any repo `pyproject.toml` changes must be proposed via `proposals/pyproject_adk.diff` and approved before editing.
  - Consider offering a lightweight fixture-mode that imports `chatterbox-tts` only when `SMOKE_TTS=1` or in non-dev flows; otherwise use a local/fixture stub to avoid heavy install in CI.

- Model lifecycle & memory:
  - Use `model = ChatterboxTTS.from_pretrained(device=device)` to instantiate; support `device="cuda"` or `device="cpu"` and provide offload/quantization options where feasible.
  - Expose `audio_prompt_path` and `language_id` via FunctionTool `opts`.

- Reproducibility & tests:
  - Upstream examples use direct `.generate()` calls; to support deterministic tests, seed RNGs where the upstream API supports it or wrap calls with `torch.manual_seed()` where possible.
  - Create unit tests that stub the `generate()` return value and assert metadata fields (duration, sample_rate, voice_id, watermark flag).

- Metadata & publishing:
  - Publish fields: `artifact_uri`, `duration_s`, `sample_rate`, `voice_id`/`voice_name`, `model_id`, `device`, `synth_time_s`, `watermarked: bool`.
  - Upstream includes watermarking; include `watermarked` flag and provide optional extraction script references in docs.

- Security & policy:
  - The upstream repo includes a disclaimer—do not use for harmful content. Ensure the agent decision layer (`tts_agent`) enforces policy checks (content moderation) and logs policy events.

Where to look upstream for implementation details and examples:

- `example_tts.py` and `example_vc.py` in the upstream repo — copy or adapt invocation patterns and CLI args.
- `gradio_tts_app.py` — demonstrates server-style usage and runtime flags.
- `pyproject.toml` — see pinned dependency versions and Python requirement (3.11).

Recommendation: Use the upstream GitHub repo (`https://github.com/resemble-ai/chatterbox`) as the authoritative source for adapter wiring, and mirror these quickstart snippets into the `docs/IMPLEMENTATION_TASKS.md` as reference usage for implementers.

### images_agent
- Task: Implement `images_agent` decision layer (caller for `images_sdxl` DiffusersAdapter)
  - Public API: `render(prompt: str, opts: dict) -> list[ArtifactRef]` (validate opts, enforce rate limits and retries).
  - Responsibilities: content policy checks (NSFW, disallowed content), provider selection (local stub vs ADK-managed `images_sdxl`), request batching, and retry/backoff for transient failures.
  - Integration: call `DiffusersAdapter.render_images(...)` (FunctionTool) inside gated flows and record policy decisions via `adk_helpers.write_memory_event()`.
  - Tests: unit tests for policy logic and fallback behavior; gated integration smoke that exercises full adapter when `SMOKE_ADK=1`.
  - Estimate: 1–2 days

**Batching, rate-limits, and policy (detailed)**

- Batching rules (explicit)
  - `max_images_per_call` (default 8): the maximum number of images to request from `images_sdxl` in a single call. If a step requests `count > max_images_per_call`, chunk into multiple adapter calls.
  - `per_batch_timeout_s` (default 120s): per-call timeout for a single `images_sdxl` invocation. If the adapter does not respond within the timeout, mark the batch attempt as failed and apply retry/backoff.
  - Splitting strategy:
    - Chunk by `count`: split the `count` evenly across batches (last batch gets remainder).
    - For multi-prompt steps (array of prompts), group similar prompts together (same style/seed/palette) when possible to reduce prompt jitter and improve batching efficiency.
    - Preserve plan ordering semantics: produce outputs in the same step order and annotate artifact metadata with `batch_index` and `item_index` to allow correct reassembly.

- Timeout & per-request guardrails
  - Global per-call timeout: apply `per_batch_timeout_s` and cancel/cleanup the `gpu_utils.model_context` on timeout.
  - Soft deadline: start a background watchdog at 80% of `per_batch_timeout_s` to capture logs and partial telemetry before forceful cancellation.

- Rate-limiting / quota enforcement
  - Enforce a token-bucket rate limiter per-tenant/user key and per-host.
    - Default token refill: 60 tokens/min (configurable); each `images_sdxl` call costs `count` tokens.
  - Behavior on limit hit:
    - If `queue_allowed=True` (agent config): enqueue the step with TTL (default 600s) and return a queued `StepExecutionRecord` with status `queued`.
    - If `queue_allowed=False`: return an immediate `RateLimitError` to the caller (agent should surface to user/UI).
  - Backpressure: when system-wide backlog exceeds `step_queue_backlog_limit`, start rejecting new plans with `PlanResourceError` to preserve stability.

- Canonical policy checks (concretized)
  - Textual prompt checks (fast, pre-render): run a lightweight text-moderation pass (policy LLM or regex-based classifier) on `prompt`/`negative_prompt`/`text` fields.
    - If `disallowed` → raise `PlanPolicyViolation` and do not render.
    - If `ambiguous` → mark for QA or apply stricter generation constraints.
  - Post-render image checks (recommended): always run `qa_qwen2vl.inspect_frames()` on outputs in `run` mode before publishing.
    - Post-render thresholds (example defaults):
      - `nsfw_confidence >= 0.70` → `reject` (delete artifact, mark step failed)
      - `nsfw_confidence >= 0.40 and < 0.70` → `escalate` (request human review via `request_human_input()` and mark artifact `quarantined`)
      - `< 0.40` → `approve`
    - The `qa_qwen2vl` call should pass a prompt like: `"Does this image contain sexual content, minors, graphic violence, or other policy-prohibited material? Provide a confidence score for each category."` and the agent should parse the structured `QAReport` to decide.
  - Duplicate detection / hashing:
    - Use perceptual hashing (pHash or `imagehash`) to detect duplicates and near-duplicates against a recent-artifacts index (sliding window, e.g., last 1000 artifacts).
    - Consider duplicates within a plan as `dedupe` (skip publishing duplicates) if `dedupe=True` in `opts`.
    - pHash threshold: Hamming distance <= 6 → treat as duplicate; values tuned per data.

- Pre-render QA via `qa_qwen2vl` (how/when)
  - For prompts that contain image references (URLs/base64) or where the user included reference images, call `qa_qwen2vl.inspect_frames()` on the referenced images with textual prompts (e.g., `"Is this reference image safe to base synthetic output on?"`). If QA flags issues, abort rendering the step and surface `PlanPolicyViolation`.
  - For text-only prompts, prefer a text moderation model first. If textual output is borderline and `SMOKE_QA=1`, optionally run a small sample render (sandbox) of 1 image with low-cost settings and run `qa_qwen2vl` on that sample before committing to full `count` renders.

- Fallback & remediation flows
  - On `reject`: surface failure to caller with `PlanPolicyViolation` and no publish.
  - On `escalate`: persist outputs to a quarantine area and call `request_human_input()` with a link to the artifact and the `QAReport`.
  - On duplicate detection: if `dedupe=True`, replace duplicate artifact URIs with the canonical artifact and mark the step as `succeeded (deduped)`.

**Example `opts` schema for `images_agent` → `images_sdxl`**

- `ImagesOpts` (example fields):
  - `seed: Optional[int]`
  - `num_images: int` (default 1)
  - `width: int`, `height: int`
  - `sampler: str` (e.g., "ddim", "euler")
  - `steps: int` (inference steps)
  - `cfg_scale: float`
  - `denoising_start: Optional[float]`, `denoising_end: Optional[float]`
  - `prompt_2: Optional[str]`, `negative_prompt_2: Optional[str]`
  - `max_images_per_call: Optional[int]` (overrides default)
  - `dedupe: bool` (default False)
  - `queue_allowed: bool` (default True)
  - `priority: Literal["low","normal","high"]` (scheduling hint)

**Deterministic test harness (stub pipeline)**

- Purpose: provide a lightweight, deterministic stub to test batching, rate-limiting, policy, and duplicate detection without heavy model deps.

- Stub characteristics:
  - Deterministic outputs based on `(prompt, seed, index)` — e.g., return a small 16x16 PNG derived from a seeded PRNG or a hash of prompt+seed.
  - Provide metadata fields (`seed`, `prompt`, `width`, `height`) and predictable pHash values to test duplicate detection.

#### Concrete file & function signatures (copy into code)

- Implement `src/sparkle_motion/images_agent.py` with public function:

```py
from typing import List
from sparkle_motion.types import ImagesOpts

def render(prompt: str, opts: ImagesOpts) -> List[dict]:
  """Render images according to opts and return ordered ArtifactRef list."""

```

- Adapter stub expected at `function_tools/images_sdxl/entrypoint.py`:

```py
from typing import Optional, List

def render_images(prompt: str, count: int, seed: Optional[int], opts: dict) -> List[dict]:
  """Return list of dicts: {'data': bytes, 'metadata': {...}}"""

```

- QA stub: `function_tools/qa_qwen2vl/entrypoint.py`:

```py
def inspect_frames(frames: List[bytes], prompts: List[str]) -> dict:
  return {'ok': True, 'reason': None, 'report': {}}

```

- DB helper: `src/sparkle_motion/db/sqlite.py` (small API):

```py
import sqlite3
from pathlib import Path

def get_conn(path: str) -> sqlite3.Connection:
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(str(p), timeout=5.0)
  conn.row_factory = sqlite3.Row
  return conn

def ensure_schema(conn: sqlite3.Connection, ddl: str) -> None:
  conn.executescript(ddl)
  conn.commit()

```

- RecentIndex implementation task: `src/sparkle_motion/utils/recent_index_sqlite.py` providing:
  - `get_canonical(phash: str) -> Optional[str]`
  - `add_or_get(phash: str, uri: str) -> str`
  - `prune(max_age_s: int, max_entries: int) -> None`

#### Tests to add (exact file names)

- `tests/unit/test_images_agent.py` — batching, ordering, within-plan dedupe, global dedupe (SQLite), QA rejection.
- `tests/unit/test_recent_index_sqlite.py` — get/add/prune behavior.
- `tests/unit/test_rate_limiter.py` — token bucket semantics.
- `tests/unit/test_adk_helpers.py` — `publish_local()` deterministic URIs in fixture mode.

#### Deterministic adapter stub guidance

- Use this deterministic byte generator in the stub so tests can assert exact values:

```py
import hashlib

def deterministic_bytes(prompt: str, seed: int, index: int) -> bytes:
  key = f"{prompt}|{seed}|{index}".encode()
  return hashlib.sha256(key).digest()[:256]  # truncated deterministic blob

```

- Tests to implement with stubbed pipeline:
  1. `test_batch_split_and_ordering`: request `num_images=20` with `max_images_per_call=8` and assert that the agent makes 3 adapter calls, preserves ordering, and returns 20 artifacts with proper `batch_index` and `item_index` metadata.
  2. `test_rate_limit_queueing`: configure token-bucket with 2 tokens and request 4 images in quick succession with `queue_allowed=True`; assert two are executed immediately and two are queued and eventually executed within TTL.
  3. `test_policy_text_rejects_prompt`: mock text-moderation to mark prompt disallowed; assert `PlanPolicyViolation` and no adapter calls.
  4. `test_post_render_nsfw_rejects`: use stub pipeline to emit an output with a pHash known to correspond to `nsfw_confidence=0.9` (inject via QA stub) and assert the artifact is rejected and not published.
  5. `test_deduplicate_within_plan`: produce two identical deterministic stub images and assert a single published artifact when `dedupe=True` and returned artifact list references canonical URI twice.

**Implementation notes**
  - Keep `images_agent` logic separate from adapter invocation: build a small orchestration layer that prepares batches, enforces rate-limits, runs policy checks, and calls `DiffusersAdapter.render_images()`.
  - Persist `pHash` and recent-artifact index in a lightweight key-value store (e.g., Redis) or an in-process LRU for dev; expose TTL and size limit in config.
  - Surface policy decisions via `adk_helpers.write_memory_event()` with the `QAReport` attached for auditing.


### videos_agent
- Task: Implement `videos_agent` decision layer (caller for `videos_wan`/video adapters)
  - Public API: `render_video(start_frames: Iterable[Frame], end_frames: Iterable[Frame], prompt: str, opts: dict) -> ArtifactRef`.
  - Responsibilities: select video provider (WanAdapter vs queued offline renderer), manage chunking/segmentation for VRAM safety, orchestrate retries/backoff, and enforce video-specific policy checks (copyright, prohibited content).
  - Integration: coordinate with `WanAdapter`/`videos_wan` FunctionTool for heavy inference inside `gpu_utils.model_context` and publish assembled artifacts via `adk_helpers.publish_artifact()`.
  - Tests: unit tests for orchestration logic and chunking; gated integration smoke when `SMOKE_ADK=1`.
  - Estimate: 2–4 days
  
  Chunking & Reassembly (algorithm parameters)

  - Goal: split long video renders into manageable chunks to avoid OOM, preserve temporal continuity, and enable parallel execution across devices when available.
  - Parameters (defaults):
    - `chunk_length_frames`: 64 — target number of frames per chunk. Tune per-host (lower for low-VRAM GPUs).
    - `chunk_overlap_frames`: 4 — number of overlapping frames between adjacent chunks to smooth transitions and allow temporal blending.
    - `min_chunk_frames`: 8 — do not produce chunks smaller than this (instead merge with adjacent chunk).
    - `max_retries_per_chunk`: 2 — adaptive retries for transient failures before applying fallback strategies.
    - `adaptive_shrink_factor`: 0.5 — on OOM reduce `chunk_length_frames` by this factor for the retry.
  - Behavior:
    - Chunk split: compute N chunks = ceil(num_frames / chunk_length_frames). For each chunk compute `start_frame` and `end_frame` inclusive. For chunks other than first/last, extend `start_frame` backwards by `chunk_overlap_frames` and `end_frame` forwards by `chunk_overlap_frames` to create overlap for blending.
    - Reassemble semantics: after pipeline returns chunk frames, trim overlap regions using a deterministic policy (e.g., keep leading overlap from earlier chunk, trailing overlap from later chunk) or perform a simple crossfade in the overlap region (configurable via `reassembly.mode` = `trim|crossfade`).
    - Ordering: preserve original temporal order when returning artifacts; each chunk should include metadata `chunk_index`, `chunk_start_frame`, `chunk_end_frame`, `overlap_left`, `overlap_right` so the assembler can recompose reliably.
    - Deterministic seeds: if a `seed` is used, derive per-chunk seeds deterministically (e.g., `chunk_seed = hash(seed, chunk_index)`) to ensure reproducibility.

  Multi‑GPU / Sharding Strategies

  - Objectives: provide recommended device_map and memory presets for common host types and describe when to prefer balanced sharding vs sequential/offload.
  - Preset examples:
    - A100 multi‑GPU (balanced sharding):
      - `device_map="balanced"` with `max_memory` mapping (e.g., {0: "74GiB", 1: "74GiB", "cpu": "120GiB"}) and `low_cpu_mem_usage=True`.
      - Use `pipeline = WanImageToVideoPipeline.from_pretrained(..., device_map="balanced", max_memory=max_memory, low_cpu_mem_usage=True)`.
    - Single high-memory GPU (A6000/80GB style):
      - `device_map="auto"` or single-device `.to("cuda:0")` and enable `offload` only if needed.
    - Consumer GPUs (4090/3090):
      - Prefer `device_map="sequential"` or explicit offload using `accelerate`/`enable_model_cpu_offload()`; reduce `chunk_length_frames` and prefer smaller `height/width` micro-batches.
    - CPU-fallback: for development or small jobs, set `device="cpu"` and reduce `num_frames`/`H/W`; this is slow but avoids GPU OOM and is a documented fallback path.
  - Sharding advice:
    - Try `balanced` on multi‑A100 hosts where each GPU has large RAM. Tune `max_memory` per-host using vendor guidance.
    - For heterogeneous clusters, prefer explicit `device_map` assignments to avoid scheduler surprises.
    - Always expose `device_map` and `max_memory` as configurable options in adapter `opts` and record them in step metadata for reproducibility.

  OOM & Fallback Strategies (formalized)

  - Strategy precedence on chunk-level failure (recommended):
    1. If chunk fails with transient error (network, grpc timeout) → retry up to `max_retries_per_chunk` with exponential backoff + jitter.
    2. If failure indicates OOM on GPU: one adaptive retry where `chunk_length_frames` is multiplied by `adaptive_shrink_factor` (round up to `min_chunk_frames`) and the chunk is retried on the same device.
    3. If repeated OOMs or reduced-chunk attempt fails: attempt a device fallback sequence:
       - Try to run the chunk on a different GPU with more free memory (if available).
       - If no GPU fallback available, attempt `cpu` fallback path (documented caveats: CPU is slower and may need memory/time limits); mark the step as `cpu_fallback=true` in metadata.
    4. If CPU fallback is not permitted or fails, mark chunk as `failed` with `failure_reason=OOM` and include diagnostic metadata (last attempted device, peak_vram_mb, stack/log excerpts). Optionally escalate to human review or mark plan as partially failed depending on `production_agent` policy.
  - Adaptive retries: on OOM-triggered adaptive retry reduce `chunk_length_frames` progressively (cap to a small number of attempts to avoid long retries). Record each attempt's `attempt_index`, `attempt_chunk_length_frames`, `device`, and `outcome` in step telemetry.

  Progress / Callback Contract

  - Purpose: give callers a stable, minimal contract for receiving progress updates from `videos_wan` and `videos_agent` during long-running renders.
  - Contracts:
    - Adapter callback shape (from `videos_wan` pipeline): adapter should accept optional `callback` and `callback_steps` kwargs. When supported emit events at `callback_steps` intervals with a `CallbackEvent` shape.
      - `CallbackEvent` fields: `plan_id`, `step_id`, `chunk_index`, `frame_index`, `num_frames`, `progress` (0.0-1.0), `eta_s` (optional), `device`, `phase` (one of `load`, `rendering`, `postprocess`).
    - `videos_agent` should expose two ways to surface progress to callers:
      1. Synchronous callback parameter: `execute_plan(..., on_progress: Optional[Callable[[CallbackEvent], None]])` — call this synchronously (or via executor) on every adapter event.
      2. Event stream/async channel: publish progress events to an event bus (e.g., websocket/topic or in-process queue) and write an audit memory event via `adk_helpers.write_memory_event()` for durable timeline entries.
    - StepExecutionRecord must include aggregated progress fields: `started_at`, `last_progress_at`, `progress_percent`, `current_chunk`, `completed_chunks`, and `logs_uri`.
  - Best practices:
    - Emit coarse events frequently (e.g., every 1–2 seconds or every `callback_steps`) and only emit heavy telemetry at important state transitions (chunk start/finish, retry, fallback, fail, success).
    - Provide an `unsubscribe` handle when the caller supplies a long-lived subscription (avoid memory leaks).

  Tests (unit + gated integration)

  - Unit tests to implement (file: `tests/unit/test_videos_agent.py`):
    1. `test_chunk_split_and_reassembly`: given `num_frames=150`, `chunk_length_frames=64`, `chunk_overlap_frames=4`, assert computed chunk ranges, overlaps and reassembly trimming produce correct global frame ordering and that per-chunk metadata is populated.
    2. `test_adaptive_oom_retry_shrinks_chunk`: simulate OOM on first attempt and assert agent reduces chunk length by `adaptive_shrink_factor` and retries (mock `videos_wan` to raise OOM on first call, succeed on second).
    3. `test_cpu_fallback_on_oom`: simulate persistent GPU OOM and assert agent attempts CPU fallback and records `cpu_fallback=true` in metadata (mock pipeline behavior accordingly).
    4. `test_progress_events_forwarded`: mock adapter to emit `CallbackEvent` and assert `videos_agent` forwards events to `on_progress` callback and writes memory events.
  - Gated integration smoke (to run behind env flags):
    - `tests/integration/test_videos_wan_smoke.py` gated by `SMOKE_ADK=1` that submits a short FLF2V job (`num_frames=8`, low `H/W`, reduced steps) and asserts artifact publish and basic progress events.
  - Test harness notes:
    - Use deterministic generator seeds and a small pipeline stub for unit tests to avoid heavy deps.
    - For integration smoke, provide a minimal plan and gate with `SMOKE_ADK=1` and `SMOKE_ADAPTERS=1` to ensure adapters run only when explicitly enabled.

## FunctionTool / Adapter Tasks

These tasks cover compute-bound adapters (FunctionTools) that perform heavy model loading, inference, and artifact publishing. Adapters must be written to run safely inside `gpu_utils.model_context` and to publish deterministic artifacts.

**Cross-cutting: `gpu_utils.model_context` (API & checklist)**

- Purpose: a lightweight context manager that standardizes model lifecycle (load, optional offload, optional compile), exposes device mapping overrides for sharded pipelines, and guarantees cleanup and telemetry on exit. Adapters must use this context to avoid VRAM leakage and ensure reproducible behavior across different hardware.

- Precise API surface (suggested):

```python
from contextlib import contextmanager
from typing import Optional, Mapping

@contextmanager
def model_context(
    model_key: str,
    *,
    weights: Optional[str] = None,
    offload: bool = True,
    xformers: bool = True,
    compile: bool = False,
    device_map: Optional[Mapping[int,str]] = None,
    low_cpu_mem_usage: bool = True,
    max_memory: Optional[Mapping[str,str]] = None,
    timeout_s: Optional[int] = None,
):
    """Context manager for guarded model loading.

    Yields: a `ModelContext` object with helpers: `pipeline`, `device_map`, `allocated_devices`.
    """
    ...
```

- Parameter semantics:
  - `model_key` (str): logical identifier for telemetry and caching (e.g., `wan2.1/flf2v`).
  - `weights` (str|Optional): HF id or path; if omitted use configured default for `model_key`.
  - `offload` (bool): enable CPU/GPU offload helpers (accelerate / device/offload APIs).
  - `xformers` (bool): attempt to enable xformers memory-efficient attention when available.
  - `compile` (bool): hint to attempt `torch.compile()` where supported to speed up inference.
  - `device_map` (Optional[Mapping]): explicit device map override for sharded pipelines (pass-through to HF/transformers/device_map APIs).
  - `low_cpu_mem_usage` (bool): pass `low_cpu_mem_usage=True` to HF loaders when helpful to reduce peak host RAM.
  - `max_memory` (Optional[Mapping[str,str]]): explicit memory budget hints used for `device_map="balanced"` loads.
  - `timeout_s` (Optional[int]): optional time limit for model load; if exceeded raise `ModelLoadTimeout`.

- ModelContext object (returned from context): fields and helpers:
  - `pipeline` / `model` : the loaded pipeline or model instance (None until loaded).
  - `device_map`: resolved device map used for this load.
  - `allocated_devices`: list of devices that hold model state.
  - `report_memory()` method: returns snapshot `{device: {'used_mb':int,'total_mb':int}}`.

- Mandatory cleanup checklist (adapters must honor this on exit):
  1. Delete references to large objects: `del pipeline`, `del model`, `del vae`, etc.
  2. Call `torch.cuda.synchronize()` (when using CUDA) to flush async work.
  3. Call `torch.cuda.empty_cache()` to release cached blocks.
  4. Call `gc.collect()` to run Python GC and free host allocations.
  5. If offloading helpers were enabled, call their cleanup hooks (`accelerate` or offload APIs) to release CPU pinned memory.
  6. Return a final memory snapshot via `report_memory()` and include it in telemetry.

- How to surface OOMs to callers:
  - During `with model_context(...)` block, any exception from model load or inference that appears as an OOM should be normalized to a domain exception type, e.g., `ModelOOMError` with structured metadata:
    - `model_key`, `weights`, `attempted_device_map`, `peak_vram_mb` (if available), `traceback_snippet`.
  - The context manager should capture CUDA OOM errors (`RuntimeError` containing "out of memory") and wrap them as `ModelOOMError` before re-raising so callers can implement adaptive fallback strategies (shrink batch/chunk, switch device, CPU fallback).
  - For load-time OOMs, include `stage='load'`, for runtime/inference OOMs include `stage='inference'`.

- Memory telemetry & reporting (recommended):
  - Emit telemetry at these points:
    1. `load_start`: record `model_key`, `weights`, `device_map_hint`, `timestamp`.
    2. `load_complete`: record `allocated_devices`, `peak_vram_mb_estimate`, `load_duration_s`.
    3. `inference_start` (per-call or per-chunk): record `input_shape`, `num_frames` (video), `batch_size`, `seed`.
    4. `inference_end`: record `inference_time_s`, `peak_vram_mb`, `gpu_utilization` (if available), `allocated_devices`.
    5. `cleanup`: final `report_memory()` and `gc_collect_duration_s`.
  - Telemetry format: structured dict with `model_key`, `step_id`, `plan_id` (when provided), `device_stats` mapping, and optional `cuda_metrics` if host supports NVML.
  - Use `adk_helpers.write_memory_event()` or the project's telemetry API to persist these events.

- Example usage patterns for adapters

1) Simple single-device pipeline (consumer GPU):

```python
from sparkle_motion.gpu_utils import model_context

with model_context('sdxl', weights='stabilityai/sdxl-base-1.0', offload=False, xformers=True) as ctx:
    pipe = ctx.pipeline
    gen = torch.Generator(device='cuda').manual_seed(seed)
    images = pipe(prompt, generator=gen, num_inference_steps=opts.steps)
# context exit ensures cleanup and telemetry emit
```

2) Wan2.1 balanced sharding example (multi‑GPU A100):

```python
max_memory = {0: '74GiB', 1: '74GiB', 'cpu': '120GiB'}
with model_context('wan2.1/flf2v', weights=MODEL_ID, offload=True, device_map=None, max_memory=max_memory) as ctx:
    pipe = ctx.pipeline
    frames = pipe(..., callback=progress_cb, callback_steps=1)
```

3) Defensive adapter pattern (handle OOM and fallback):

```python
try:
    with model_context(... ) as ctx:
        result = ctx.pipeline(...)
except ModelOOMError as e:
    # record attempt, shrink batch/chunk and retry or escalate
    log.warning('OOM for %s on %s', e.model_key, e.attempted_device_map)
    raise
```

- Implementation checklist for `gpu_utils.model_context` maintainers
  - [ ] Provide idiomatic context manager compatible with `with` and `async with` patterns if adapters use async.
  - [ ] Normalize exceptions: `ModelLoadTimeout`, `ModelOOMError`, `ModelLoadError` with structured fields.
  - [ ] Expose `report_memory()` that tries CUDA queries first, then falls back to host-process `/proc` (best-effort; avoid crashing if metrics unavailable).
  - [ ] Integrate optional NVML support (but fail gracefully if NVML not installed); gate NVML metrics behind a capability check.
  - [ ] Emit structured telemetry events via `adk_helpers.write_memory_event()` at `load_start`, `load_complete`, `inference_start`, `inference_end`, and `cleanup`.
  - [ ] Provide a simple helper `suggest_shrink_for_oom(attempt_state) -> new_chunk_size` to centralize adaptive-shrink heuristics used by higher-level agents (videos_agent, images_agent).
  - [ ] Document platform expectations and common device_map presets for A100/4090 hosts.

- Notes & guidance
  - The context must not swallow exceptions silently — always re-raise after emitting telemetry and performing cleanup.
  - Keep the context implementation minimal and robust: prefer best-effort metrics (non-critical) and deterministic cleanup (critical).
  

### images_sdxl
- Task: Implement `render_images(prompt, opts) -> list[ArtifactRef]`
  - Add `gpu_utils.model_context('sdxl', weights=...)` context manager for load/unload.
  - Implement a `DiffusersAdapter` that instantiates the pipeline inside the context and performs deterministic sampling options.
  - Publish outputs as PNG artifacts and create per-shot metadata (seed, prompt, sampler).
  - Tests: unit tests using a tiny pipeline stub; ADK-gated smoke for real pipeline.
  - Estimate: 3–4 days (model paging and memory checks included)

  SDXL-specific guidance (from Hugging Face Diffusers docs):
  - Use the official SDXL pipelines (e.g., `StableDiffusionXLPipeline` / `AutoPipelineForText2Image`).
  - Load models with `torch_dtype=torch.float16`, `variant="fp16"` and `use_safetensors=True` when available, and move to CUDA: `pipeline = StableDiffusionXLPipeline.from_pretrained(..., torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")`.
  - Support the two-stage base+refiner workflow: either run base→refiner as an ensemble (use `denoising_end` / `denoising_start`) or call the refiner on the base output to improve quality.
  - Prefer `from_pipe` when composing pipelines (saves memory) and `from_single_file` for single-file checkpoints when appropriate.
  - Provide `prompt_2` / `negative_prompt_2` support for the second text encoder to improve prompt adherence.
  - Deterministic sampling: accept a `seed` parameter and use `torch.Generator(device).manual_seed(seed)` when calling the pipeline so outputs are reproducible for tests.
  - Memory optimizations to expose in `gpu_utils.model_context`:
    - `enable_model_cpu_offload()` for OOM avoidance.
    - `enable_xformers_memory_efficient_attention()` when `xformers` is available.
    - optional `torch.compile` for speed-ups when `torch>=2.0`.
  - Respect SDXL micro-conditioning options (e.g., `original_size`, `target_size`, crop conditioning) and expose them as `opts` where useful for shot-level control.
  - Document that heavy/real runs must be gated by `SMOKE_ADK=1` and that manifest/runtime deps must follow the repo proposal process.

Implementation checklist & notes (practical details)

- **Public API**: implement `render_images(prompt: str, opts: dict) -> list[ArtifactRef]` on `DiffusersAdapter` and expose a thin wrapper used by `images_agent` decision layer.
- **GPU context**: `gpu_utils.model_context('sdxl', *, weights: str|Path, offload: bool=True, xformers: bool=True, compile: bool=False)` must:
  - load model with suggested args (`torch_dtype=torch.float16`, `use_safetensors=True`) and call `.to("cuda")` for inference.
  - enable `accelerate`/offload helpers (e.g., `enable_model_cpu_offload()`), enable `xformers` memory friendly attention when available, and optionally call `torch.compile()` when requested and supported.
  - ensure explicit cleanup on exit: delete pipeline, call `torch.cuda.empty_cache()`, and release CUDA context to avoid leaking VRAM between invocations.

- **Adapter responsibilities** (`DiffusersAdapter`):
  - Accept `prompt: str`, `opts: ImagesOpts` (see schema below), and an optional `device` override.
  - Create a reproducible `torch.Generator(device)` when `seed` is provided and pass it to the pipeline call.
  - Support base+refiner flow: either run base→refiner or call refiner separately on base outputs to improve quality. Expose `denoising_start`/`denoising_end` in `opts`.
  - Save outputs as PNG files (sRGB) and return `ArtifactRef` objects with metadata: `seed`, `sampler`, `prompt`, `model_id`, `device`, `width`, `height`, `steps`, `scheduler`.
  - Ensure outputs are written to a temp dir and published using `adk_helpers.publish_artifact()` (with file:// fallback in local dev/test modes).

- **Options schema (suggested)**
  - `ImagesOpts` (example fields):
    - `seed: int | None`
    - `num_images: int` (default 1)
    - `width: int`, `height: int`
    - `sampler: str` (e.g., "ddim", "ddpm", "euler")
    - `steps: int`
    - `cfg_scale: float`
    - `denoising_start: float | None`, `denoising_end: float | None` (for base+refiner control)
    - `prompt_2`, `negative_prompt_2` (second text encoder)
    - `original_size`, `target_size`, `crop_conditioning` (micro-conditioning)

- **Deterministic sampling example**

```python
import torch

gen = torch.Generator(device="cuda").manual_seed(seed)
images = pipeline(prompt, num_images_per_prompt=opts.num_images, generator=gen, guidance_scale=opts.cfg_scale, num_inference_steps=opts.steps)
```

- **Pipeline load example**

```python
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipeline.to("cuda")
if xformers_available:
    pipeline.enable_xformers_memory_efficient_attention()
```

- **Memory & runtime hygiene**
  - Use `with gpu_utils.model_context('sdxl', weights=model_id, offload=True, xformers=True):` in entrypoints that call the adapter.
  - After inference: `del pipeline; torch.cuda.empty_cache()` and consider calling `gc.collect()`.
  - Instrument VRAM usage in smoke tests and fail fast with clear error messages if memory thresholds are exceeded.

- **Publish artifacts**

```python
from sparkle_motion.adk_helpers import publish_artifact

artifact_uri = publish_artifact(path=png_path, media_type="image/png", metadata=meta_dict)
```

- **Tests**
  - Unit: stub the pipeline object (small lightweight fake) that returns a deterministic PIL image or numpy array; assert `render_images` returns correctly shaped PNG artifacts and metadata.
  - Integration (gated): `SMOKE_ADK=1` test that instantiates a tiny model (or CI-approved model) in `gpu_utils.model_context` and verifies outputs and publish path.
  - Add a test that simulates OOM and asserts cleanup path runs (e.g., monkeypatch pipeline to raise OOM and assert `torch.cuda.empty_cache()` called).

- **Manifest & dependency proposal**
  - Proposed runtime deps (for `proposals/pyproject_adk.diff`): `torch>=2.0+cu*`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `bitsandbytes` (optional), `xformers` (optional), `huggingface_hub`.
  - Document expected CUDA/torch combos in the proposal (cu118 vs cu120) and recommend pinned minor versions used in validated CI runners.

- **Telemetry & metadata**
  - Record: `model_id`, `device`, `inference_time_s`, `peak_vram_mb` (when available), `seed`, and `opts` snapshot.
  - Include these fields in published artifact metadata to aid reproducibility and debugging.

- **Security & policy**
  - `images_agent` must perform content policy checks (NSFW, disallowed content) before invoking `DiffusersAdapter`. Log policy decisions and memory events via `adk_helpers.write_memory_event()`.

### videos_wan (pilot)
- Task: Implement `run_wan_inference(start_frames, end_frames, prompt, opts) -> ArtifactRef` for Wan2.1-style pipelines (FLF2V / I2V / T2V flows).
  - Purpose: provide a safe, gated adapter (`WanAdapter` / `videos_wan` FunctionTool) that can run Wan2.1 First‑Last‑Frame→Video (FLF2V) and related pipelines via Diffusers and publish video artifacts.
  - Model reference: `Wan-AI/Wan2.1-FLF2V-14B-720P` (Hugging Face Wan2.1 model family). Use the official HF model card and the Wan repo for runtime knobs and multi‑GPU strategies.

  - Key implementation notes and sample code (adapted from Wan2.1 notebook):

```python
# Example: load FLF2V pipeline (balanced sharding example)
import os, torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers import DPMSolverMultistepScheduler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MODEL_ID = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"

image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)

# Balanced sharding example (good for large A100 hosts)
max_memory = {0: "74GiB", "cpu": "120GiB"}  # tune per-host
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    image_encoder=image_encoder,
    device_map="balanced",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)

# Replace processor if needed
from transformers import CLIPImageProcessor
if not isinstance(getattr(pipe, "image_processor", None), CLIPImageProcessor):
    pipe.image_processor = CLIPImageProcessor.from_pretrained(MODEL_ID, subfolder="image_processor")

# Use a faster scheduler (dpmsolver++ / UniPC depending on model)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++")
```

  - Inference flow & helpers (notes):
    - Resize / crop inputs using the pipeline's `vae_scale_factor_spatial` and `pipe.transformer.config.patch_size` to compute valid (mod-aligned) width/height; Wan model docs and the notebook include `aspect_ratio_resize()` and `center_crop_resize()` helpers — reuse these in `videos_wan` to ensure valid dimensions.
    - For deterministic tests accept a `seed` and create `generator = torch.Generator(device).manual_seed(seed)` and pass `generator=generator` to the pipeline call.
    - Support callback hooks for progress and ETA: detect `callback`, `callback_on_step_end`, `callback_steps`, and `callback_on_step_end_tensor_inputs` in the pipeline signature and wire lightweight progress callbacks for smoke tests and local runs.
    - Call pipeline with `output_type="pil"` (or `numpy`) to get frames back, then assemble to MP4 using `diffusers.utils.export_to_video()` and fallback to `imageio`/`ffmpeg` if export fails.

  - Example inference call (simplified):

```python
result = pipe(
    image=first_frame,
    last_image=last_frame,
    prompt=prompt,
    negative_prompt=negative,
    height=H, width=W,
    num_frames=num_frames,
    num_inference_steps=steps,
    guidance_scale=guidance,
    generator=generator,
    output_type="pil",
    **callback_kwargs,
)
frames = getattr(result, "frames", getattr(result, "images", None))[0]
```

  - Memory & runtime hygiene:
    - Implement `gpu_utils.model_context("wan2.1", weights=MODEL_ID, offload=True, xformers=True)` that loads the pipeline safely and enforces explicit cleanup (del pipeline; torch.cuda.empty_cache(); gc.collect()).
    - Provide `balanced` device_map examples for A100 (tune `max_memory` and `low_cpu_mem_usage`) and `sequential`/offload examples for single‑GPU (4090/3090) via the Wan repo guidance.
    - Document recommended GPU budgets (e.g., A100-80GB balanced sharding settings from the notebook) and fail fast with descriptive OOM errors.

  - Publish & artifact handling:
    - Save a temporary MP4 (or PNG frames) and call `adk_helpers.publish_artifact(path=out_path, media_type="video/mp4", metadata=meta)` with metadata including `model_id`, `device`, `seed`, `inference_time_s`, `peak_vram_mb` (if available), and `plan_step`.

  - Tests:
    - Unit: stub `WanImageToVideoPipeline` to return deterministic PIL frames; assert `run_wan_inference` returns a valid artifact ref and metadata.
    - Integration (gated): `SMOKE_ADK=1` smoke that runs a short FLF2V job (e.g., small `max_area`, low `num_frames`, reduced `steps`) on approved hardware and verifies publish path and artifact playback.

  - Manifest / dependency proposal (for `proposals/pyproject_adk.diff`):
    - Suggested runtime deps: `diffusers` (with Wan pipeline support), `transformers`, `accelerate`, `safetensors`, `imageio`, `imageio-ffmpeg`, `torch>=2.0`, `torchvision`, `huggingface_hub`.
    - Optional: `xfuser` / `TeaCache` / `TeaCache`-like accelerators or vendor libs for multi‑GPU; document exact CUDA/torch combos (cu118/cu120) in the proposal.

  - Safety & policy:
    - `videos_agent` or `production_agent` must run content policy checks before invoking `videos_wan`. Log policy decisions and escalate via `adk_helpers.write_memory_event()` when needed.

  - Estimate: 1–2 weeks for a robust, multi‑host aware pilot (including multi‑GPU validation); shorter (3–5 days) for a dev-only FunctionTool stub and gated smoke tests.


### lipsync_wav2lip
- Task: Implement `run_wav2lip(video_path, audio_path, out_path, *, opts)`
  - Purpose: provide a safe, testable adapter that lip-syncs a target video to input audio using the open-source Wav2Lip implementation. The adapter should support both using the Wav2Lip Python API (when vendored/installed) and a pinned-subprocess fallback that runs a known commit of the upstream repo.

  - Key upstream references (canonical):
    - GitHub: `https://github.com/Rudrabha/Wav2Lip`
    - Colab / notebooks and demo links referenced from upstream README (for example quickstart and inference patterns).

  - Prerequisites (from upstream README):
    - `Python 3.6+` (project historically used 3.6; prefer modern 3.10+ within repo policies).
    - `ffmpeg` binary available on PATH (CI images or container must provide it).
    - Face detector weight `face_detection/detection/sfd/s3fd.pth` (upstream link or mirrored internal location); `face_alignment` model support expected.
    - Python dependencies: `torch`, `opencv-python`, `numpy`, `scipy`, `face-alignment` (or `face_alignment`), and any other packages listed in upstream `requirements.txt` (pin via `proposals/pyproject_adk.diff` before adding to manifest).

  - Public API (suggested):

```python
def run_wav2lip(
    face_video: Path | str,
    audio: Path | str,
    out_path: Path | str,
    *,
    opts: dict | None = None,
) -> dict:
    """Lip-sync `face_video` to `audio`, write result to `out_path`.

    Returns metadata dict with `artifact_uri`, `duration_s`, `model_checkpoint`, `face_detector`, `opts`, and `logs`.
    """
```

  - Options schema (common Wav2Lip flags mapped as `opts`):
    - `checkpoint_path: str` — path to Wav2Lip model checkpoint (required).
    - `face_det_checkpoint: str` — path to `s3fd.pth` face detector (optional; ensure a default/mirrored path exists).
    - `pads: tuple[int,int,int,int]` — bounding-box padding (up, down, left, right) to improve chin capture.
    - `resize_factor: int` — downscale factor for faster processing / better results on low-res faces.
    - `nosmooth: bool` — disable smoothing of face detections to avoid double-mouth artifacts.
    - `crop: Optional[tuple[int,int,int,int]]` — manual crop (x,y,w,h) if provided.
    - `fps: float | None` — output framerate override.
    - `gpu: int | None` — device id to run inference on; default to cuda:0 if available and allowed.
    - `verbose: bool` — include full ffmpeg/Wav2Lip logs in metadata when true.

  - Recommended implementation pieces
    1) Model & artifact preparation helper
       - Validate `checkpoint_path` and `face_det_checkpoint` exist; if not, provide a helper to download/mirror upstream weights into an approved `resources/` location (do NOT commit those weights; only provide URLs and a download helper).
       - Fail early and with a clear message if required weights are missing.

    2) API-first path: import & call upstream inference code
       - Prefer importing the upstream inference routines where possible (e.g., call functions from `inference.py` or expose a small wrapper `wav2lip.infer(face, audio, checkpoint, opts)` if the package is installed in the environment).
       - Use `torch` device selection (support CPU for dev & CUDA for gated smoke tests). Seed RNG where upstream supports it for deterministic tests.

    3) Subprocess fallback (pinned-commit strategy)
       - When the Python package is not available or in CI isolation, use a pinned clone of the upstream repo and run `python inference.py --checkpoint_path ... --face ... --audio ...` in a safe subprocess.
       - Use a safe `run_command(cmd, cwd, timeout_s, retries)` helper that:
         - Captures stdout/stderr to files / strings.
         - Kills process groups on timeout (use `preexec_fn=os.setsid` + `os.killpg`).
         - Validates exit codes and returns structured `SubprocessResult`.
         - Returns logs and the final output path.

    4) Pre- and post-processing
       - Provide helpers for face detection / cropping adjustments (wrap upstream face detection helpers or reuse `face_alignment`) and expose `pads` and `resize_factor` as first-class opts.
       - After inference, ensure audio is preserved or re-mixed with ffmpeg as needed (Wav2Lip inference typically preserves or re-attaches audio in its pipeline; validate behavior and fix if necessary using `ffmpeg` to copy/mix audio).

    5) Output handling & publishing
       - Write final MP4 to `out_path` and return/publish via `adk_helpers.publish_artifact()` in production flows.
       - Include metadata: `model_checkpoint`, `face_detector_path`, `device`, `duration_s`, `opts_snapshot`, `ffmpeg_version`, `stdout_log_uri`, `stderr_log_uri`.

  - Recommended CLI / invocation example (mirrors upstream usage):

```
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <audio.wav> --outfile <out.mp4> --pads 0 20 0 0 --resize_factor 1
```

  - Upstream tips (extracted from README)
    - Experiment with `--pads` (increase bottom padding to include chin) to fix dislocated mouth artifacts.
    - If double-mouth or artifacts appear, try `--nosmooth` to disable smoothing of face detections.
    - For higher-resolution videos, `--resize_factor` can improve visual quality or speed trade-offs.
    - Upstream provides multiple checkpoints (Wav2Lip, Wav2Lip+GAN). Choose the checkpoint appropriate for quality vs. artifact trade-offs.

  - Tests
    - Unit tests (fast, no heavy deps): mock the Wav2Lip inference call to return a deterministic short MP4 (or frames) and assert `run_wav2lip` metadata and publishing behavior.
    - Command-generation tests: pure functions that produce subprocess command lists for given `opts` should be tested to ensure correct flags and escaping.
    - Integration (gated): gated by `SMOKE_LIPSYNC=1` (or `SMOKE_ADK=1`) to run a short inference using a small checkpoint or a fixture checkpoint on supported hardware. CI must provide `ffmpeg` binary for this.
    - Failure path tests: simulate subprocess timeouts and non-zero exits; assert retries, logs capture, and cleanup behavior.

  - Manifest / dependency proposal
    - Upstream runtime deps: `torch`, `numpy`, `scipy`, `opencv-python`, `face-alignment` (or `face_alignment`), and any upstream pinned versions. `ffmpeg` binary required on system image.
    - Do NOT modify `pyproject.toml` directly — propose changes via `proposals/pyproject_adk.diff` and wait for explicit approval before adding heavy packages.

  - Security & licensing notes
    - The open-source Wav2Lip code and checkpoints are for research/non-commercial use per upstream README. Ensure project adheres to this restriction — production/commercial usage must contact authors.
    - Validate and sanitize any user-provided paths/URIs; do not allow arbitrary shell injection; map structured options to flags.

  - Estimate: 2–3 days for a robust adapter skeleton + unit tests; 3–5 days including gated integration smoke tests and CI image setup for `ffmpeg`.

  - Next steps (if you want me to implement):
    - A: Create a local adapter skeleton at `src/sparkle_motion/function_tools/lipsync_wav2lip/` with `run_wav2lip` and `run_command` helpers (local patch only). Requires no manifest edits.
    - B: Also add unit tests under `tests/unit/test_lipsync_wav2lip.py` that mock subprocess/pipeline outputs.
    - C: Prepare `proposals/pyproject_adk.diff` suggesting runtime deps for review (will not apply without your explicit approval phrase).

If you want me to produce the adapter skeleton and tests locally, choose A/B/C (you can pick multiple). I will not edit manifests or push changes without your explicit approval phrase.

### qa_qwen2vl
- Task: Implement `inspect_frames(frames, prompts) -> QAReport`
  - Adapter to Qwen-2-VL or ADK multimodal agent; produce structured `QAReport` artifact.
  - Integrate `request_human_input` on policy escalation and write memory timeline events.
  - Tests: unit tests for report shape; gated integration sampling for visual checks.
  - Estimate: 2–4 days

### assemble_ffmpeg
- Task: Implement deterministic assembly pipeline using `ffmpeg`
  - Provide helper `assemble_clips(movie_plan, clips, audio, out_path, opts)` that performs concat, overlay, transitions, and audio mixing with reproducible options.
  - Use a safe subprocess wrapper that validates exit codes, captures stdout/stderr, handles timeouts and retries, and records logs/metrics.
  - Tests: unit tests for command generation and error handling; end-to-end assembly unit test using short synthetic clips; gated integration smoke test that verifies final artifact integrity.
  - Estimate: 1–2 days for a robust skeleton and unit tests; 3–4 days with CI integration and gated smoke tests.

Details (design, API, and implementation guidance)

- Purpose: deterministically assemble clips, overlays, audio mixing, subtitles, and transitions into a final published artifact (MP4/MOV) while recording reproducible metadata and logs for debugging and auditing.

- Public API (suggested):

```python
def assemble_clips(movie_plan: dict, clips: list[dict], audio: Optional[dict], out_path: Path, opts: dict) -> ArtifactRef:
    """Assemble ordered clips and audio into final output.

    Returns an `ArtifactRef` (or dict) with `artifact_uri`, `media_type`, and metadata.
    """

def run_command(cmd: list[str], cwd: Path, timeout_s: int = 600, retries: int = 1) -> SubprocessResult:
    """Run a subprocess safely and return structured result (exit_code, stdout, stderr, duration).
    """
```

- Clip descriptor (example):

```json
{
  "uri": "/tmp/clip1.mp4",
  "start_s": 0.0,
  "end_s": 2.5,
  "transition": {"type":"fade", "duration":0.2},
  "z": 0
}
```

- Core implementation pieces:
  1) High-level assembler: translate `movie_plan`/`clips`/`audio` into one or more deterministic `ffmpeg` invocations. Prefer `filter_complex` for mixed-format operations (overlay, crossfade, audio mixing), and the `concat` demuxer for same-format, same-codec sequences.
  2) Safe subprocess wrapper: implement `run_command` that:
     - Uses `subprocess.Popen` with captured stdout/stderr redirected to temp files.
     - Enforces timeouts and kills process groups on timeout (use `preexec_fn=os.setsid` and `os.killpg`).
     - Retries transient failures (configurable retries, e.g., 1 retry).
     - Returns structured logs and duration.
  3) Staging & temp filesystem management: create a per-run temp dir (`tempfile.TemporaryDirectory()`), copy or hard-link inputs into it, and use deterministic file names.
  4) Cleanup & failure handling: on failure, collect ffmpeg logs as companion artifacts, optionally retain temp dir when `debug=True`, and ensure child processes are terminated.
  5) Publishing: call `adk_helpers.publish_artifact(path=out_path, media_type="video/mp4", metadata=meta)` after a successful run. Include assembly metadata: `plan_id`, `ffmpeg_version`, encoding args, runtime durations, and logs URI.
  6) Logging & telemetry: save full ffmpeg command, stdout/stderr, runtime, exit code, and peak resource hints (when available) as metadata and separate `.log` artifact.

- Determinism & reproducibility:
  - Pin encoding settings (codec, CRF, pixel format, framerate) in `opts` and record them in metadata.
  - Record the `ffmpeg` version (`ffmpeg -version`) and platform/OS info.
  - If any randomness is used (for stochastic transitions), accept a `seed` option and record it.

- Security & input validation:
  - Reject arbitrary shell passthrough. Only accept structured options and map them to safe ffmpeg flags.
  - Validate that input URIs exist and are within expected workspaces.
  - Sanitize metadata to avoid disclosing secrets.

- FFmpeg patterns & recommended args:
  - Concatenate same-format clips: use the concat demuxer with a deterministic list file when possible.
  - Use `filter_complex` for overlays, `xfade` for crossfades, and `acrossfade`/`amix` for audio transitions/mixing.
  - Deterministic encode args example:

```
-c:v libx264 -preset veryslow -crf 18 -pix_fmt yuv420p -movflags +faststart
-c:a aac -b:a 192k
```

- Subprocess wrapper features (API sketch):

```python
from dataclasses import dataclass

@dataclass
class SubprocessResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

def run_command(cmd: List[str], cwd: Path, timeout_s: int = 600, retries: int = 1) -> SubprocessResult:
    ...
```

- Testing strategy:
  - Unit tests:
    - Test command-generation only (pure functions) to assert correct filter strings for concat, overlay, and mix.
    - Mock `run_command` to simulate ffmpeg failures and assert retry and cleanup behavior.
  - Integration (gated):
    - Smoke test that uses small synthetic clips (generated by `ffmpeg` color source or programmatically created images and WAV) to produce a final MP4. Gate with `SMOKE_ADK=1` or `SMOKE_ASSEMBLE=1`. CI job must provide an `ffmpeg` binary.
  - Edge cases: mixed framerates (either normalize or fail), missing inputs, invalid ranges — fail with descriptive errors.

- Documentation & developer notes:
  - Update `docs/IMPLEMENTATION_TASKS.md` with the public API and example usage.
  - Add a `docs/assemble_ffmpeg.md` with example `ffmpeg` filter snippets, typical `movie_plan` shapes, and CI setup notes.

- Manifest / dependency considerations:
  - System: `ffmpeg` binary required (install via apt/yum/homebrew or provide a Docker image like `jrottenberg/ffmpeg` in CI).
  - Optional Python packages: `imageio-ffmpeg` or `ffmpeg-python` (prefer direct subprocess for explicit control). If added, prepare `proposals/pyproject_adk.diff` per repo policy.

- CI / environment:
  - Ensure `ffmpeg` is available in CI images for the gated smoke test, or run the smoke test in a container with `ffmpeg` installed.
  - Provide a lightweight fixture mode for tests to avoid heavy binaries when not gating (unit tests should not need `ffmpeg`).

- Security / policy notes:
  - The assembler must not execute arbitrary user-supplied shell commands. Validate and whitelist options.
  - Production callers (e.g., `production_agent`) must run content policy checks before assembly.

If you want, I can implement the skeleton and safe subprocess helper as a local patch (showing diffs), create the unit tests, and add example docs. I will not modify `pyproject.toml` or push changes without your explicit approval phrase.

------------------------------------------------------------

### production_agent (expanded)

- **Purpose**: execute `MoviePlan` objects end-to-end; orchestrate `images_agent`, `tts_agent`, `videos_agent`, and `assemble_ffmpeg`; enforce policy and gating; publish final artifacts.

- **Public API**: `execute_plan(plan: MoviePlan, *, mode: Literal["dry", "run"] = "dry") -> list[ArtifactRef]`
  - `dry`: validate and simulate orchestration without invoking heavy FunctionTools.
  - `run`: perform guarded execution (checks, gated services, publish artifacts).

- **Responsibilities & behavior**:
  - Validate `MoviePlan` schema and preconditions before execution.
  - Run content & safety policy checks early; on rejection raise `PolicyViolationError` with clear reason and memory event.
  - Apply gating: only call heavy FunctionTools when the appropriate environment flags are set (e.g., `SMOKE_ADK`, `SMOKE_LIPSYNC`) and when `adk_helpers.require_adk()` succeeds for ADK-backed providers.
  - Orchestrate per-step execution with retries/backoff and bounded concurrency for expensive steps (e.g., chunked video renders).
  - Track and publish intermediate artifacts (images, wavs, clips) and final assembled artifact via `adk_helpers.publish_artifact()`; include rich metadata (plan_id, step_id, model_id, device, seed, durations).
  - Provide observability: for each step record timing, peak-memory hints, stdout/stderr logs for subprocess-backed tools and callback progress events for pipeline-backed tools.
  - Provide a `simulate_execution_report(plan, policy_decisions)` output for `dry` runs that lists expected invocation graph, resource estimates, and publish URIs that would be created.

- **Error handling & cleanup**:
  - Implement bounded retries for transient errors with exponential backoff and jitter.
  - Fail-fast on deterministic policy violations; for runtime OOM or hardware errors attempt controlled fallback (reduce chunk size, switch to CPU-limited path) where available and log memory telemetry.
  - Ensure temporary files and GPU contexts are released on both success and failure (call `gpu_utils.model_context` cleanup, `torch.cuda.empty_cache()`, `gc.collect()`).

- **Example orchestration pseudocode**:

```python
def execute_plan(plan, mode="dry"):
    validate_plan_schema(plan)
    policy_decisions = run_policy_checks(plan)
    if policy_decisions.reject:
        raise PolicyViolationError(policy_decisions.reason)
    if mode == "dry":
        return simulate_execution_report(plan, policy_decisions)

    artifacts = []
    for step in plan.steps:
        if step.type == "image":
            art = images_agent.render(step.prompt, opts=step.opts)
        elif step.type == "tts":
            art = tts_agent.synthesize(step.text, voice_config=step.voice)
        elif step.type == "video":
            art = videos_agent.render_video(step.start_frames, step.end_frames, step.prompt, opts=step.opts)
        elif step.type == "lipsync":
            art = lipsync_wav2lip.run_wav2lip(step.face_video, step.audio, step.out_path, opts=step.opts)
        else:
            art = handle_custom_step(step)
        artifacts.append(art)

    final_artifact = assemble_ffmpeg(plan, artifacts, opts=plan.assemble_opts)
    publish_uri = adk_helpers.publish_artifact(path=final_artifact.path, media_type="video/mp4", metadata={"plan_id": plan.id})
    return artifacts + [publish_uri]
```

- **Tests**:
  - Unit tests: mock FunctionTools and assert orchestration, policy enforcement, and retry behavior.
  - Integration/gated: full end-to-end smoke test run behind `SMOKE_ADK=1` that exercises a tiny plan, verifies published artifacts and metadata.

- **Estimate**: 2–4 days to implement a robust `production_agent` scaffold and unit tests.

**Orchestration policy & operational knobs**

- Decision matrix (gating): which environment flags gate which FunctionTools
  - `SMOKE_ADK=1`: enables ADK-backed FunctionTools and real artifact publishing. When not set, the `production_agent` must only call local stubs or simulate artifact creation.
  - `SMOKE_LIPSYNC=1`: allows `lipsync_wav2lip` to run real inference; when unset, lipsync steps should be simulated or fail fast with a clear message.
  - `SMOKE_TTS=1`: allows `tts_chatterbox` real calls; otherwise use a fixture TTS stub.
  - `SMOKE_QA=1`: enables real `qa_qwen2vl` checks; when unset, run a lightweight heuristic QA or mark QA steps as `simulated`
  - `SMOKE_ADAPTERS=1`: general flag the team can use to gate any heavy adapter outside the specific flags above (optional override)

- Default concurrency & chunking strategy
  - `max_concurrent_image_renders`: 4 (default) — cap concurrent calls to `images_sdxl` to avoid host overload.
  - `max_concurrent_video_chunk_renders`: 2 (default) — cap parallel chunk renders per host; for high‑capacity hosts this can be increased via runtime config.
  - `video_chunk_length_frames`: default 64 frames per chunk (tunable per plan/host). Use overlap of 2–4 frames between chunks for continuity.
  - `video_chunk_overlap_frames`: 4 — how many frames overlap between adjacent chunks to enable smooth crossfade/temporal continuity.
  - `step_queue_backlog_limit`: 100 — reject plans that would enqueue more than this many pending steps to avoid unbounded memory growth.

- Retry / backoff policy (recommended default)
  - Retry classification:
    - Transient errors: network timeouts, temporary OOMs (recoverable by retrying with adjustments), intermittent API errors → retryable.
    - Fatal errors: policy violations, invalid inputs, unsupported step types → non-retryable.
  - Default retry policy:
    - `max_attempts`: 3 (initial attempt + 2 retries)
    - `backoff_base_seconds`: 1.0
    - `backoff_multiplier`: 2.0
    - `jitter`: 0.2 (fractional jitter added to backoff to avoid thundering herd)
    - Example: attempt 1, wait 1s +/- jitter; attempt 2, wait 2s +/- jitter; attempt 3, wait 4s +/- jitter.
  - Special handling: on OOM during a video chunk render, attempt a single adaptive retry that reduces chunk size by 50% before falling back to CPU/failed state.

- `dry` vs `run` semantics (behavior contract)
  - `mode="dry"`:
    - Validate plan schema and policy checks fully.
    - Simulate resource usage estimates for each step (GPU hours, memory, estimated runtime) and return a `simulate_execution_report` (see below) instead of invoking heavy adapters.
    - Emit simulated artifact URIs (e.g., `file:///simulated/{plan_id}/{step_id}.png`) but do not call `adk_helpers.publish_artifact()`.
    - Allowed side-effects: writing a small dry-report JSON to a local temp dir and logging; disallowed: network calls to ADK or model loads.
  - `mode="run"`:
    - Full guarded execution: require gates (flags + `adk_helpers.require_adk()` where applicable).
    - Invoke adapters under `gpu_utils.model_context` with real publishes via `adk_helpers.publish_artifact()`.
    - Ensure retries, telemetry, and cleanup happen as defined.

**Simulate execution report (dry) shape**

- `simulate_execution_report` (object)
  - `plan_id: str`
  - `steps: list[{step_id, step_type, estimated_runtime_s, estimated_gpu_memory_mb, simulated_artifact_uri}]`
  - `resource_summary: {total_estimated_gpu_hours, total_estimated_runtime_s}`
  - `policy_decisions: list` — any policy flags or warnings

**Observable telemetry contract (per-step metadata)**

- Each step execution must emit or return a `StepExecutionRecord` (structured metadata) with fields:
  - `plan_id: str`
  - `step_id: str`
  - `step_type: str`
  - `status: Literal["queued","running","succeeded","failed","skipped","simulated"]`
  - `start_time: str` (ISO 8601)
  - `end_time: str` (ISO 8601)
  - `duration_s: float`
  - `model_id: Optional[str]` — model identifier used (if applicable)
  - `device: Optional[str]` — e.g., `cuda:0`, `cpu`
  - `memory_hint_mb: Optional[int]` — peak or estimated memory usage
  - `attempts: int` — number of attempts taken
  - `logs_uri: Optional[str]` — path/URI to stdout/stderr logs or adapter logs
  - `artifact_uri: Optional[str]` — published artifact (if any)
  - `error_type: Optional[str]` — one of the `Plan*Error` codes if failed
  - `meta: Optional[dict]` — adapter-specific metadata (seed, sampler, fps, etc.)

Emit these records to logger/metrics sink and include them in any artifact metadata published.

**Failure handling / cleanup**

- On failure of a step:
  - Record `StepExecutionRecord` with `status="failed"` and `error_type` populated.
  - If transient and attempts remain, schedule retry with backoff. If retries exhausted, mark dependent steps as `skipped` and include explanation in final artifact metadata.
  - Always run cleanup hooks: release GPU contexts, call `torch.cuda.empty_cache()`, `gc.collect()`, and remove or archive temp files per `debug` flag.

**Sample unit & integration tests (suggested)**

- Unit tests (fast, mocked adapters):
  1. `test_execute_plan_dry_simulation`: feed a simple plan into `execute_plan(mode='dry')` and assert `simulate_execution_report` shape and simulated URIs.
  2. `test_retry_on_transient_error`: mock an adapter to raise a transient exception on first call and succeed on the second; assert `attempts == 2` and final `status == 'succeeded'`.
  3. `test_policy_rejection`: create a plan with disallowed content; assert `PlanPolicyViolation` is raised and no adapters are called.
  4. `test_oom_adaptive_retry`: mock a video chunk renderer to raise an OOM on first attempt; assert that `production_agent` reduces chunk size and retries once before failing or succeeding per mock.

- Integration/gated smoke tests (requires flags):
  1. `smoke_execute_plan_end_to_end` (gated by `SMOKE_ADK=1`): run a tiny plan (1 image + 1 short TTS) against fixture/backed adapters and assert artifacts are published and `StepExecutionRecord` entries are produced.
  2. `smoke_lipsync_flag_gate` (gated by `SMOKE_LIPSYNC=1`): verify lipsync steps run only when the flag is set; otherwise they are simulated and reported as `skipped`/`simulated`.

**Developer notes**

- Configuration: surface these knobs via a config object (env vars, config file or `production_agent` init args) so teams can tune per-deployment.
- Observability: integrate `StepExecutionRecord` emission with existing telemetry/logging (e.g., `adk_helpers.write_memory_event()` or metrics export) so CI and operator dashboards can aggregate run results.
- Backwards compatibility: if older plans lack `assemble_opts` or `step.id`, the agent should normalize them (generate `id` if missing) and record a warning rather than failing a `dry` run.


### qa_qwen2vl (expanded)

- **Purpose**: frame-level QA and inspection adapter. Run multimodal QA using Qwen2‑VL (or another ADK multimodal provider) over frames or image+text prompts and produce a structured `QAReport` artifact used by agents and human reviewers.

- **Model & resources**:
  - Canonical model id: `Qwen/Qwen2-VL-7B-Instruct` (Hugging Face).
  - Upstream recommends using `Qwen2VLForConditionalGeneration`, `AutoProcessor`, and the `qwen_vl_utils` helper toolkit for vision pre-processing and packaging multi-image/video inputs.
  - Note: building `transformers` from source is recommended in some environments to avoid `KeyError: 'qwen2_vl'` (e.g., `pip install git+https://github.com/huggingface/transformers`).

- **Public API**: `inspect_frames(frames: Iterable[Path|str], prompts: list[str], *, opts: dict = None) -> QAReport`
  - `QAReport` fields (recommended): `per_frame: list[{frame_index, scores, flags, annotations}]`, `global_flags`, `top_responses`, `confidence_summary`, `artifact_uri`, `logs_uri`, `opts_snapshot`.

- **Quickstart / Usage (canonical snippet)**

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load model (device_map + dtype recommended for your infra)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

# Processor: handles chat template and input packaging (images/videos)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Example messages: interleaved image + text content per the model chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/frame1.jpg"},
            {"type": "text", "text": "Any safety issues in this frame?"},
        ],
    }
]

# Prepare text and vision inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to("cuda")

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)

# Trim prompt tokens and decode
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
```

- **Input formats supported**
  - Images: local file paths (`file:///`), HTTP/HTTPS URLs, and base64 strings.
  - Videos: local file paths only (video inference currently accepts local files).

- **Performance & inference tips (from upstream)**
  - Use `min_pixels` / `max_pixels` in `AutoProcessor.from_pretrained(..., min_pixels=..., max_pixels=...)` to control the dynamic visual tokenization range and balance memory vs. fidelity (example tokens range: 256–1280 tokens mapped to pixels via 28*28 factor).
  - Alternatively set `resized_height` and `resized_width` for exact control (values are rounded to multiples of 28).
  - Enable accelerated attention implementations when available (e.g., `attn_implementation="flash_attention_2"`) and prefer `torch_dtype=torch.bfloat16` / `torch_dtype="auto"` for BF16-capable hardware.
  - Use `device_map="auto"` for multi‑GPU hosts and `torch_dtype="auto"` to let HF pick BF16/FP16 where supported.
  - Upstream provides `qwen-vl-utils` (`pip install qwen-vl-utils`) for `process_vision_info` helpers that convert messages into `images`/`videos` tensors accepted by the processor.

- **Recommended behavior for the adapter**
  - Provide a small wrapper that accepts a list of frames and a list of QA prompts and maps them to the `messages` structure used by the processor.
  - Batch frames into as few model calls as possible, respecting input token/visual token limits.
  - Expose `opts` for `min_pixels`, `max_pixels`, `resized_height`, `resized_width`, `max_new_tokens`, `attn_implementation`, `torch_dtype`, and `device_map` so callers can tune resource vs. quality.
  - Provide fallbacks: a lightweight stub for unit tests (no heavy deps) and a gated real-call path controlled by `SMOKE_ADK=1` (or similar).

- **Policy & escalation**
  - Run automated policy checks (NSFW, copyright, identity) by templating prompts for model-based detectors (e.g., "Does this image contain adult content?") and mapping response confidences to actions.
  - If confidence is below the configured threshold, call `request_human_input()` and attach the `QAReport` artifact to the memory timeline using `adk_helpers.write_memory_event()`.

- **Tests**
  - Unit tests: stub `Qwen2VLForConditionalGeneration` / processor and assert `inspect_frames` returns correct `QAReport` shape and threshold logic. Use deterministic stubs for generated responses.
  - Command/packaging tests: ensure `process_vision_info` and `processor.apply_chat_template` are called with expected messages when given frames+prompts.
  - Integration/gated: run a small sample through the real model with `SMOKE_ADK=1` (requires `qwen-vl-utils`, `torch` with BF16 support or CPU fallback, and local `ffmpeg` if doing video input handling).

- **Limitations & notes (from upstream)**
  - Qwen2‑VL does not process audio embedded in videos — only visual frames and associated text.
  - Data is current up to the model's training cutoff; validate factual claims accordingly.
  - Some complex spatial reasoning, accurate counting, or person/IP identification may be limited — configure QA thresholds and human review conservatively.

- **Manifest & dependency proposal guidance**
  - Suggested runtime deps for proposals: `transformers` (recommend build-from-source in some CI images), `qwen-vl-utils`, `torch` (BF16/FP16 support per infra), `safetensors` (if using safetensors weights), and `accelerate` if using offload/device_map strategies.
  - Do NOT modify `pyproject.toml` directly — prepare `proposals/pyproject_adk.diff` and wait for explicit approval before adding heavy packages.

- **Estimate**: 2–4 days to implement adapter + unit tests and gated integration smoke tests.

## Cross-cutting tasks
- `gpu_utils.model_context` — implement consistent context manager for model load/unload and CUDA cleanup (must be used by all heavy tools).
- `adk_helpers.require_adk()` vs `adk_helpers.probe_sdk(non_fatal=True)` — audit callers and apply non-fatal probe in scripts and fail-fast in entrypoints.
- Add gated smoke tests (`tests/smoke/<tool>_adk_integration.py`) that run only when `SMOKE_ADK=1` is set.
- Document exact CUDA/toolkit choices for `torch` in a final `proposals/pyproject_adk.diff` (e.g., cu118 vs cu120) before applying manifest edits.

------------------------------------------------------------


If you want these created as issues in the repo, I can open PRs/Issues for each task (requires your confirmation to push branches). For now these are a documentation-level TODO reference.
