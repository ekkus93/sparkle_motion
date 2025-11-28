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
  - Tests: unit tests for decision logic and metadata/format validation; integration smoke tests gated by `SMOKE_TTS=1` that exercise a small real or fixture TTS provider.
  - Estimate: 2–4 days
  - Notes: runtime dependency or manifest changes must follow the `proposals/pyproject_adk.diff` process and require explicit approval before editing `pyproject.toml` or pushing runtime changes.

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

### videos_agent
- Task: Implement `videos_agent` decision layer (caller for `videos_wan`/video adapters)
  - Public API: `render_video(start_frames: Iterable[Frame], end_frames: Iterable[Frame], prompt: str, opts: dict) -> ArtifactRef`.
  - Responsibilities: select video provider (WanAdapter vs queued offline renderer), manage chunking/segmentation for VRAM safety, orchestrate retries/backoff, and enforce video-specific policy checks (copyright, prohibited content).
  - Integration: coordinate with `WanAdapter`/`videos_wan` FunctionTool for heavy inference inside `gpu_utils.model_context` and publish assembled artifacts via `adk_helpers.publish_artifact()`.
  - Tests: unit tests for orchestration logic and chunking; gated integration smoke when `SMOKE_ADK=1`.
  - Estimate: 2–4 days

## FunctionTool / Adapter Tasks

These tasks cover compute-bound adapters (FunctionTools) that perform heavy model loading, inference, and artifact publishing. Adapters must be written to run safely inside `gpu_utils.model_context` and to publish deterministic artifacts.

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
# Implementation Tasks — Per-Tool (issue-style)

This document contains issue-style, actionable TODOs for converting FunctionTool
scaffolds into production-capable, ADK-integrated implementations. These are
meant to be reference tasks for engineering sprints; each item should be
implemented behind feature branches and reviewed before changing manifests.

Guidelines:
- All runtime dependency changes must be proposed via `proposals/pyproject_adk.diff` and approved before editing `pyproject.toml`.
- Integration tests requiring real ADK credentials or heavy weights are gated by `SMOKE_ADK=1`.
- Entrypoints that instantiate agents must `require_adk()` and fail loudly if SDK or credentials are missing.

------------------------------------------------------------

## script_agent
- Task: Implement `generate_plan(prompt: str) -> MoviePlan`
  - Use `adk_factory.get_agent('script_agent', model_spec)` to construct the LlmAgent.
  - Validate output against `MoviePlan` schema (use `schema_registry` artifact URIs).
  - Publish plan artifact via `adk_helpers.publish_artifact()` and write a memory event.
  - Tests: unit test for schema conformance; gated integration smoke that exercises a small LLM (fixture-mode or real SDK).
  - Estimate: 2–3 days (including prompts+validation)

### production_agent
- Task: Implement `production_agent` to execute `MoviePlan` objects and orchestrate media generation.
  - Public API: `execute_plan(plan: MoviePlan, *, mode: Literal["dry","run"] = "dry") -> list[ArtifactRef]` — `dry` validates and simulates orchestration without invoking heavy FunctionTools; `run` performs gated execution.
  - Responsibilities: enforce content & safety policy, select providers/gates (e.g., `SMOKE_ADK`), manage orchestration and retries, call per-media agents (`images_agent`, `tts_agent`, `videos_agent`) and assembly helpers (`assemble_ffmpeg`), and publish final artifacts via `adk_helpers.publish_artifact()`.
  - Tests: unit tests for orchestration logic (mocked FunctionTools), policy enforcement tests, and gated integration smoke tests that run full end-to-end execution when `SMOKE_ADK=1`.
  - Estimate: 2–4 days (including gating, retries, and publish logic)

#### Example `execute_plan` flow (pseudocode)

```python
# Example pseudocode for production_agent.execute_plan
def execute_plan(plan, mode="dry"):
  # Validate the plan schema and preconditions
  validate_plan_schema(plan)
  simulate_only = (mode == "dry")

  # Optionally run lightweight policy checks without invoking heavy tools
  policy_decisions = run_policy_checks(plan)
  if policy_decisions.reject:
    raise PolicyViolationError(policy_decisions.reason)

  if simulate_only:
    return simulate_execution_report(plan, policy_decisions)

  artifacts = []
  for step in plan.steps:
    if step.type == "image":
      art = images_agent.render(step.prompt, opts=step.opts)
    elif step.type == "tts":
      art = tts_agent.synthesize(step.text, voice_config=step.voice)
    elif step.type == "video":
      art = videos_agent.render_video(step.start_frames, step.end_frames, step.prompt, opts=step.opts)
    else:
      art = handle_custom_step(step)
    artifacts.append(art)

  final_artifact = assemble_ffmpeg(plan, artifacts, opts=plan.assemble_opts)
  publish_uri = adk_helpers.publish_artifact(path=final_artifact.path, media_type="video/mp4", metadata={"plan_id": plan.id})
  return artifacts + [publish_uri]
```


## images_sdxl
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

If you'd like, I can create a patch that adds the `DiffusersAdapter` skeleton under `src/` and the minimal unit tests (stubs) — tell me if you want me to prepare that patch next.

## videos_wan (pilot)
- Task: Implement `run_wan_inference(start_frames, end_frames, prompt) -> mp4`
  - Pilot first: highest VRAM and driver risk. Implement `WanAdapter` and `gpu_utils.model_context('wan2.1')`.
  - Implement deterministic output checks, codec validation, and chunked rendering to limit VRAM.
  - Add explicit load/unload, CUDA context release, and robust error handling with retries/backoff.
  - Tests: heavy integration tests gated by `SMOKE_ADK=1` and run only on approved hosts; unit tests stub the adapter.
  - Estimate: 1–2 weeks (research + validation on A100 required)

## tts_agent
- Task: Implement `tts_agent` decision layer and `tts_chatterbox` FunctionTool adapter
  - `tts_agent` (decision layer): responsible for provider selection, policy checks, retries/backoff, rate limiting, and orchestrating FunctionTool invocations. Public API: `synthesize(text: str, voice_config: dict) -> ArtifactRef` (returns a WAV artifact reference and metadata).
  - `tts_chatterbox` (FunctionTool adapter): compute-bound adapter that performs heavy TTS work (model load, synthesis, wav export). Implement as an ADK FunctionTool so the agent can call it; keep model lifecycle inside a guarded context manager to ensure safe load/unload.
  - Fallbacks: prefer ADK-managed TTS providers when available; fall back to local Coqui TTS for developer workflows. Entrypoints that instantiate agents must call `adk_helpers.require_adk()` or use `adk_helpers.probe_sdk(non_fatal=True)` where appropriate.
  - Metadata: publish duration, sample_rate, voice_id, format, seed (if applicable), and runtime info (model id, inference device, synth time). Publish WAV artifacts via `adk_helpers.publish_artifact()`.
  - Tests: unit tests for decision logic and metadata/format validation; integration smoke tests gated by `SMOKE_TTS=1` that exercise a small real or fixture TTS provider.
  - Estimate: 2–4 days
  - Notes: runtime dependency or manifest changes must follow the `proposals/pyproject_adk.diff` process and require explicit approval before editing `pyproject.toml` or pushing runtime changes.

### Chatterbox upstream (Resemble AI)

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

- Safety & policy:
  - The upstream repo includes a disclaimer—do not use for harmful content. Ensure the agent decision layer (`tts_agent`) enforces policy checks (content moderation) and logs policy events.

Where to look upstream for implementation details and examples:

- `example_tts.py` and `example_vc.py` in the upstream repo — copy or adapt invocation patterns and CLI args.
- `gradio_tts_app.py` — demonstrates server-style usage and runtime flags.
- `pyproject.toml` — see pinned dependency versions and Python requirement (3.11).

Recommendation: Use the upstream GitHub repo (`https://github.com/resemble-ai/chatterbox`) as the authoritative source for adapter wiring, and mirror these quickstart snippets into the `docs/IMPLEMENTATION_TASKS.md` as reference usage for implementers.

## lipsync_wav2lip
- Task: Implement `run_wav2lip(video_path, audio_path, out_path)`
  - Prefer Python API; if not available, use a subprocess wrapper with a pinned Wav2Lip repo commit.
  - Ensure ffmpeg/ OpenCV availability and robust temp-file handling.
  - Tests: unit tests with short fixture clips; gated integration smoke.
  - Estimate: 2–3 days



## qa_qwen2vl
- Task: Implement `inspect_frames(frames, prompts) -> QAReport`
  - Adapter to Qwen-2-VL or ADK multimodal agent; produce structured `QAReport` artifact.
  - Integrate `request_human_input` on policy escalation and write memory timeline events.
  - Tests: unit tests for report shape; gated integration sampling for visual checks.
  - Estimate: 2–4 days

## assemble_ffmpeg
- Task: Implement deterministic assembly pipeline using `ffmpeg`
  - Provide helper `assemble_clips(movie_plan, clips, audio)` that performs concat, overlay, and audio mixing with reproducible options.
  - Use a safe subprocess wrapper that validates exit codes and captures logs/metrics.
  - Tests: end-to-end assembly unit test (short synthetic clips), artifact integrity checks.
  - Estimate: 1–2 days

------------------------------------------------------------

## Cross-cutting tasks
- `gpu_utils.model_context` — implement consistent context manager for model load/unload and CUDA cleanup (must be used by all heavy tools).
- `adk_helpers.require_adk()` vs `adk_helpers.probe_sdk(non_fatal=True)` — audit callers and apply non-fatal probe in scripts and fail-fast in entrypoints.
- Add gated smoke tests (`tests/smoke/<tool>_adk_integration.py`) that run only when `SMOKE_ADK=1` is set.
- Document exact CUDA/toolkit choices for `torch` in a final `proposals/pyproject_adk.diff` (e.g., cu118 vs cu120) before applying manifest edits.

------------------------------------------------------------

If you want these created as issues in the repo, I can open PRs/Issues for each task (requires your confirmation to push branches). For now these are a documentation-level TODO reference.
