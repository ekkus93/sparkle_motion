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
  - Validate output against `MoviePlan` schema (use `schema_registry` artifact URIs).
  - Publish plan artifact via `adk_helpers.publish_artifact()` and write a memory event.
  - Tests: unit test for schema conformance; gated integration smoke that exercises a small LLM (fixture-mode or real SDK).
  - Estimate: 2–3 days (including prompts+validation)

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
- Task: Implement `run_wav2lip(video_path, audio_path, out_path)`
  - Prefer Python API; if not available, use a subprocess wrapper with a pinned Wav2Lip repo commit.
  - Ensure ffmpeg/ OpenCV availability and robust temp-file handling.
  - Tests: unit tests with short fixture clips; gated integration smoke.
  - Estimate: 2–3 days

### qa_qwen2vl
- Task: Implement `inspect_frames(frames, prompts) -> QAReport`
  - Adapter to Qwen-2-VL or ADK multimodal agent; produce structured `QAReport` artifact.
  - Integrate `request_human_input` on policy escalation and write memory timeline events.
  - Tests: unit tests for report shape; gated integration sampling for visual checks.
  - Estimate: 2–4 days

### assemble_ffmpeg
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
