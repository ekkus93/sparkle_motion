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

## assemble_ffmpeg
- Task: Implement deterministic assembly pipeline using `ffmpeg`
  - Provide helper `assemble_clips(movie_plan, clips, audio)` that performs concat, overlay, and audio mixing with reproducible options.
  - Use a safe subprocess wrapper that validates exit codes and captures logs/metrics.
  - Tests: end-to-end assembly unit test (short synthetic clips), artifact integrity checks.
  - Estimate: 1–2 days

## qa_qwen2vl
- Task: Implement `inspect_frames(frames, prompts) -> QAReport`
  - Adapter to Qwen-2-VL or ADK multimodal agent; produce structured `QAReport` artifact.
  - Integrate `request_human_input` on policy escalation and write memory timeline events.
  - Tests: unit tests for report shape; gated integration sampling for visual checks.
  - Estimate: 2–4 days

------------------------------------------------------------

## Cross-cutting tasks
- `gpu_utils.model_context` — implement consistent context manager for model load/unload and CUDA cleanup (must be used by all heavy tools).
- `adk_helpers.require_adk()` vs `adk_helpers.probe_sdk(non_fatal=True)` — audit callers and apply non-fatal probe in scripts and fail-fast in entrypoints.
- Add gated smoke tests (`tests/smoke/<tool>_adk_integration.py`) that run only when `SMOKE_ADK=1` is set.
- Document exact CUDA/toolkit choices for `torch` in a final `proposals/pyproject_adk.diff` (e.g., cu118 vs cu120) before applying manifest edits.

------------------------------------------------------------

If you want these created as issues in the repo, I can open PRs/Issues for each task (requires your confirmation to push branches). For now these are a documentation-level TODO reference.
