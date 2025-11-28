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

## videos_wan (pilot)
- Task: Implement `run_wan_inference(start_frames, end_frames, prompt) -> mp4`
  - Pilot first: highest VRAM and driver risk. Implement `WanAdapter` and `gpu_utils.model_context('wan2.1')`.
  - Implement deterministic output checks, codec validation, and chunked rendering to limit VRAM.
  - Add explicit load/unload, CUDA context release, and robust error handling with retries/backoff.
  - Tests: heavy integration tests gated by `SMOKE_ADK=1` and run only on approved hosts; unit tests stub the adapter.
  - Estimate: 1–2 weeks (research + validation on A100 required)

## tts_agent
- Task: Implement `synthesize_speech(text, voice_config) -> wav` via a `TtsAgent` that selects a backend TTS implementation (Chatterbox function tool or Coqui function tool) at runtime.
  - The `TtsAgent` should prefer ADK-exposed Chatterbox when available and fall back to a local Coqui TTS function tool in dev or when ADK is unavailable.
  - Produce metadata (duration_seconds, sample_rate, voice_id, engine) and publish WAV artifact via ADK when available.
  - Tests: unit tests for format/metadata; gated TTS smoke when `SMOKE_ADK=1` (real Chatterbox on GPU).
  - Estimate (planning-only): minimal dev wiring: a few hours; production-ready agent (ADK publishing, retries, gating, tests): 1–2 days.

  **Implementation checklist (doc-level wiring — add before coding)**
  - Purpose: list the exact items, schemas, wiring notes and files that must exist or be created when the `tts_agent` and Chatterbox FunctionTool are implemented. This is intentionally prescriptive so implementers can follow it without guessing.
  - Voice config schema: create `schemas/voice_config.schema.json` (minimal, extendable). Fields to include at minimum: `voice` (string), `language` (BCP-47), `sample_rate` (integer), `format` (enum: `wav|pcm16|flac|mp3`), `speed` (number), `pitch_shift` (number), `speaker_id` (string|int), `audio_prompt_path` (string, optional). Keep `required` empty for now for max compatibility.
  - Artifact metadata schema: create `schemas/artifact_metadata.schema.json` with fields: `artifact_type`, `created_at` (RFC3339), `generated_by.tool_id`, `generated_by.invocation_id`, `model.name`, `model.version` (optional), `backend.name`, `backend.device`, `audio.sample_rate`, `audio.channels`, `audio.duration_ms`, `checksum_sha256`, `watermarked` (bool), `provenance` (free-form).
  - FunctionTool registration metadata (Chatterbox): prepare `configs/tool_chatterbox.json` containing:
    - `tool_id`: `com.sparkle_motion.tts.chatterbox`
    - `name`: `tts_chatterbox.generate`
    - `description`: short description
    - `input_schema`: object with `text` (string, required), `voice_config` (`$ref` to `schemas/voice_config.schema.json`), `request_id` (optional)
    - `output_schema`: object with `artifact_uri`, `sample_rate`, `duration_ms`, `checksum_sha256`, `artifact_metadata` (ref to artifact schema)
  - Gating / env: canonical gate = `SMOKE_TTS=1`. Backward-compat behavior: if `SMOKE_TTS` unset and `SMOKE_ADK=1`, treat TTS smoke as enabled. Document the exact evaluation expression in the adapter README.
  - Retry/backoff defaults: attempts=3, initial_delay_s=0.5, backoff_factor=2.0, max_delay_s=10.0, jitter_s=0–0.25. Document which exceptions are retryable (transient CUDA OOM with retry flag, 5xx RPCs, connection resets).
  - Code layout (suggested file paths):
    - `src/sparkle_motion/adapters/tts_agent.py` — decision-layer agent with `synthesize(text, voice_config)` returning `(wav_path, metadata)`.
    - `src/sparkle_motion/adapters/backends/chatterbox_backend.py` — Chatterbox adapter; use guarded runtime imports for `chatterbox`/`torchaudio` and `probe_sdk(non_fatal=True)` if ADK publishing is used.
    - `src/sparkle_motion/adapters/backends/coqui_backend.py` — local fallback adapter for dev/smoke tests.
    - `tests/unit/test_tts_agent.py` — unit tests (mocks); `tests/smoke/test_tts_chatterbox_adk.py` — gated integration smoke (runs when `SMOKE_TTS=1` or `SMOKE_ADK=1`).
  - ADK publishing: adapters MUST call `probe_sdk(non_fatal=True)` and only call `adk_helpers.publish_artifact()` when ADK is present; otherwise return a local `file://` path plus artifact metadata.
  - Colab / torch note (docs-only): Google Colab typically ships a compatible `torch`/`torchaudio` + CUDA runtime (A100/GPU) — **do not** add an automatic install step. Document an optional manual install snippet in the docs for operators who need to re-install or replicate the runtime, but do not attempt to install during adapter startup. Example note:
    - "Note: Colab A100 runtimes generally include a suitable PyTorch+CUDA wheel. Only install a pinned wheel manually when reproducing locally or if a runtime reports missing/corrupt binaries. Example manual (do not run automatically): `pip install \"torch==2.2.2+cu126\" \"torchaudio==2.2.2+cu126\" --index-url https://download.pytorch.org/whl/cu126`"
  - Manifest guidance (docs-only): instead of committing dependency pins now, document that `chatterbox-tts` and `torchaudio` should be added to an optional `tts` extra in `pyproject.toml` and that `torch` wheel must be pinned per-host (Colab: prefer the runtime-provided wheel; CI: pin `cu126` if the runner matches). Add a short placeholder line referencing `proposals/pyproject_adk.diff` as the place to propose exact pins before any manifest edits.
  - Tests & QA: include unit tests that mock `ChatterboxTTS` and assert saved WAV properties (sample rate from `model.sr`, duration, checksum). Local smoke: use Coqui for fast synth. Gated GPU smoke: `SMOKE_TTS=1`.
  - Files to create once authorized (explicit list so reviewers know what to expect):
    - `schemas/voice_config.schema.json`
    - `schemas/artifact_metadata.schema.json`
    - `configs/tool_chatterbox.json`
    - `src/sparkle_motion/adapters/tts_agent.py`
    - `src/sparkle_motion/adapters/backends/chatterbox_backend.py`
    - `src/sparkle_motion/adapters/backends/coqui_backend.py`
    - `tests/unit/test_tts_agent.py`
    - `tests/smoke/test_tts_chatterbox_adk.py`
  - Acceptance criteria for initial PR (for reviewers):
    - Unit tests for `TtsAgent` pass locally without heavy dependencies (use mocks/fakes).
    - Documentation update in `docs/IMPLEMENTATION_TASKS.md` listing the created files and gating instructions.
    - A `README` or docstring in `chatterbox_backend.py` describing ADK probe usage, watermark flagging, and required model runtime properties (`model.sr`, `model.generate`).
    - A prepared `configs/tool_chatterbox.json` with input/output schemas (self-contained or with valid local `$ref`).


  Research & implementation notes (Chatterbox-specific)
  - Repo / Installation:
    - Public repo: https://github.com/resemble-ai/chatterbox
    - PyPI package: `pip install chatterbox-tts` (or install from source for dev: `git clone ... && pip install -e .`).
    - Chatterbox was developed/tested on Python 3.11; dependencies pinned in `pyproject.toml` in the repo.

  - Runtime usage (exact patterns from README):
    - Import & model instantiation:
      ```py
      import torchaudio as ta
      from chatterbox.tts import ChatterboxTTS

      model = ChatterboxTTS.from_pretrained(device="cuda")
      wav = model.generate(text)
      ta.save("out.wav", wav, model.sr)
      ```
    - Multilingual use:
      ```py
      from chatterbox.mtl_tts import ChatterboxMultilingualTTS
      model = ChatterboxMultilingualTTS.from_pretrained(device=device)
      wav = model.generate(text, language_id="fr")
      ta.save(..., model.sr)
      ```
    - Voice cloning / audio prompt: pass `audio_prompt_path` (path to WAV) to `generate()`.

  - Important runtime properties to capture & enforce in adapter:
    - `model.sr` (sample rate) — use for saving and metadata.
    - Output is watermarked (PerTh watermark). Provide a metadata flag `watermarked: true` and optionally support watermark extraction for QA.
    - Controls available: `exaggeration`, `cfg_weight` (affect pacing/intensity); expose these as voice_config options.

  - Dependencies & environment notes:
    - Requires `torchaudio` and a matching `torch` wheel compatible with CUDA (if using GPU). Ensure pinned torch/CUDA versions in `proposals/pyproject_adk.diff` before manifest changes.
    - Recommend GPU (`cuda`) for acceptable latencies; CPU runs are possible for smoke/dev but slower.
    - Use Python 3.11 for parity with upstream tests where practical.

  - Testing strategy (planning; no code yet):
    - Unit tests (fast): mock `ChatterboxTTS.from_pretrained` and `model.generate` to return a small numpy/torch tensor; verify adapter saves WAV, returns metadata with `duration_seconds` and `sample_rate` that matches `model.sr`.
    - Local smoke test: use `coqui TTS` as a lightweight fallback to exercise the full pipeline (real synthesis, fast) in CI/unit runs.
    - Gated integration: a GPU-backed smoke that runs Chatterbox real model; gated by `SMOKE_ADK=1` (or `SMOKE_TTS=1`) to avoid running on every CI.
    - Watermark verification test: optional gated test that runs the watermark extraction path using `perth` helper in the repo.

  - ADK & artifact publishing considerations:
    - Adapter should `probe_sdk(non_fatal=True)` and only call `adk_helpers.publish_artifact()` when ADK is present; otherwise skip publishing and return local path + metadata.
    - Artifact metadata: include `engine: "chatterbox"`, `model_version` (from `from_pretrained` repo/version), `voice_config`, `duration_seconds`, `sample_rate`, `watermarked`.

  - Security / license / policy:
    - Chatterbox is MIT-licensed (per README) but outputs are watermarked; document this in the adapter README and in the QA gating so downstream consumers know watermark is present.
    - Ensure any production use complies with internal policy on watermarked TTS outputs and consent for voice cloning.

  - Edge cases & robustness (planning):
    - Retry/backoff for transient CUDA/OOM errors; fail-fast for missing device when `require_adk()` semantics demand GPU.
    - Validate input sample/voice prompt format before calling `generate()`; reject non-mono or non-matching sample rates or resample as necessary.
    - Support `device` selection in config (e.g., `"cuda"`, `"cpu"`, or explicit device index).

  - Minimal adapter interface (for later implementation):
    - `class TtsAdapter(Protocol):`
      - `def synthesize(self, text: str, voice_config: dict | None = None) -> tuple[Path, dict]:` returns `(wav_path, metadata)`
    - Runtime config keys: `engine`, `device`, `audio_prompt_path`, `exaggeration`, `cfg_weight`, `model_version`.

  - Next planning steps (no code until authorization):
    1. Confirm target Python version and whether `chatterbox-tts` will be added to the `adk` optional extra in `pyproject` (I will prepare `proposals/pyproject_adk.diff` when you approve manifest changes).
    2. Confirm gating environment variable for Chatterbox smoke runs (`SMOKE_TTS=1` or reuse `SMOKE_ADK=1`).
    3. On your explicit "you may write code" signal, I will implement the adapter scaffold + unit tests and present a diff for review.

  #### Chatterbox Implementation — Developer Notes (detailed, append-only)

  - Purpose: this section collects the exact, actionable developer notes that should live in this doc so implementers have everything they need without creating files yet. **Do not create any files**; these notes belong in `docs/IMPLEMENTATION_TASKS.md` only until you explicitly authorize file creation.

  - Quick wiring checklist (developer steps when implementing):
    1. Implement `TtsAgent` with a simple decision heuristic: prefer ADK/FunctionTool invocation when `probe_sdk(non_fatal=True)` indicates ADK present and `adk_helpers.tool_available('com.sparkle_motion.tts.chatterbox')` is true. Fall back to the `Coqui` adapter otherwise.
    2. Implement adapter interface `TtsAdapter` (protocol) with `synthesize(text, voice_config) -> tuple[Path, dict]` returning a local `file://` WAV path plus metadata dict.
    3. Adapter must capture `model.sr` and compute `duration_seconds` from saved audio (use `soundfile` or `torchaudio.info()`) and compute `checksum_sha256`.
    4. When ADK is present and `publish=True` option is requested, call `adk_helpers.publish_artifact(path, metadata)` and include the returned artifact URI in the FunctionTool output.

  - Example FunctionTool invocation (pseudocode — keep in docs):

    ```json
    {
      "tool_id": "com.sparkle_motion.tts.chatterbox",
      "name": "tts_chatterbox.generate",
      "input": {
        "text": "Hello world",
        "voice_config": { "voice": "alloy", "language": "en-US" }
      }
    }
    ```

  - Example voice_config snippet (for implementers to reference):

    ```json
    {
      "voice": "alloy",
      "language": "en-US",
      "sample_rate": 24000,
      "format": "wav",
      "speed": 1.0,
      "pitch_shift": 0.0,
      "audio_prompt_path": null
    }
    ```

  - Gating expression (exact form to document in adapter README/tests):
    - Treat TTS smoke as enabled when `(os.getenv('SMOKE_TTS') == '1') or (os.getenv('SMOKE_TTS') is None and os.getenv('SMOKE_ADK') == '1')`.

  - Retryable exceptions (recommended):
    - RPC 5xx / transient network errors
    - CUDA OOM when model can be retried with a different device fallback or after a short delay
    - Connection reset / transient filesystem I/O errors when saving artifacts
    - Implementation note: be conservative — only retry on exceptions that are known-transient; log full trace and surface a clear user-facing error when exhausted.

  - Small code snippets (pseudocode guidance — put in adapter docstrings):

    ```py
    # probe and choose backend
    sdk_ok = probe_sdk(non_fatal=True)
    if sdk_ok and adk_helpers.tool_available('com.sparkle_motion.tts.chatterbox'):
        backend = ChatterboxBackend(device=cfg.device)
    else:
        backend = CoquiBackend(device='cpu')

    wav_path, metadata = backend.synthesize(text, voice_config)
    if sdk_ok and publish:
        artifact_uri = adk_helpers.publish_artifact(wav_path, metadata)
        metadata['artifact_uri'] = artifact_uri
    return wav_path, metadata
    ```

  - Unit test patterns (to include in `tests/unit/test_tts_agent.py` once authorized):
    - Mock `ChatterboxTTS.from_pretrained` to return an object with `.generate()` that returns a small tensor/array and `.sr` set.
    - Assert that `TtsAgent.synthesize()` writes a WAV file, returns `sample_rate==model.sr`, `duration_seconds` is approximately correct, and `checksum_sha256` matches the file content.
    - Test ADK absent vs present flows by mocking `probe_sdk(non_fatal=True)` and `adk_helpers.publish_artifact()`.

  - Example local smoke run (docs guidance only — do not automate installs):

    ```bash
    # Run fast Coqui-based smoke locally (no ADK):
    SMOKE_TTS=0 pytest -q tests/unit/test_tts_agent.py::test_coqui_smoke

    # Run gated Chatterbox smoke (requires ADK and GPU):
    SMOKE_TTS=1 pytest -q tests/smoke/test_tts_chatterbox_adk.py::test_chatterbox_inference
    ```

  - Artifact metadata mapping (fields that must be present):
    - `artifact_type`: "audio/wav"
    - `created_at`: RFC3339 timestamp
    - `generated_by.tool_id`: `com.sparkle_motion.tts.chatterbox` (or `coqui`)
    - `model.name` / `model.version` when available
    - `backend.name`: `chatterbox` or `coqui`
    - `audio.sample_rate`, `audio.channels`, `audio.duration_ms`
    - `checksum_sha256`, `watermarked` (bool)

  - Watermarking / QA notes:
    - Chatterbox outputs are watermarked; include `watermarked: true` in metadata and a pointer to the QA guidance for watermark verification.
    - If downstream systems need non-watermarked audio, document policy + approval path — do not provide a bypass in code.

  - Security & licensing reminder (must be present in adapter README):
    - Chatterbox is MIT-licensed (per README) but outputs are watermarked; document this in the adapter README and in the QA gating so downstream consumers know watermark is present.
    - Ensure any production use complies with internal policy on watermarked TTS outputs and consent for voice cloning.

  - Edge cases & robustness (planning):
    - Retry/backoff for transient CUDA/OOM errors; fail-fast for missing device when `require_adk()` semantics demand GPU.
    - Validate input sample/voice prompt format before calling `generate()`; reject non-mono or non-matching sample rates or resample as necessary.
    - Support `device` selection in config (e.g., `"cuda"`, `"cpu"`, or explicit device index).

  - Minimal adapter interface (for later implementation):
    - `class TtsAdapter(Protocol):`
      - `def synthesize(self, text: str, voice_config: dict | None = None) -> tuple[Path, dict]:` returns `(wav_path, metadata)`
    - Runtime config keys: `engine`, `device`, `audio_prompt_path`, `exaggeration`, `cfg_weight`, `model_version`.

  - Next planning steps (no code until authorization):
    1. Confirm target Python version and whether `chatterbox-tts` will be added to the `adk` optional extra in `pyproject` (I will prepare `proposals/pyproject_adk.diff` when you approve manifest changes).
    2. Confirm gating environment variable for Chatterbox smoke runs (`SMOKE_TTS=1` or reuse `SMOKE_ADK=1`).
    3. On your explicit "you may write code" signal, I will implement the adapter scaffold + unit tests and present a diff for review.

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
