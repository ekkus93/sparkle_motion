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

## tts_chatterbox
- Task: Implement `synthesize_speech(text, voice_config) -> wav`
  - Adapter should support ADK-exposed TTS via agent OR local fallbacks (Coqui TTS) in dev.
  - Produce metadata (duration, sample_rate, voice_id) and publish WAV artifact.
  - Tests: unit tests for format/metadata; gated TTS smoke.
  - Estimate: 2–4 days

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
