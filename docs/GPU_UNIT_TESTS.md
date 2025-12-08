# GPU Unit Tests

This document outlines a set of GPU-enabled unit tests that exercise the real model adapters without mocks. These tests require an NVIDIA GPU, appropriate CUDA drivers, and downloaded model weights. They are intended to catch integration issues, validate end-to-end behavior, and ensure the fixture fallbacks correctly mirror real adapter outputs.

## Test Markers & Configuration

- **Marker**: `@pytest.mark.gpu` — identifies tests requiring GPU hardware.
- **Environment gating**: tests now fail immediately when `CUDA_VISIBLE_DEVICES=""` or no CUDA-capable GPU is detected (no automatic skips).
- **Model downloads**: tests assume Hugging Face models are cached locally or download on first run (expect long initial execution).
- **pytest.ini additions**:
  ```ini
  markers =
      gpu: tests requiring NVIDIA GPU and real model weights (skip by default)
  ```
- **Selective execution**: `pytest -m gpu` runs only GPU tests; `pytest -m "not gpu"` runs everything else.

---

## Image Generation (SDXL) Tests

### `test_sdxl_render_single_image_real`
- **Purpose**: Verify SDXL pipeline loads, renders a single 1024×1024 PNG, and returns valid metadata.
- **Inputs**: prompt="Cinematic sunrise over mountains", seed=42, steps=20, cfg_scale=7.5.
- **Assertions**:
  - Output file exists and is a valid PNG.
  - Metadata includes `engine="sdxl"`, `seed=42`, `phash` (perceptual hash), and `width=1024`.
  - File size > 100KB (real render, not fixture placeholder).
- **Flags**: `SMOKE_ADAPTERS=1`, `IMAGES_SDXL_FIXTURE_ONLY=0`.
- **Status**: done (2025-12-08) — see `tests/gpu/test_images_sdxl_gpu.py::test_sdxl_render_single_image_real`.

### `test_sdxl_render_batch`
- **Purpose**: Render batch of 4 images with different seeds, ensure each is distinct.
- **Inputs**: prompt="Hero portrait", count=4, batch_start=0, seed=100.
- **Assertions**:
  - 4 files generated with sequential seeds (100, 101, 102, 103).
  - Each `phash` is unique (no duplicate images).
  - All images have identical dimensions.
- **Status**: done (2025-12-08) — see `tests/gpu/test_images_sdxl_gpu.py::test_sdxl_render_batch`.

### `test_sdxl_negative_prompt`
- **Purpose**: Validate negative prompt steering.
- **Inputs**: prompt="Forest", negative_prompt="dark, gloomy", seed=99.
- **Assertions**:
  - Image rendered successfully.
  - Metadata includes both `prompt` and `negative_prompt` fields.
  - Compare perceptual hash against baseline (optional).
- **Status**: done (2025-12-08) — see `tests/gpu/test_images_sdxl_gpu.py::test_sdxl_negative_prompt`.

### `test_sdxl_custom_dimensions`
- **Purpose**: Test non-square resolutions (512×768, 1280×720).
- **Inputs**: prompt="Wide landscape", width=1280, height=720, seed=50.
- **Assertions**:
  - Output PNG matches requested dimensions.
  - No CUDA OOM errors for supported resolutions.
- **Status**: done (2025-12-08) — see `tests/gpu/test_images_sdxl_gpu.py::test_sdxl_custom_dimensions`.

### `test_sdxl_determinism`
- **Purpose**: Run same prompt/seed twice, verify identical outputs.
- **Inputs**: prompt="Test consistency", seed=7, steps=15, run twice.
- **Assertions**:
  - Both renders produce byte-identical PNG files (or matching phashes within tolerance).
- **Status**: done (2025-12-08) — see `tests/gpu/test_images_sdxl_gpu.py::test_sdxl_determinism`.

---

## Video Generation (Wan) Tests

### `test_wan_render_short_clip`
- **Purpose**: Validate Wan2.1 I2V pipeline renders a 16-frame clip.
- **Inputs**: prompt="Gentle camera pan", num_frames=16, fps=8, seed=200, start_frame=PNG bytes, end_frame=PNG bytes.
- **Assertions**:
  - Output MP4 exists and is playable.
  - Metadata includes `engine="wan"`, `frame_count=16`, `fps=8`, `duration_s=2.0`.
  - File size > 10KB (real video, not JSON placeholder).
- **Status**: done (2025-12-08) — see `tests/gpu/test_videos_wan_gpu.py::test_wan_render_short_clip`.

### `test_wan_keyframe_interpolation`
- **Purpose**: Provide distinct start/end frames, ensure smooth interpolation.
- **Inputs**: start_frame=desert PNG, end_frame=ocean PNG, num_frames=32, fps=16.
- **Assertions**:
  - Video transitions from start to end frame.
  - No abrupt cuts or black frames (manual or automated visual check).
  - Duration matches `num_frames / fps`.
- **Status**: done (2025-12-08) — see `tests/gpu/test_videos_wan_gpu.py::test_wan_keyframe_interpolation`.

### `test_wan_seed_reproducibility`
- **Purpose**: Render same clip with fixed seed twice, verify consistency.
- **Inputs**: prompt="Slow zoom", seed=150, num_frames=24, fps=12, run twice.
- **Assertions**:
  - Both outputs have identical frame checksums (or high SSIM score).
- **Status**: done (2025-12-08) — see `tests/gpu/test_videos_wan_gpu.py::test_wan_seed_reproducibility`.

### `test_wan_adaptive_chunking`
- **Purpose**: Request 128-frame clip, ensure adapter chunks correctly without OOM.
- **Inputs**: num_frames=128, fps=16, prompt="Long sequence", seed=300.
- **Assertions**:
  - Final MP4 contains ~128 frames (allow ±2 for codec rounding).
  - No CUDA OOM or chunking errors.
  - `metadata.chunk_count` reflects chunking strategy.
- **Status**: done (2025-12-08) — see `tests/gpu/test_videos_wan_gpu.py::test_wan_adaptive_chunking`.

---

## TTS (Text-to-Speech) Tests

### `test_tts_synthesize_single_line`
- **Purpose**: Synthesize one dialogue line with Chatterbox provider.
- **Inputs**: text="Hello, world.", voice_config={"provider": "chatterbox", "voice_id": "en_us_001"}, seed=42.
- **Assertions**:
  - Output WAV file exists and is valid audio.
  - Duration > 0.5s.
  - Metadata includes `provider_id="chatterbox"`, `voice_id`, `sample_rate`, `watermarked=true/false`.
- **Status**: done (2025-12-08) — see `tests/gpu/test_tts_chatterbox_gpu.py::test_tts_synthesize_single_line`.

### `test_tts_multiple_lines_deterministic`
- **Purpose**: Synthesize 3 lines with same seed, verify consistent outputs.
- **Inputs**: lines=["Line one.", "Line two.", "Line three."], seed=100.
- **Assertions**:
  - 3 WAV files generated.
  - Re-running with same seed produces byte-identical (or near-identical) audio.
  - Metadata includes per-line `artifact_uri`.
- **Status**: done (2025-12-08) — see `tests/gpu/test_tts_chatterbox_gpu.py::test_tts_multiple_lines_deterministic`.

### `test_tts_voice_profile_routing`
- **Purpose**: Request specific voice profile, ensure correct provider selected.
- **Inputs**: voice_config={"provider": "polly", "voice_id": "Joanna"}, text="Testing voice".
- **Assertions**:
  - Adapter routes to Polly provider (or fallback if unavailable).
  - Metadata reflects chosen provider.
- **Status**: done (2025-12-08) — see `tests/gpu/test_tts_chatterbox_gpu.py::test_tts_voice_profile_routing`.

### `test_tts_policy_violation`
- **Purpose**: Submit text containing banned words, expect TTSPolicyViolation.
- **Inputs**: text="This contains weaponized content".
- **Assertions**:
  - `TTSPolicyViolation` raised.
  - No WAV file created.
- **Status**: done (2025-12-08) — see `tests/gpu/test_tts_chatterbox_gpu.py::test_tts_policy_violation`.

### `test_tts_quota_handling`
- **Purpose**: Mock quota exhaustion, verify fallback to next provider.
- **Setup**: Inject `TTSQuotaExceeded` from primary provider.
- **Assertions**:
  - Adapter selects fallback provider.
  - WAV still generated.
  - Metadata indicates failover.
- **Status**: done (2025-12-08) — see `tests/gpu/test_tts_chatterbox_gpu.py::test_tts_quota_handling`.

---

## Lipsync (Wav2Lip) Tests

### `test_lipsync_single_clip`
- **Purpose**: Apply Wav2Lip to video + audio, verify output.
- **Inputs**: video_path=short MP4 (raw video), audio_path=WAV (dialogue).
- **Assertions**:
  - Output MP4 exists and is playable.
  - Duration matches input video duration.
  - Audio track embedded correctly.
  - Metadata includes `engine="wav2lip"` (hard fails if fixture path used).
- **Status**: done (2025-12-08) — see `tests/gpu/test_lipsync_wav2lip_gpu.py::test_lipsync_single_clip`.

### `test_lipsync_multiple_audio_tracks`
- **Purpose**: Merge video with concatenated dialogue audio.
- **Inputs**: video_path=MP4, audio_paths=[line1.wav, line2.wav, line3.wav].
- **Assertions**:
  - Output MP4 duration = sum of audio durations.
  - No audio sync drift.
- **Status**: done (2025-12-08) — see `tests/gpu/test_lipsync_wav2lip_gpu.py::test_lipsync_multiple_audio_tracks` (fails if adapter reports fixture engine).
- **Status**: done (2025-12-08) — see `tests/gpu/test_lipsync_wav2lip_gpu.py::test_lipsync_multiple_audio_tracks`.

---

## Assembly (ffmpeg) Tests

### `test_assemble_single_clip`
- **Purpose**: Concatenate 1 video clip (no audio), produce final MP4.
- **Inputs**: clips=[shot1.mp4], audio=None.
- **Assertions**:
  - Output file exists and is valid MP4.
  - Duration matches input.
  - No audio track present.
- **Status**: done (2025-12-08) — see `tests/gpu/test_assemble_ffmpeg_gpu.py::test_assemble_single_clip`.
- **Notes**: Test enforces the real ffmpeg engine (`engine="ffmpeg"`) and fails if the adapter falls back to fixtures.

### `test_assemble_multiple_clips_with_audio`
- **Purpose**: Concatenate 3 clips + background audio track.
- **Inputs**: clips=[shot1.mp4, shot2.mp4, shot3.mp4], audio=bgm.mp3.
- **Assertions**:
  - Final MP4 duration = sum of clip durations.
  - Audio track plays continuously.
  - No frame drops or sync issues.
- **Status**: done (2025-12-08) — see `tests/gpu/test_assemble_ffmpeg_gpu.py::test_assemble_multiple_clips_with_audio`.
- **Notes**: Requires `SMOKE_ASSEMBLE=1` with `ASSEMBLE_FFMPEG_FIXTURE_ONLY=0` and fails if metadata reports any engine other than `ffmpeg`.

### `test_assemble_heterogeneous_clips`
- **Purpose**: Mix clips with different resolutions/codecs, verify ffmpeg normalization.
- **Inputs**: clips=[1280×720.mp4, 1920×1080.mp4].
- **Assertions**:
  - ffmpeg rescales/reencodes to consistent output.
  - Final MP4 playable.
  - Logs warning about resolution mismatch.

---

## Script Agent (LLM Plan Generation) Tests

### `test_script_agent_generate_plan_real_llm`
- **Purpose**: Call `script_agent.generate_plan` with real ADK agent (Gemini/GPT).
- **Inputs**: prompt="Generate a 2-shot plan about a robot exploring Mars", seed=42.
- **Assertions**:
  - `MoviePlan` returned with `shots` list (length ≥ 2).
  - `base_images` list populated.
  - `dialogue_timeline` valid.
  - No synthetic fallback metadata (`source != "script_agent.entrypoint.synthetic"`).

### `test_script_agent_determinism`
- **Purpose**: Generate plan with fixed seed twice, verify consistent structure.
- **Inputs**: same prompt, seed=100, run twice.
- **Assertions**:
  - Both plans have identical shot count.
  - Shot descriptions/prompts match (allowing minor LLM variance if acceptable).

### `test_script_agent_resource_limits`
- **Purpose**: Request plan with 50 shots, verify `PlanResourceError` raised.
- **Inputs**: prompt="Create 50 shots", seed=200.
- **Assertions**:
  - `PlanResourceError` raised (exceeds `SCRIPT_AGENT_MAX_SHOTS`).
  - No plan returned.

### `test_script_agent_policy_violation`
- **Purpose**: Submit prompt with banned content, expect `PlanPolicyViolation`.
- **Inputs**: prompt="Show weaponized robots attacking civilians".
- **Assertions**:
  - `PlanPolicyViolation` raised.
  - No plan artifact persisted.

---

## Production Agent (End-to-End Orchestration) Tests

### `test_production_agent_full_run_real_adapters`
- **Purpose**: Execute complete pipeline with real SDXL/Wan/TTS/Lipsync/ffmpeg.
- **Inputs**: small 2-shot MoviePlan, mode="run", all `SMOKE_*` flags enabled.
- **Assertions**:
  - `status="success"` in response.
  - `artifact_uris` includes `video_final` entry.
  - Final MP4 exists at `artifacts/runs/<run_id>/<plan_id>/final/<plan_id>-video_final.mp4`.
  - File size > 50KB (real render).
  - All steps succeeded (no `status="failed"` in step records).

### `test_production_agent_resume_after_failure`
- **Purpose**: Simulate failure at `video` stage, verify resume logic.
- **Setup**: Inject failure mid-run, then re-invoke with `resume=True`.
- **Assertions**:
  - First run fails at `video` stage.
  - Resume skips completed stages (`images`, `tts`).
  - Final run succeeds and generates `video_final`.

### `test_production_agent_rate_limit_handling`
- **Purpose**: Trigger rate-limit condition, verify queuing.
- **Setup**: Inject `RateLimitQueued` error from `images` stage.
- **Assertions**:
  - Response includes `status="queued"` and `queue.ticket_id`.
  - Step record marks `status="queued"`.
  - Retry logic dequeues and completes run.

---

## GPU Context Management Tests

### `test_gpu_context_model_offload`
- **Purpose**: Validate `gpu_utils.model_context` correctly offloads/reloads models.
- **Inputs**: Load SDXL, render image, load Wan, render video, unload both.
- **Assertions**:
  - VRAM usage drops after each unload.
  - No CUDA OOM errors.
  - Models reload successfully on next use.

### `test_gpu_context_concurrent_requests`
- **Purpose**: Simulate 2 overlapping requests (SDXL + Wan), ensure serialization.
- **Setup**: Start SDXL render, then attempt Wan render before SDXL finishes.
- **Assertions**:
  - Wan render waits for SDXL to release GPU.
  - Both renders succeed.
  - `GpuBusyError` handled gracefully.

### `test_gpu_oom_recovery`
- **Purpose**: Force OOM condition, verify adapter fallback.
- **Setup**: Request 2048×2048 SDXL image on GPU with insufficient VRAM.
- **Assertions**:
  - `ModelOOMError` caught.
  - Adapter falls back to fixture or lower resolution.
  - No process crash.

---

## Deduplication Tests

### `test_images_dedupe_identical_prompts`
- **Purpose**: Render same prompt twice with dedupe enabled, verify only 1 artifact stored.
- **Inputs**: prompt="Test", seed=42, dedupe=True, run twice.
- **Assertions**:
  - Second render returns cached URI.
  - Only 1 PNG file on disk.
  - Metadata includes `deduped=True` on second call.

### `test_images_dedupe_phash_matching`
- **Purpose**: Render similar prompts, verify perceptual hash deduplication.
- **Inputs**: prompt1="Sunset", prompt2="Sunset scene", dedupe=True, phash_threshold=5.
- **Assertions**:
  - Second render reuses first artifact (phashes within threshold).
  - Metadata includes `duplicate_of=<uri>`.

---

## Test Execution Guide

1. **Install GPU dependencies**:
   ```bash
   pip install -r requirements-ml.txt
   ```

2. **Download models** (first run):
   ```bash
   python scripts/colab_drive_setup.py --download-models
   ```

3. **Run GPU tests**:
   ```bash
   pytest -m gpu --maxfail=1 -v
   ```

4. **Skip GPU tests** (CI or CPU-only):
   ```bash
   pytest -m "not gpu"
   ```

5. **Environment variables**:
   - `SMOKE_ADAPTERS=1` — enable real SDXL/Wan adapters.
   - `SMOKE_TTS=1` — enable real TTS synthesis.
   - `SMOKE_LIPSYNC=1` — enable real Wav2Lip.
   - `CUDA_VISIBLE_DEVICES=""` — force GPU tests to skip.

---

## Next Steps

- [ ] Implement pytest fixtures for model loading/caching.
- [ ] Add GPU skip markers (`@pytest.mark.skipif(not torch.cuda.is_available(), ...)`).
- [ ] Capture baseline outputs (phashes, checksums) for regression testing.
- [ ] Add timeout decorators (GPU tests can run 30s–5min each).
- [ ] Create CI workflow for nightly GPU test runs (separate from PR checks).
- [ ] Document expected VRAM requirements per test (e.g., SDXL ≥12GB, Wan ≥16GB).
