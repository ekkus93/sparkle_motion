# FunctionTool packaging

This directory contains the runtime scaffolding for every Sparkle Motion stage.
Each folder mirrors the ADK ToolRegistry entry names and houses the config
needed to launch the lightweight FastAPI server implemented in
`sparkle_motion.function_tools`.

## Tools

| Tool ID | Stage | Default port | Description |
| --- | --- | --- | --- |
| `script_agent` | Script planning | 5001 | Emits MoviePlan artifacts using the schema registry. |
| `images_sdxl` | Image generation | 5002 | Produces start/end frames + prompts. |
| `videos_wan` | Video synthesis | 5003 | Generates raw clips per shot. |
| `tts_chatterbox` | Dialogue audio | 5004 | Creates speech WAVs for dialogue lines. |
| `lipsync_wav2lip` | Lipsync | 5005 | Merges video clips with audio to produce synced shots. |
| `assemble_ffmpeg` | Final assembly | 5006 | Concatenates shot clips into a final movie artifact. |
| `qa_qwen2vl` | QA/Policy | 5007 | Emits QAReport artifacts driven by the QA policy bundle. |

## Running locally (Colab-friendly)

The Google Colab environment we target cannot run Docker, so every tool should
be started directly from the repo using the shared runner script. From the repo
root:

```bash
PYTHONPATH=src python scripts/run_function_tool.py --tool <tool_id> --port <port>
```

Use the port column from the table above. The server exposes
Prometheus-style health endpoints at `/healthz` and the ADK `/invoke`
endpoint. To run multiple tools simultaneously (e.g., for local testing), open
additional terminals and launch each tool with its own port number.

### TTS flow

- `tts_chatterbox` now mirrors the production per-line synthesis flow: the
  FunctionTool receives a dialogue line at a time, calls `tts_agent.synthesize`,
  and publishes each clip as a `tts_audio` artifact with metadata covering
  `provider_id`, `voice_id`, sample rate, bit depth, duration, and a
  `watermarked` indicator. The orchestrator records these entries under
  `StepExecutionRecord.meta["tts"]["line_artifacts"]` so downstream stages can
  trace every WAV.
- Set `SMOKE_TTS=1` (or `SMOKE_ADAPTERS=1`) when you want the real adapter to
  run. Leaving both unset keeps the fixture adapter active, which emits
  deterministic WAVs with the same metadata schema—recommended for Colab/CI.
- The adapter should continue to respect `ADK_USE_FIXTURE=1`, but that flag no
  longer controls whether per-line artifacts are published; even fixture mode
  must return the full metadata envelope so policy gates and QA have consistent
  observability.

### Assemble flow

- `assemble_ffmpeg` defaults to a deterministic MP4 fixture so developers can
  exercise the production agent without `ffmpeg` installed. Enable the real
  concat/render path by setting `SMOKE_ASSEMBLE=1` (or the broader
  `SMOKE_ADAPTERS=1`) and ensuring an `ffmpeg` binary is available on `PATH`
  (override via `FFMPEG_PATH`).
- Requests must supply ordered clip descriptors (and optional audio) and the
  entrypoint publishes a `video_final` artifact with metadata covering the
  engine (`fixture` vs `ffmpeg`), plan identifiers, and the command tail
  captured from stdout/stderr.
- Force fixture mode even when smoke flags are set by exporting
  `ASSEMBLE_FFMPEG_FIXTURE_ONLY=1`; this mirrors the `options.fixture_only`
  field on the `/invoke` request body.

### Videos flow

- `videos_wan` now routes `/invoke` requests through the Wan2.1 adapter. The
  adapter defaults to a deterministic MP4 fixture so CI and Colab users can
  stay GPU-free while still exercising the metadata envelope (`plan_id`,
  `chunk_index`, seeds, and option snapshots).
- Enable the real Wan pipeline with `SMOKE_VIDEOS=1` (or
  `SMOKE_ADAPTERS=1/SMOKE_ADK=1`). The adapter honors
  `VIDEOS_WAN_MODEL/VIDEOS_WAN_DEVICE_PRESET/VIDEOS_WAN_DEVICE_MAP` and keeps
  the weights warm via `gpu_utils.model_context`. Set
  `VIDEOS_WAN_FIXTURE_ONLY=1` (or request `options.fixture_only=true`) to
  force the deterministic path even when smoke flags are present.
- Artifacts are published as `videos_wan_clip` MP4s alongside metadata from
  the adapter (engine, dimensions, chunk stats) plus entrypoint annotations
  (`request_id`, `plan_id`, `step_id`, `run_id`). Local fixture runs always
  resolve `artifact_uri` to `file://…` for easy assertion in unit/smoke tests.

## Tool metadata

Deployment metadata shared with ADK lives in `configs/tool_registry.yaml`. That
file lists the endpoint URL, schema artifacts, and retry hints for each tool.
Update the metadata file whenever you change a port or add new runtime
parameters so the bootstrap scripts can sync with the ToolRegistry and the
Colab profile.
