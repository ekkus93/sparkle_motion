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

## Tool metadata

Deployment metadata shared with ADK lives in `configs/tool_registry.yaml`. That
file lists the endpoint URL, schema artifacts, and retry hints for each tool.
Update the metadata file whenever you change a port or add new runtime
parameters so the bootstrap scripts can sync with the ToolRegistry and the
Colab profile.
