# Sparkle Motion Command-Line Playbook

These instructions walk through running the full Sparkle Motion pipeline without the Colab UI. You will generate a MoviePlan with `script_agent`, feed it into `production_agent`, monitor the run, and download the final `video_final.mp4` artifact — all from a shell.

## 1. Prerequisites

- Linux host with Python 3.10+, `ffmpeg`, and (optionally) an NVIDIA GPU if you plan to run the heavy adapters instead of the fixtures.
- Access to the repo root (`sparkle_motion/`).
- Credentials exported in your shell: `ADK_PROJECT`, `ADK_API_KEY`, `HF_TOKEN` (or whatever secrets your ADK project requires).
- Hugging Face CLI logged in (`huggingface-cli login`) if you expect the production adapters to download private repos.
- `jq` installed for parsing JSON responses (`sudo apt-get install jq` on Debian/Ubuntu).

## 2. Create a Python environment

```bash
cd /home/phil/work/sparkle_motion
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-ml.txt
```

If you only need the lightweight fixtures, `requirements-dev.txt` is enough. For real SDXL/Wan/tts workloads, `requirements-ml.txt` pulls the GPU stacks defined in `docs/MODEL_INSTALL_NOTES.md`.

## 3. Export runtime configuration

These flags keep artifacts on disk, disable fixture-only behavior, and ensure every production stage actually runs. Adjust values for your environment before starting the servers:

```bash
export ADK_PROJECT="your-project"
export ADK_API_KEY="your-token"
export HF_TOKEN="hf_xxx"              # optional unless your models are private
export ARTIFACTS_BACKEND=filesystem    # keep every artifact under ./artifacts/
export ARTIFACTS_DIR="$PWD/artifacts"
export SPARKLE_LOCAL_RUNS_ROOT="$PWD/artifacts/runs"
export ADK_USE_FIXTURE=0               # set to 1 to force deterministic fixtures everywhere
export DETERMINISTIC=0                 # disable name mangling when you want unique files
export SMOKE_ADAPTERS=1                # REQUIRED: actually run images/videos stages
export SMOKE_TTS=1                     # REQUIRED: run dialogue synthesis
export SMOKE_LIPSYNC=1                 # REQUIRED: run lipsync when dialogue exists
```

With `ADK_USE_FIXTURE=0` the adapters will try real engines (SDXL, Wan, etc.) whenever the corresponding `SMOKE_*` flag is `1`. If you prefer the fast deterministic fixtures, leave `ADK_USE_FIXTURE=1`; the rest of the flow stays identical and still outputs a valid `video_final.mp4` placeholder.

## 4. Start the FunctionTools

Open two terminals (keep them running):

**Terminal A – script_agent**

```bash
cd /home/phil/work/sparkle_motion
source .venv/bin/activate
PYTHONPATH=src python scripts/run_function_tool.py --tool script_agent --port 5001
```

**Terminal B – production_agent**

```bash
cd /home/phil/work/sparkle_motion
source .venv/bin/activate
PYTHONPATH=src python scripts/run_function_tool.py --tool production_agent --port 5008
```

The helper script consults `configs/tool_registry.yaml`, so the default host/port pairing matches the notebook UI (127.0.0.1:5001 and 127.0.0.1:5008). Leave both servers running while you issue the CLI requests below.

## 5. Generate a MoviePlan from the CLI

Create a request payload that contains the title and prompt you want to test:

```bash
mkdir -p requests responses
cat > requests/script_agent.json <<'JSON'
{
	"title": "Luminous Canyon",
	"prompt": "Cinematic 30-second teaser that opens on a glowing canyon, cuts to a wandering hero, and ends with a whispered line about hope.",
	"shots": []
}
JSON
```

Send it to `script_agent` and capture the artifact URI (the plan JSON is emitted under `artifacts/` when `ARTIFACTS_BACKEND=filesystem`):

```bash
curl -sS -X POST http://127.0.0.1:5001/invoke \
	-H "Content-Type: application/json" \
	--data-binary @requests/script_agent.json \
	| tee responses/script_agent.json

PLAN_URI=$(jq -r '.artifact_uri' responses/script_agent.json | sed 's#^file://##')
echo "MoviePlan stored at: $PLAN_URI"
```

- Edit `requests/script_agent.json` whenever you need a new title or prompt.
- If `PLAN_URI` is empty, check the server log for `script_agent.generate_plan failed` messages; fix credentials or disable fixture mode before retrying.

## 6. Execute the plan with production_agent

Point `production_agent` at the plan file you just created. Use `mode: "dry"` for a zero-cost simulation or `mode: "run"` to render assets and build the final MP4.

```bash
cat > requests/production_run.json <<JSON
{
	"plan_uri": "${PLAN_URI}",
	"mode": "run"
}
JSON

curl -sS -X POST http://127.0.0.1:5008/invoke \
	-H "Content-Type: application/json" \
	--data-binary @requests/production_run.json \
	| tee responses/production_run.json

RUN_ID=$(jq -r '.run_id' responses/production_run.json)
echo "Run ID: $RUN_ID"
```

The response also includes a `steps` array (dry mode) or an empty array while `production_agent` is still working. Keep the JSON file — it is helpful when filing bugs.

### Monitor progress

Poll `/status` until every step reports `succeeded`:

```bash
watch -n 5 "curl -sS 'http://127.0.0.1:5008/status?run_id=${RUN_ID}' | jq '.steps'"
```

### Inspect artifacts

Once the run completes, list every stage artifact (handy for logs, intermediate clips, etc.):

```bash
curl -sS "http://127.0.0.1:5008/artifacts?run_id=${RUN_ID}" | jq '.stages'
```

To grab only the final deliverable metadata:

```bash
curl -sS "http://127.0.0.1:5008/artifacts?run_id=${RUN_ID}&stage=finalize" \
	| jq '.artifacts[0]'
```

When `ARTIFACTS_BACKEND=filesystem`, the finalize payload includes `local_path`, pointing at `artifacts/runs/<run_id>/<plan_id>/final/<plan_id>-video_final.mp4`.

## 7. Retrieve the MP4

```bash
RUN_DIR="artifacts/runs/${RUN_ID}"
PLAN_DIR=$(ls "$RUN_DIR" | head -n 1)   # directory is the plan_id slug
FINAL_MP4="$RUN_DIR/$PLAN_DIR/final/${PLAN_DIR}-video_final.mp4"
cp "$FINAL_MP4" outputs/${RUN_ID}_video_final.mp4
```

Play the clip (`ffplay outputs/${RUN_ID}_video_final.mp4`) or upload it to your review tooling. If you enabled the real adapters, expect multi-minute runs; with fixtures the full pipeline typically finishes in seconds.

## 8. Shutdown & cleanup

1. `Ctrl+C` the FunctionTool servers (script_agent and production_agent terminals).
2. Deactivate the virtualenv (`deactivate`).
3. Remove temporary artifacts when you are done debugging: `rm -rf artifacts/runs/* outputs/*`.

## 9. Optional: single-command workflow dry-run

If you just want to validate the registry wiring (no custom prompt), you can run the bundled workflow definition:

```bash
./scripts/sparkle-motion configs/workflow_agent.yaml --dry-run
```

The CLI runner walks the same `script_stage → production_stage` graph that the ADK WorkflowAgent uses, but it always feeds a canned prompt. For real creative input, stick to the two-step HTTP flow above so you can edit the plan request freely.

---

Once the CLI path above succeeds, the Colab UI should behave the same way (it issues the same `/invoke`, `/status`, and `/artifacts` calls). If the UI still misbehaves, open an issue with:
- `requests/script_agent.json`
- `responses/script_agent.json`
- `responses/production_run.json`
- The relevant server logs (`artifacts/runs/<run_id>/run_events.json`)

That bundle makes it easy to diff CLI vs. notebook behavior.
