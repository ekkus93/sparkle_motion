# Sparkle Motion Operations Guide

This document collects the detailed operational notes that back the slimmed-down `README.md`. Refer to it whenever you need the full ADK setup steps, GPU smoke instructions, or notebook walkthroughs.

## Agents vs FunctionTools
- **Agents (keep the `_agent` suffix):** `production_agent` is now the ADK root `LlmAgent` entry point. It calls the script stage (powered by `script_agent`) to draft a `MoviePlan`, then executes the downstream images/video/TTS/assemble stages plus QA. `script_agent` still runs as an ADK agent, but it is only invoked by the workflow's script stage—operators do not call it directly anymore.
- **Stages implemented as FunctionTools:** `images_sdxl`, `videos_wan`, `tts_chatterbox`, `lipsync_wav2lip`, `assemble_ffmpeg`, and `qa_qwen2vl` provide the heavy GPU/IO work. They no longer use the `_agent` suffix, and notebook/control-panel docs reference them strictly as FunctionTools.
- **Stages vs adapters:** higher-level orchestration modules inside `src/` (e.g., `images_stage`, `tts_stage`) coordinate retries/QA/policy checks and call the FunctionTools above. Only the two orchestration agents listed here surface ADK agent identifiers to users.

## Running ADK integration tests
This project includes an env-gated ADK integration test that is skipped by default to avoid accidental calls to external services. To run the integration test locally against the bundled fixture shim (safe default):

```bash
PYTHONPATH=.:src ADK_PUBLISH_INTEGRATION=1 ADK_PROJECT=testproject \
  pytest -q tests/test_function_tools/test_script_agent_entrypoint_adk_integration.py::test_publish_artifact_returns_artifact_uri
```

If you want the test to run against a real `google.adk` SDK (installed and authenticated in your environment), set `ADK_USE_FIXTURE=0` and make sure `ADK_PROJECT` is set to a valid project id. Example:

```bash
PYTHONPATH=.:src ADK_PUBLISH_INTEGRATION=1 ADK_PROJECT=<your-project> ADK_USE_FIXTURE=0 \
  pytest -q tests/test_function_tools/test_script_agent_entrypoint_adk_integration.py::test_publish_artifact_returns_artifact_uri
```

CI note: Do not add `tests/fixtures` to `PYTHONPATH` globally in CI jobs where the real ADK SDK is expected; the test inserts the fixtures by default and provides `ADK_USE_FIXTURE` to opt out when needed.

## Installing ADK extras and authentication
### Install & usage (ADK optional extra)
To install the optional ADK runtime extras for this repository:

```bash
pip install .[adk]
```

Environment variables required for ADK integration tests / features:

- `ADK_PUBLISH_INTEGRATION`
- `ADK_PROJECT`
- `GOOGLE_APPLICATION_CREDENTIALS`

### Configuring ADK authentication (`ADK_API_KEY`)
Some workflows (schema publishing, artifact uploads, production agents) also require `ADK_API_KEY` so the ADK SDK/CLI can authenticate. Quick setup:

1. **Obtain the key** from your ADK deployment (console/CLI) for the `sparkle-motion` project, or ask the team operating the ADK environment to issue one. Keys are not generated inside this repo.
2. **Populate `data/content/.sparkle.env`** with the issued values:

   ```bash
   # data/content/.sparkle.env
   ADK_PROJECT=sparkle-motion
   ADK_API_KEY=sk-...
   ```

   The file already contains these variables with empty defaults; editing it keeps secrets out of version control while remaining source-able for local work.
3. **Load the values** before running scripts that touch ADK:

   ```bash
   source data/content/.sparkle.env
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
   ```

   Alternatively, export `ADK_PROJECT` and `ADK_API_KEY` directly in your shell or CI environment variables.

Without these variables the ADK CLI/SDK cannot create artifacts (e.g., `scripts/publish_schemas.py` will fail with an auth error), so set them before running any publish/deploy steps.

#### Step-by-step: creating an `ADK_API_KEY`
1. **Locate the ADK control plane.** The URL/CLI profile is managed by your infra/ML platform team. Check `resources/adk_projects.json` for hints or ping the owners for the "ADK console" link and login instructions.
2. **Sign in.** Either:
   - Use the web console → log in with your corporate account and select the `sparkle-motion` project; or
   - Use the CLI → run `adk auth login --project sparkle-motion` (the command opens a browser window to finish auth).
3. **Create the key.** In the console, navigate to **Projects → sparkle-motion → Security → API keys → Create key** (menu labels may vary slightly) and grant at least "artifacts.write" scope. In the CLI you can run:

   ```bash
   adk keys create --project sparkle-motion --display-name "sparkle-motion-local" \
     --scopes artifacts.read artifacts.write
   ```

   The CLI prints a JSON blob that includes `apiKey`. Copy it immediately; most ADK deployments only show the secret once.
4. **Store the secret safely.** Paste the key into `data/content/.sparkle.env` (or your secrets manager) and keep it out of version control. Example:

   ```bash
   echo "ADK_API_KEY=sk-live-..." >> data/content/.sparkle.env
   ```
5. **Load it before publishing.** Source the env file or export the variable in CI:

   ```bash
   source data/content/.sparkle.env
   export ADK_PROJECT=sparkle-motion
   ```

If you do not have console access, file a request with the ADK owners—they are the only ones who can mint keys for the control plane.

### Forcing production runs to use real ADK agents
The repository defaults to fixture mode so local development does not make real ADK control-plane calls. To guarantee that production (or any run) uses the backed `google.adk` agents instead of fakes, complete **all** of the following:

1. **Install the ADK extras** so the real SDK is present:

   ```bash
   pip install .[adk]
   ```

2. **Disable fixture mode** by unsetting the variable or forcing it to zero:

   ```bash
   export ADK_USE_FIXTURE=0
   ```

   Any non-zero/truthy value keeps the local shim active, so production deploys must explicitly leave this unset or set to `0`.

3. **Provide the required credentials and project metadata:**

   ```bash
   export ADK_PROJECT=sparkle-motion            # or another valid project
   export ADK_API_KEY=sk-live-...
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

   When running the ADK publish integration test, also set `ADK_PUBLISH_INTEGRATION=1` so pytest collects it.

4. **Load the environment before invoking CLI/entrypoints.** A common pattern is sourcing `data/content/.sparkle.env` (where the variables above live) and exporting `PYTHONPATH=.:src` so scripts and `src/` imports coexist.

With those steps in place, the FunctionTool entrypoints instantiate real ADK agents via `adk_factory` and no fixture adapters are consulted. Any missing variable will cause the SDK constructors to raise, so treat failures as misconfiguration rather than falling back to local mocks.

## Running tests (canonical commands)
Use one of the commands below depending on the scope you want to run.

- **Run unit tests (folder-based):** runs tests placed under `tests/unit`.

  ```bash
  source /home/phil/mambaforge/bin/activate sparkle_motion
  PYTHONPATH=.:src pytest -q tests/unit
  ```

- **Run all tests (full repo run):** collects and runs every test under `tests/`.

  ```bash
  source /home/phil/mambaforge/bin/activate sparkle_motion
  PYTHONPATH=.:src pytest -q
  ```

Notes:
- We recommend keeping `PYTHONPATH=.:src` so both top-level scripts (e.g., `scripts/*`) and package code under `src/` import correctly during test collection.
- If you prefer marker- or CI-driven selection, use pytest markers (e.g., `-m unit`) or targeted folders instead of the full run.

## GPU-backed SDXL smoke tests
The `images_sdxl` FunctionTool defaults to a deterministic PNG fixture so you can develop without specialized hardware. To exercise the real SDXL pipeline on hardware such as Google Colab’s T4/A100 instances:

1. **Install the ML/runtime extras** once the repo is available in your Colab session (adjust CUDA wheels as needed):

   ```bash
   pip install -r requirements-ml.txt diffusers==0.30.2 torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **Export the GPU env vars** so the adapter leaves fixture mode but still publishes artifacts via the local shim:

   ```bash
   export PYTHONPATH=.:src
   export ADK_USE_FIXTURE=1
   export SMOKE_IMAGES=1
   export IMAGES_SDXL_FIXTURE_ONLY=0
   export IMAGES_SDXL_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
   export IMAGES_SDXL_DEVICE="cuda"
   ```

   Optional knobs:
   - `IMAGES_SDXL_CACHE_TTL_S` — seconds to keep the SDXL pipeline warm inside `gpu_utils.model_context`.
   - `IMAGES_SDXL_ENABLE_XFORMERS=0` — disable memory-efficient attention if the bundled `xformers` wheel is incompatible with your runtime.

3. **Run the focused smoke tests** to verify real image generation:

   ```bash
   pytest -q tests/test_function_tools/test_images_sdxl_entrypoint.py tests/smoke/test_images_sdxl_smoke.py
   ```

Troubleshooting:
- A `503` with `{"detail":"gpu busy"}` indicates another process is holding the GPU lock; rerun once the previous job exits.
- Missing `torch`/`diffusers` wheels trigger a logged warning and automatic fallback to the deterministic fixture so tests still pass, but no real GPU validation occurs.

## GPU-backed Wan 2.1 smoke tests
The `videos_wan` FunctionTool mirrors the SDXL setup: fixture MP4s by default so development is GPU-free, and the real Wan2.1 I2V pipeline is enabled only when explicitly requested. To validate the real pipeline on GPU hardware (Colab, A100, etc.):

1. **Install the Wan runtime extras** (diffusers with Wan support + CUDA wheels). Adjust versions to match your CUDA runtime.

   ```bash
   pip install -r requirements-ml.txt diffusers==0.30.2 torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **Export the env vars** so the adapter leaves fixture mode but still publishes artifacts via the local shim:

   ```bash
   export PYTHONPATH=.:src
   export ADK_USE_FIXTURE=1                 # keep ADK shim local
   export SMOKE_VIDEOS=1                   # enable Wan adapter
   export VIDEOS_WAN_FIXTURE_ONLY=0        # ensure real pipeline is used
   export VIDEOS_WAN_MODEL="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
   export VIDEOS_WAN_DEVICE_PRESET="a100-80gb"   # or set VIDEOS_WAN_DEVICE_MAP JSON
   ```

   Optional knobs:
   - `VIDEOS_WAN_CACHE_TTL` — seconds to keep the Wan weights warm inside `gpu_utils.model_context`.
   - `VIDEOS_WAN_DEVICE` — explicit torch device (defaults to `cuda`).

3. **Run the focused suites** to exercise both the adapter and entrypoint:

   ```bash
   pytest -q \
     tests/unit/test_videos_wan_adapter.py \
     tests/unit/test_videos_wan_models.py \
     tests/unit/test_entrypoints_parametrized.py \
     tests/smoke/test_function_tools_smoke.py -k videos_wan
   ```

Troubleshooting mirrors SDXL:
- A `503` with `{"detail":"gpu busy"}` means another process has the GPU lock; wait for it to finish.
- Missing WAN weights or incompatible wheels fall back to the deterministic fixture, so the tests still pass but no GPU work occurs (check logs for the fallback notice).

## A2A integration roadmap
Sparkle Motion currently orchestrates every FunctionTool from a single `production_agent`, so peer-to-peer agent messaging is unnecessary and we have not wired up the A2A protocol yet. Once we invite external or independently hosted agents, A2A becomes attractive because it formalizes discovery, capability negotiation, and transport between heterogeneous agent runtimes.

Why adopt A2A later?
- Third-party or remote teams can plug their agents into the pipeline without us baking their SDKs directly into `production_agent`.
- We can scale individual agents independently and still reuse the same message contract for telemetry + error propagation.
- A2A helps enforce a clean “tools over the wire” boundary, which is useful once runs span multiple services or data centers.

High-level adoption plan
1. Introduce an A2A server (or client) shim inside `production_agent` so plan steps map to A2A invocations instead of direct FunctionTool calls.
2. Wrap the existing adapters (images, videos, TTS, QA, assemble, lipsync) with lightweight A2A endpoints that expose the same payload schema they already validate today.
3. Extend the artifact/telemetry hooks to include A2A trace IDs so cross-agent debugging remains intact, and document the contract for external partners.

## Release notes
- **2025-12-01 — Agent naming cleanup.** Only `script_agent` and `production_agent` remain ADK agents. All other runtime components have been renamed to FunctionTools or stage modules (for example, `images_stage` orchestrates the `images_sdxl` FunctionTool). Update any local configs or notebooks that previously referenced `*_agent` identifiers to use the new names listed in `docs/ARCHITECTURE.md#_agent-naming-matrix`.

## Running the Colab notebook (`notebooks/sparkle_motion.ipynb`)
Follow these steps to run the full notebook workflow inside Google Colab with a GPU runtime. The steps map directly to the notebook sections so you can keep track of where you are.

1. **Open Colab + set runtime.** Upload or clone this repo into Drive, open `notebooks/sparkle_motion.ipynb` in Colab, and switch the runtime to GPU (Runtime → Change runtime type → GPU).
2. **Configure workspace inputs (Cell 1).** Edit the `WORKSPACE_NAME`, `HF_MODELS`, `DRY_RUN`, and `MOUNT_POINT` values to match the Drive folder you want to use. This cell also prints the repo root Colab detects.
3. **Load secrets (Cell 2).** Run the `.env` loader so ADK variables (`ADK_PROJECT`, `ADK_API_KEY`, etc.) are available to the Workflow Agent helpers. The cell auto-installs `python-dotenv` if needed and scans common locations (`.env.local`, `/content/.env`, Drive workspace).
4. **Mount Google Drive (Cell 3).** The cell detects Colab, mounts Drive under `MOUNT_POINT`, and ensures `MyDrive/WORKSPACE_NAME/` exists. Skip manually mounting in the sidebar—the cell handles everything.
5. **Install ML deps (Cell 4).** Installs the packages listed in `requirements-ml.txt`. This is optional but recommended for GPU smoke tests. Outside Colab the cell prints the equivalent pip command for local shells.
6. **Run the Colab preflight helper (Cell 5).** Executes `sparkle_motion.notebook_preflight` to confirm Drive is mounted, `/ready` endpoints respond, required env vars are present, and GPU checks pass. Fix any failures before moving on.
7. **Prepare Drive workspace + download models (Cells 6–7).**
   - Cell 6 pins `REPO_ROOT` if you opened Colab from a different path.
   - Cell 7 invokes `scripts/colab_drive_setup.py` to create the workspace folders, optionally download Hugging Face weights listed in `HF_MODELS`, and emit a smoke JSON under `outputs/colab_smoke.json`. Use Cell 7b to inspect that smoke artifact.
8. **Launch Workflow Agent servers ("Workflow Agent server controls").** Use the start/stop widgets to spawn `script_agent` (port 8101) and `production_agent` (port 8200) via uvicorn inside Colab. Logs land in `tmp/script_agent.log` and `tmp/production_agent.log`. Skip this if you already run the servers elsewhere—just leave the widgets alone.
9. **Quickstart control panel (Cells 8–11).**
   - Cell 8 makes sure `REPO_ROOT/src` is on `sys.path`.
   - Cell 9 imports and displays the ipywidgets control panel wired to the `local-colab` profile described in `configs/tool_registry.yaml`.
   - Cell 10 syncs a known run ID into the control panel.
   - Cell 11 demonstrates starting a production run via raw HTTP (`/invoke`) and polling `/status`, updating the control panel with the new run ID when finished.
10. **Advanced control + artifacts tooling (Cells 12–19).**
    - The advanced control panel cell shows how to instantiate `ControlPanel` manually with custom profiles/timeouts.
    - The final deliverable helper (Cells 13–14) fetches the `qa_publish` manifest, embeds QA badges, downloads the final MP4, and optionally triggers a Colab download prompt.
    - The artifacts viewer (Cells 15–18) refreshes `/artifacts` responses for any run/stage, supports auto-refresh, and renders inline previews when local paths exist.
11. **Optional smoke + notebook helpers (Cells 20+).**
    - Cell 19 runs `preview_helpers` to embed audio/video previews inline.
    - Cell 20 triggers a manual artifacts refresh and captures a JSON snapshot for logs.
    - Cell 21 runs the orchestrator `Runner` in pure Python (fixture) mode to ensure Drive folders are writable without touching Workflow Agent APIs.
12. **Shut down servers.** When your session ends, use the server-control widgets to stop `script_agent` and `production_agent`, releasing the GPU. If you started them outside Colab, stop those processes in the original terminal instead.

For more context on each helper cell, see `docs/NOTEBOOK_AGENT_INTEGRATION.md` (control panel behavior, QA manifest layouts) and `docs/TODO.md` (live checklist + validation notes). If anything differs from these instructions, re-sync the repo and re-open the notebook to pick up the latest workflow helpers.
