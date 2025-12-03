# Sparkle Motion Movie Machine

Sparkle Motion is an AI-powered video generation system built on top of the ADK. It transforms text prompts into fully rendered short films by orchestrating LLM-driven planning, image/video synthesis, text-to-speech, lip-syncing, and final assembly.

## Getting started 

This repository is meant to run from a single environment: **Google Colab with an A100 GPU runtime**. Everything else (local shells, other GPUs, CI) is out of scope for this README. Follow the steps below to clone the repo inside Colab, bring up the notebook helpers, run a plan, and download the final video.

> For expanded background, see `docs/`

## Prerequisites

- Google account with access to Colab Pro/Pro+ and an A100 runtime quota.
- ADK credentials stored in your Google Drive workspace (`ADK_PROJECT`, `ADK_API_KEY`, `HF_KEY`).
- Hugging Face token (if any private models are listed in `HF_MODELS`).
- Enough Drive space for model weights (~100 GB for SDXL + Wan fixtures) plus run artifacts.

## One-Time Drive Workspace Setup

1. Create (or reuse) a Drive folder, e.g. `MyDrive/sparkle-motion-workspace/`.
2. Place your `.env` or `.sparkle.env` file inside that folder with the ADK variables mentioned above.
3. Optional: pre-populate the folder with Hugging Face model snapshots if you want to skip downloads during the session.

## Launching the Colab Notebook

1. Open <https://colab.research.google.com> → **File → Open Notebook → GitHub** → paste this repo URL (`https://github.com/ekkus93/sparkle_motion`).
2. Select `notebooks/sparkle_motion.ipynb` and switch the runtime to **GPU → A100**.
3. Execute the notebook cells in order. Every required action is already scripted inside the notebook—there is no need to open a Colab terminal.

### Cell-to-Cell Checklist

| Cell(s) | What happens | Notes |
| --- | --- | --- |
| 1. Workspace config | Set workspace name, repo URL, and Drive mount point. | Adjust `WORKSPACE_NAME` to match your Drive folder.
| 2. Secrets loader | Searches Drive + notebook directory for `.env` files and exports ADK vars. | Confirms `ADK_PROJECT`/`ADK_API_KEY` are present.
| 3. Drive mount | Mounts Google Drive programmatically under `MOUNT_POINT`. | Skips mounting when not on Colab.
| 4. Repo pull | Clones/updates `sparkle_motion` into `/content/sparkle_motion` and exports `SPARKLE_MOTION_REPO_ROOT`. | No manual git commands needed.
| 5. Workspace config | Derives `REPO_ROOT`, extends `sys.path`, and prints detected paths. | Downstream cells rely on this.
| 6. Requirements install | Installs `requirements-ml.txt` (ipywidgets, diffusers, torch) inside Colab. | Safe to rerun; idempotent.
| 7–8. Drive helper + smoke | Runs `scripts/colab_drive_setup.py` to create workspace dirs and optionally download models. | Produces `outputs/colab_smoke.json` for verification.
| 9. Server controls | ipywidgets to start/stop `script_agent` (port 8101) and `production_agent` (port 8200) via uvicorn inside Colab. | Keep them running for the remaining cells.
| 10–12. Control panel | Launches the production control panel (plan loader, run controls, status polling) and syncs run IDs. | Use this UI to run plans end-to-end.
| 13. Production helper | Makes a Test_Film request directly via HTTP for reference and updates the control panel with the new run ID. | Useful sanity check before custom plans.
| 14. Final deliverable helper | Reads the `finalize` stage manifest, previews the `video_final` artifact inline, and offers a Colab download. | Requires a completed run.
| 15–18. Artifacts viewer | Polls `/artifacts` for any run/stage, renders per-stage summaries, and previews images/audio/video inline. | Handy for troubleshooting.
| Remaining cells | Preview helpers, manual artifacts refresh, optional pure-Python runner. | Use only if instructed; otherwise stop after downloading artifacts.

## Running a Production Plan

1. Start both agents with the server-control widgets (Cell group 9). Wait for the status indicator to show **running**.
2. In the control panel (Cell group 10), choose **Dry-Run** to validate the workflow, or **Run** for the real pipeline. QA automation is disabled, so every run follows the same finalize-first path regardless of earlier toggle settings.
3. Click **Generate Plan** (or load an existing plan JSON) and inspect the summary widget.
4. Click **Execute Production**. The panel streams `/status` updates automatically; you can also enable auto-polling to keep the log fresh.
5. Once the run reaches the `finalize` stage, use the **Final Deliverable Helper** (Cell 14) to preview and download the MP4, then capture any manual review notes outside the notebook.
6. Use the **Artifacts Viewer** to inspect intermediate manifests if finalize reveals a gap or you need stage-by-stage evidence for manual review.

## Shutting Down the Session

1. Stop both agents via the server-control widgets.
2. Unmount Drive by clicking the stop button in the Drive cell (if still mounted) or leave it for Colab to clean up.
3. Close the Colab tab; artifacts remain in your Drive workspace under `artifacts/runs/<run_id>/`.

## Troubleshooting (Colab-only)

- **GPU busy / 503 errors**: Another cell is still running a FunctionTool. Wait for the previous request to finish or click **Stop** in the server controls, then start the agents again.
- **Missing models**: Rerun the Drive helper cell to download the required HF models. Check `outputs/colab_smoke.json` for a list of downloaded assets.
- **Env vars not found**: Ensure your `.env` file lives inside the Drive workspace root and that the secrets loader cell points to the correct directory.
- **Notebook path issues**: Always rerun the repo pull + workspace config cells after restarting the Colab runtime so `SPARKLE_MOTION_REPO_ROOT` is defined.
- **Large downloads**: SDXL and Wan weights are big. Keep the Colab tab active during the initial download; repeats are faster thanks to Drive caching.

That’s the entire workflow: open the notebook on Colab A100, run the cells in order, watch the plan execute, and download your final video. For any non-Colab use cases, see `docs/OPERATIONS_GUIDE.md` instead of this README.
