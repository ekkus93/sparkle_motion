# Notebook + Control Panel Integration Guide

This guide explains how to operate Sparkle Motion from Google Colab or a local
Jupyter runtime. It mirrors the finalize-only pipeline implemented in
`notebooks/sparkle_motion.ipynb` and keeps notebook users aligned with the CLI.

## 1. Environment recap

1. Start the local FunctionTools and WorkflowAgent (see `docs/HowToRunLocalTools.md`).
2. Launch the notebook on the same machine (or Colab connected notebook) and set
   `PYTHONPATH=src` so `notebooks.control_panel` can be imported.
3. Export the usual helper variables before opening the notebook session:

```bash
export ADK_USE_FIXTURE=1
export ARTIFACTS_DIR="$REPO/artifacts"
export CONTROL_PANEL_LOG=1
```

The notebook assumes the workflow is exposed via `tool_registry.get_local_endpoint`
for the `local-colab` profile. Update `configs/tool_registry.yaml` if you move
ports or rename hosts.

## 2. Control panel quick start

Add the following cell near the top of the notebook:

```python
from notebooks.control_panel import create_control_panel
panel = create_control_panel()
panel
```

The panel wires together:

- Prompt + title inputs for plan generation.
- Plan URI loader (paste an existing plan artifact and cache it locally).
- Generate / Run / Pause / Resume / Stop buttons mapped to `/invoke` and
  `/control/*` endpoints.
- Status polling toggle (streams `/ready` + `/status`).
- Artifacts viewer tied to `/artifacts` with stage filtering.

### Typical workflow

1. Fill in **Title** and **Prompt**, then click **Generate Plan**.
2. Review the JSON summary in the plan details pane; adjust the plan if needed.
3. Choose `Dry-Run` or `Run` in the **Mode** dropdown and click **Run Production**.
4. Enable **Poll Status** once `/status` reports ready to watch progress.
5. Use **Refresh Artifacts** (or auto-fetch) to list stage manifests.

The control panel automatically caches the latest `run_id`, threads run IDs into
status/artifact calls, and prints backend health information in the sidebar.

## 3. Artifacts viewer and final deliverable helper

The notebook includes a dedicated cell for browsing manifest entries:

```python
from notebooks import preview_helpers
preview_helpers.render_stage_summary(stage_manifest)
preview_helpers.display_artifact_previews(stage_manifest)
```

Key behaviors:

- Filters by `run_id` (required) and optional `stage` text box.
- Displays per-stage summaries (artifact counts, media types, status).
- Embeds images/audio/video inline using ipywidgets for quick inspection.

For finalize, use the "final deliverable" helper cell near the bottom of the
notebook:

1. Set `RUN_ID` and run the helper cell.
2. The helper fetches `/artifacts?stage=finalize`, embeds the `video_final`
   preview inline, and surfaces download links (local path + Drive/ADK fallback).
3. If the manifest is missing, the helper prints instructions for rerunning with
   `resume_from="finalize"`.

## 4. Status & troubleshooting tips

- **Status toggle disabled**: the panel enables it after the first successful `/invoke`.
  Use the "Check Status" button to probe readiness manually.
- **Artifacts viewer empty**: ensure a run ID is set and the pipeline has published
  manifests. Use the control panel log output to confirm publish events.
- **Filesystem backend controls**: when `ARTIFACTS_BACKEND=filesystem` the panel
  shows a health button and log pane for the shim; otherwise it displays "Backend unset".
- **Widget layout tweaks**: the control panel exposes `.container`; re-arrange it
  in the notebook if you prefer a different layout.

## 5. Reference snippets

Load an existing plan JSON artifact within the notebook:

```python
from notebooks.control_panel import ControlPanel
panel = ControlPanel(auto_display=False)
panel.plan_uri_input.value = "file:///path/to/plan.json"
panel._handle_load_plan_payload(None)
panel.container  # display manually if needed
```

Fetch artifacts without the widgets (useful for scripted checks):

```python
import requests
from sparkle_motion import tool_registry

endpoints = ControlPanel.PanelEndpoints.from_registry()
response = requests.get(endpoints.production_artifacts, params={"run_id": "run_123"}, timeout=30)
response.raise_for_status()
print(response.json())
```

The notebook, CLI, and docs now describe a single finalize-first delivery workflow.
