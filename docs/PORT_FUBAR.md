# PORT_FUBAR.md — port assignments and callers

This ledger tracks every HTTP port hard-coded or implied inside the repo so we can see **who binds what** and **who calls whom** before we refactor anything. Sources cite the file that sets the port, followed by callers that bake the corresponding URL.

## TL;DR table

| Port | Service / Purpose | Source of truth | Known callers (direct port refs) |
| --- | --- | --- | --- |
| 5001 | `script_agent` plan generator | `configs/tool_registry.yaml`, `function_tools/script_agent/metadata.json` | `notebooks/control_panel.py` (via `PanelEndpoints`), `notebooks/sparkle_motion.ipynb` server widget, `docs/archived/HowToRunLocalTools.md`, `artifacts/function_tools_inventory.md` |
| 5008 | `production_agent` workflow runner | `configs/tool_registry.yaml` | same notebook widgets as above, `PanelEndpoints.production_*` helpers, `README.md` (legacy 8200 doc) |
| 5002 | `images_sdxl` FunctionTool | `configs/tool_registry.yaml`, `function_tools/images_sdxl/metadata.json` | Tool runners via `ToolRegistry` (`scripts/run_function_tool.py`), stage definitions inside `production_agent` |
| 5003 | `videos_wan` FunctionTool | `configs/tool_registry.yaml`, `function_tools/videos_wan/metadata.json` | Same pattern as 5002 |
| 5004 | `tts_chatterbox` FunctionTool | `configs/tool_registry.yaml`, `function_tools/tts_chatterbox/metadata.json` | Same pattern as 5002 |
| 5005 | `lipsync_wav2lip` FunctionTool | `configs/tool_registry.yaml`, `function_tools/lipsync_wav2lip/metadata.json` | Same pattern as 5002 |
| 5006 | `assemble_ffmpeg` FunctionTool | `configs/tool_registry.yaml`, `function_tools/assemble_ffmpeg/metadata.json` | Same pattern as 5002 |
| 7077 | Filesystem ArtifactService shim (`filesystem_artifacts`) | `src/sparkle_motion/filesystem_artifacts/config.py`, CLI defaults in `src/sparkle_motion/filesystem_artifacts/cli.py`, notebook widget defaults | `notebooks/control_panel.py`, `notebooks/sparkle_motion.ipynb` shim controls, `tests/unit/test_filesystem_artifacts_app.py`, retention tests |
| 8200 | QA stub server + legacy production_agent docs | `tmp/qa_mode_stub_server.py`, `.env_sample`, `README.md` | Notebook README table, `.env_sample` overrides, any manual QA-mode workflows |
| 8101 | Legacy script_agent doc/env reference | `.env_sample`, `README.md` | Only documentation/env; no runtime binds |

> Everything else (CLI tools, Control Panel, tests) obtains host/port values by reading `ToolRegistry` or env vars, so the table above captures all authoritative bindings.

## Detailed notes by surface

### ToolRegistry-backed agents (ports 5001–5008)

- **Source files**: `configs/tool_registry.yaml` is the canonical list for `script_agent`, `production_agent`, and all FunctionTools. Each FunctionTool also repeats its port inside `function_tools/<tool>/metadata.json` (used by ADK/fixtures) so both must stay in sync.
- **Launchers**: `scripts/run_function_tool.py` parses the registry via `sparkle_motion.tool_registry.get_local_endpoint` and falls back to CLI `--host/--port`. The uvicorn invocation inherits whatever the registry says.
- **Notebook callers**:
	- `notebooks/control_panel.py` constructs `PanelEndpoints` by calling `_require_endpoint("script_agent")` and `_require_endpoint("production_agent")`. Those methods strip `/invoke` to derive `/ready`, `/status`, `/artifacts`, and `/control/*` URLs; all assume the registry-provided base (5001 / 5008 today).
	- `notebooks/sparkle_motion.ipynb` server widget shells out to `scripts/run_function_tool.py` for startup, then performs `/ready` health checks using the same registry-derived endpoints.
- **Docs/tests**: `docs/archived/HowToRunLocalTools.md` still mentions manual uvicorn invocations for 5001+ (`script_agent` example uses 5001, older snippet mentions 5002). `artifacts/function_tools_inventory.md` lists the same ports for auditing purposes.
- **Legacy drift**: `.env_sample` and the README table still advertise `script_agent` on 8101 and `production_agent` on 8200. These values are no longer consumed by the control panel (it ignores the env overrides once ToolRegistry is present) but they continue to confuse new users and should be updated or dropped.

### Downstream FunctionTools (5002–5006) callers

- No first-class notebooks call these ports directly. Instead, `production_agent` proxies every stage through ADK Workflow definitions. The stage configs (`function_tools/*/workflows/*.yaml` under `artifacts/prompt_templates` and `configs/root_agent.yaml`) reference each tool **by name**; the actual host/port resolution again flows through ToolRegistry at runtime.
- The only places the raw ports appear are `tool_registry.yaml` and each tool’s `metadata.json` file. Runner scripts (`scripts/run_function_tool.py`, `notebooks/sparkle_motion.ipynb` server control cell) resolve them dynamically.
- Tests rely on in-process FastAPI apps rather than raw TCP ports, so there are no additional hard-coded numbers for these services.

### Filesystem ArtifactService shim (port 7077)

- **Binding defaults**:
	- `src/sparkle_motion/filesystem_artifacts/config.py` falls back to `http://127.0.0.1:7077` when `ARTIFACTS_FS_BASE_URL` is unset.
	- `src/sparkle_motion/filesystem_artifacts/cli.py` sets the `serve` subcommand default to `--port 7077` and mirrors that URL when generating env exports.
	- `notebooks/sparkle_motion.ipynb` defines `FILESYSTEM_SHIM_PORT = os.environ.get("FILESYSTEM_SHIM_PORT", "7077")` before invoking `scripts/filesystem_artifacts.py serve`.
	- `notebooks/control_panel.py` uses the same `DEFAULT_FS_BASE_URL = os.environ.get("ARTIFACTS_FS_BASE_URL", "http://127.0.0.1:7077")` for preview widgets and `/healthz` checks.
- **Callers**:
	- Control Panel filesystem health checks (`/healthz` + `/artifacts`) and the notebook shim helper both post to this base URL.
	- `tests/unit/test_filesystem_artifacts_app.py`, `tests/unit/test_filesystem_manifest_parity.py`, and `tests/unit/test_filesystem_retention.py` all construct configs with `base_url="http://127.0.0.1:7077"` so the test fixtures mirror production defaults.
	- Archived design docs (`docs/archived/filesystem_artifact_shim_design.md`) cite the same base.

### QA stub + legacy overrides (ports 8200 / 8101)

- `tmp/qa_mode_stub_server.py` exports `QA_MODE_SERVER_PORT` (default `8200`) and spins up a FastAPI app that mimics `production_agent` responses for manual QA-mode demos. Anyone running that server must still point their client to `http://127.0.0.1:8200/status|artifacts`.
- `.env_sample` keeps `PRODUCTION_AGENT_BASE=http://127.0.0.1:8200` and `SCRIPT_AGENT_BASE=http://127.0.0.1:8101`. Those were required before ToolRegistry existed; now they only affect older scripts that still read env overrides. Control Panel ignores them because it resolves endpoints from the registry, but they remain a documentation hazard.
- `README.md` (table in section “Notebook Flow overview”) repeats the obsolete 8101/8200 assignments. Until we edit the README/env sample, folks reading the docs will try to hit the wrong port.

## Outstanding cleanup items

1. Update `.env_sample` and `README.md` to match the ToolRegistry defaults (5001/5008) so newcomers stop launching on 8101/8200.
2. Decide whether `tmp/qa_mode_stub_server.py` should also look up ToolRegistry endpoints (or at least document that it still binds to 8200).
3. Consider generating metadata from ToolRegistry to avoid having the same port duplicated in `function_tools/*/metadata.json`.

With this map in hand we can safely centralize port definitions or migrate to service discovery without losing track of any caller that still embeds a literal URL.
