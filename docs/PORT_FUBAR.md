# PORT_FUBAR.md — port assignments and callers

This ledger tracks every HTTP port hard-coded or implied inside the repo so we can see **who binds what** and **who calls whom** before we refactor anything. Sources cite the file that sets the port, followed by callers that bake the corresponding URL.

## TL;DR table

| Port | Service / Purpose | Callee config (where it binds) | Caller overrides (how clients change) |
| --- | --- | --- | --- |
| 5001 | `script_agent` plan generator | `configs/tool_registry.yaml` → `script_agent.invoke`, `function_tools/script_agent/metadata.json` (ADK mirror). `scripts/run_function_tool.py` reads these values when uvicorn starts. | Most callers auto-resolve via `ToolRegistry`. Manual override: export `SCRIPT_AGENT_BASE=http://host:port` before launching `notebooks/control_panel.py` or other legacy scripts. |
| 5008 | `production_agent` workflow runner | `configs/tool_registry.yaml` → `production_agent.invoke`. No separate metadata file. | `PanelEndpoints.production_*` and notebooks use registry defaults; override with `PRODUCTION_AGENT_BASE=http://host:port` when targeting a remote instance. |
| 5002 | `images_sdxl` FunctionTool | `configs/tool_registry.yaml`, `function_tools/images_sdxl/metadata.json`. Runner CLI can still pass `--port`. | No direct callers besides `production_agent`. To run standalone, invoke `scripts/run_function_tool.py images_sdxl --port <new>` or edit the registry + metadata pair so future launches inherit the change. |
| 5003 | `videos_wan` FunctionTool | Same as above: registry entry plus `function_tools/videos_wan/metadata.json`. | Same as above; only ToolRegistry consumers touch it, but `scripts/run_function_tool.py videos_wan --port` overrides per session. |
| 5004 | `tts_chatterbox` FunctionTool | Registry entry + `function_tools/tts_chatterbox/metadata.json`. | Same operational story as 5002–5003. |
| 5005 | `lipsync_wav2lip` FunctionTool | Registry entry + `function_tools/lipsync_wav2lip/metadata.json`. | Same operational story as 5002–5003. |
| 5006 | `assemble_ffmpeg` FunctionTool | Registry entry + `function_tools/assemble_ffmpeg/metadata.json`. | Same operational story as 5002–5003. |
| 7077 | Filesystem ArtifactService shim (`filesystem_artifacts`) | Defaulted in `src/sparkle_motion/filesystem_artifacts/config.py` (`DEFAULT_ARTIFACTS_FS_BASE_URL`) and `src/sparkle_motion/filesystem_artifacts/cli.py` (`serve --port`). Notebook helpers inherit the same constant. | Override globally with `ARTIFACTS_FS_BASE_URL=http://host:port` or via CLI flags when launching the shim (`scripts/filesystem_artifacts.py serve --port 7450`). Control panel + tests honor the env var. |

> Everything else (CLI tools, Control Panel, tests) obtains host/port values by reading `ToolRegistry` or env vars, so the table above captures all authoritative bindings.

## Detailed notes by surface

### ToolRegistry-backed agents (ports 5001–5008)

- **Source files**: `configs/tool_registry.yaml` is the canonical list for `script_agent`, `production_agent`, and all FunctionTools. Each FunctionTool also repeats its port inside `function_tools/<tool>/metadata.json` (used by ADK/fixtures) so both must stay in sync.
- **Changing the callee**: edit the registry entry (`invoke.host` / `invoke.port` / `ready.path`) and keep the matching `metadata.json` file in lock-step. The next `scripts/run_function_tool.py <tool>` invocation will bind to the new port automatically.
- **Changing a caller**: notebooks and the control panel call `sparkle_motion.tool_registry.get_local_endpoint_info()` so they pick up registry edits immediately. For ad-hoc overrides, export `SCRIPT_AGENT_BASE` or `PRODUCTION_AGENT_BASE` before launching notebooks/tests to point at remote instances.
- **Docs/tests**: `docs/archived/HowToRunLocalTools.md` still shows manual uvicorn commands using 5001+. `artifacts/function_tools_inventory.md` duplicates the registry for auditing.

### Downstream FunctionTools (5002–5006) callers

- No first-class notebooks call these ports directly. Instead, `production_agent` proxies every stage through ADK Workflow definitions. The stage configs (`function_tools/*/workflows/*.yaml` under `artifacts/prompt_templates` and `configs/root_agent.yaml`) reference each tool **by name**; the actual host/port resolution again flows through ToolRegistry at runtime.
- The only places the raw ports appear are `tool_registry.yaml` and each tool’s `metadata.json` file. Runner scripts (`scripts/run_function_tool.py`, `notebooks/sparkle_motion.ipynb` server control cell) resolve them dynamically.
- Tests rely on in-process FastAPI apps rather than raw TCP ports, so there are no additional hard-coded numbers for these services.

### Filesystem ArtifactService shim (port 7077)

- **Binding defaults**:
	- `src/sparkle_motion/filesystem_artifacts/config.py` exposes `DEFAULT_ARTIFACTS_FS_BASE_URL = "http://127.0.0.1:7077"` and every consumer imports that constant.
	- `src/sparkle_motion/filesystem_artifacts/cli.py` sets `serve --host 127.0.0.1 --port 7077` by default.
	- `notebooks/sparkle_motion.ipynb` and `notebooks/control_panel.py` both rely on `ARTIFACTS_FS_BASE_URL` (or the same constant) for widget defaults.
- **Changing the callee**: start the shim with `scripts/filesystem_artifacts.py serve --host 0.0.0.0 --port 7450` (or similar). For permanent changes, edit the constant in `config.py`.
- **Changing callers**: set `ARTIFACTS_FS_BASE_URL=http://host:port` (and optionally `FILESYSTEM_SHIM_PORT`) before launching notebooks, the CLI, or tests; everything reads from that env var and requires no code edit.
- **Callers**: control panel filesystem buttons, notebook shim helper, and retention tests all hit the configured base URL (`/healthz`, `/artifacts`, `/cleanup` endpoints).

## Outstanding cleanup items

1. Consider generating metadata from ToolRegistry to avoid having the same port duplicated in `function_tools/*/metadata.json`.

With this map in hand we can safely centralize port definitions or migrate to service discovery without losing track of any caller that still embeds a literal URL.
