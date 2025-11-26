# FunctionTools Inventory

Generated: 2025-11-26

This inventory was produced by scanning `configs/tool_registry.yaml`, the
`function_tools/` directory, and `src/sparkle_motion/function_tools/` for
available entrypoints and tests. The goal is to identify missing runtime
entrypoints and test coverage required to complete FunctionTool packaging.

Summary

| Tool ID | Declared in `configs/tool_registry.yaml` | `function_tools/<tool>/metadata.json` | Entrypoint (src or tool dir) | Tests present |
|---|---:|:---:|:---:|:---:|
| script_agent | yes | yes | `src/sparkle_motion/function_tools/script_agent/entrypoint.py` and `function_tools/script_agent/entrypoint.py` | yes (`tests/test_function_tools/test_script_agent_entrypoint*.py`) |
| images_sdxl | yes | yes | none found | no |
| videos_wan | yes | yes | none found | no |
| tts_chatterbox | yes | yes | none found | no |
| lipsync_wav2lip | yes | yes | none found | no |
| assemble_ffmpeg | yes | yes | none found | no |
| qa_qwen2vl | yes | yes | none found | no |

Notes and recommendations

- Metadata: Every declared tool has a `function_tools/<tool>/metadata.json` file. These metadata files already contain the minimal keys (name/package_name/version/endpoints/schemas) used by `scripts/register_tools.py`.

- Entrypoints: Only `script_agent` currently has a runtime FastAPI entrypoint implemented under `src/sparkle_motion/function_tools/script_agent/entrypoint.py` (and a mirror under `function_tools/script_agent/entrypoint.py`). All other tools lack a runnable entrypoint under `src/sparkle_motion/function_tools/<tool>/entrypoint.py` and therefore cannot be started using the repo runner (`scripts/run_function_tool.py`).

- Tests: The test suite includes focused tests for `script_agent`'s entrypoint and for the ADK fixture shims. There are no per-tool smoke tests for the other tools. Adding a small smoke test per tool (start runner, hit `/health`, call `/invoke` with a minimal payload, assert artifact created) will provide clear acceptance criteria for packaging.

- Next actions (recommended):
  1. Create canonical entrypoint scaffolding for each tool under `src/sparkle_motion/function_tools/<tool>/entrypoint.py` that exposes `/health`, `/ready`, `/metrics`, and `/invoke` and uses the same validation and deterministic artifact behavior as `script_agent`.
  2. Add minimal smoke tests under `tests/test_function_tools/test_<tool>_entrypoint.py` for each tool that exercise health and a deterministic `invoke` payload.
  3. Optionally add a simple `function_tools/<tool>/entrypoint.py` (dev Docker/packaging artifacts) if you want a per-tool runtime packaged in the `function_tools` folder as well.
  4. Re-run `scripts/register_tools.py --dry-run --metadata-dir=function_tools` to ensure metadata is discoverable and valid (the script already supports `--metadata-dir`).

If you'd like, I can start implementing step 1 for one tool (e.g., `images_sdxl`) and add the corresponding smoke test. Which tool should I scaffold first?
