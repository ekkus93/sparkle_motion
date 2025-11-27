# How to Run Local FunctionTools (local-colab / dev)

This document explains how to run FunctionTool ASGI entrypoints locally (in-process or via `uvicorn`) and how to run the unit and integration tests that exercise the real entrypoint.

Note: the repository uses a conda environment named `sparkle_motion` for development. Follow the steps below inside that environment.

Prerequisites
- Activate the conda env you use for this project:

```bash
conda activate sparkle_motion
```

- Install development dependencies (only once, inside the active env):

```bash
pip install -r requirements-dev.txt
```

Run the real ASGI entrypoint (script_agent example)

- Recommended: run the ASGI app using the conda env Python and set `PYTHONPATH=src` so the package is importable.

```bash
PYTHONPATH=src ADK_USE_FIXTURE=1 ARTIFACTS_DIR=artifacts/test DETERMINISTIC=1 \
  python -m uvicorn \
  sparkle_motion.function_tools.script_agent.entrypoint:app --host 127.0.0.1 --port 5002
```

- Alternative: start from the repo root using `uvicorn` on the active Python in your PATH:

```bash
PYTHONPATH=src ADK_USE_FIXTURE=1 ARTIFACTS_DIR=artifacts/test DETERMINISTIC=1 \
  python -m uvicorn sparkle_motion.function_tools.script_agent.entrypoint:app --host 127.0.0.1 --port 5002
```

Key environment variables
- `ADK_USE_FIXTURE`: defaults to `1` in these docs. When `1`, the entrypoint will persist artifacts to the local filesystem (a `file://` URI). When set to `0`, the code will attempt to publish to ADK/`adk` CLI or SDK if available.
- `ARTIFACTS_DIR`: directory where artifact files are written when fixture mode is enabled. Example: `artifacts/test`.
- `DETERMINISTIC`: when `1` creates deterministic filenames useful for tests; when `0` adds randomness/pid to filenames.

Using the repository runner
- `scripts/run_function_tool.py` provides a lightweight runner. By default it creates a simple echo app for many tools. To start the real `script_agent` app you can either run the `uvicorn` command above, or run the runner and pass `--host`/`--port` to avoid relying on the registry:

```bash
PYTHONPATH=src ADK_USE_FIXTURE=1 ARTIFACTS_DIR=artifacts/test DETERMINISTIC=1 \
  python scripts/run_function_tool.py --tool script_agent --host 127.0.0.1 --port 5001
```

Running tests
- Unit test (fast, imports the app directly):

```bash
# run the unit test we added
PYTHONPATH=src python -m pytest -q tests/test_script_agent_invoke.py::test_invoke_persists_artifact
```

- Integration test (starts uvicorn and exercises the real entrypoint):

```bash
PYTHONPATH=src python -m pytest -q tests/test_script_agent_integration.py
```

Notes and troubleshooting
- If `fastapi` or `uvicorn` import fails, ensure you installed `requirements-dev.txt` into the `sparkle_motion` environment and that you are invoking the env's Python (activate the conda env and use `python -m uvicorn` / `python -m pytest` so you don't need to hard-code an interpreter path).
- To stop any running uvicorn server started in the background: find the process and terminate it (`ps aux | grep uvicorn` / `pkill -f uvicorn`).
- Tests are written to use `ADK_USE_FIXTURE=1` to avoid requiring ADK credentials; integration tests spawn `uvicorn` using `sys.executable` so they run in the same Python environment pytest was invoked from.

If you want, I can:
- Add a small `scripts/stop_local_tools.sh` helper to safely stop uvicorn processes started for local testing.
- Make `scripts/run_function_tool.py` optionally start the real entrypoint (e.g., `--real-entrypoint`) and update its docs.
