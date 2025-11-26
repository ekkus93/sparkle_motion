"""Run a lightweight FastAPI FunctionTool server for local-colab development.

This runner prefers endpoints declared in `configs/tool_registry.yaml` and
exposes a minimal `/healthz` and `/invoke` HTTP API. Use in Colab or locally
when Docker is unavailable.

Examples:
  PYTHONPATH=src python scripts/run_function_tool.py --tool script_agent
  PYTHONPATH=src python scripts/run_function_tool.py --tool script_agent --port 5001
"""
from __future__ import annotations

import argparse
import json
import logging
from urllib.parse import urlparse

from typing import Optional

from fastapi import FastAPI, Request
import uvicorn
from sparkle_motion.tool_entrypoint import create_app

from sparkle_motion.tool_registry import get_local_endpoint


LOG = logging.getLogger("run_function_tool")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tool", required=True, help="tool id, e.g. script_agent")
    p.add_argument("--host", default=None, help="host to bind (overrides registry)")
    p.add_argument("--port", type=int, default=None, help="port to bind (overrides registry)")
    p.add_argument("--profile", default="local-colab", help="tool registry profile to consult")
    return p.parse_args()


def resolve_host_port(tool: str, port: Optional[int], host: Optional[str], profile: str) -> tuple[str, int, str]:
    """Resolve host, port and invoke-path from registry or args.

    Returns: (host, port, invoke_path)
    """
    if host and port:
        return host, port, "/invoke"

    ep = get_local_endpoint(tool, profile=profile)
    if not ep:
        raise RuntimeError(f"No endpoint for tool '{tool}' in profile '{profile}' and no host/port provided")
    parsed = urlparse(ep)
    bind_host = parsed.hostname or "127.0.0.1"
    bind_port = parsed.port or 80
    invoke_path = parsed.path or "/invoke"
    # allow individual overrides
    if host:
        bind_host = host
    if port:
        bind_port = port
    return bind_host, bind_port, invoke_path


def make_app(tool_id: str) -> FastAPI:
    """Create an app using the shared `create_app` helper.

    This preserves the minimal echo behavior used by the runner: the
    invoke handler returns the received body under `received`.
    """
    def invoke_fn(body: dict):
        return {"received": body}

    return create_app(tool_id, invoke_fn)


def main() -> None:
    args = parse_args()
    host, port, invoke_path = resolve_host_port(args.tool, args.port, args.host, args.profile)
    LOG.info("Starting tool %s on %s:%s (invoke path: %s)", args.tool, host, port, invoke_path)
    app = make_app(args.tool)
    # uvicorn will mount at root; our invoke handler is at /invoke
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
