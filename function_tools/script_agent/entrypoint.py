"""Example entrypoint for the ScriptAgent FunctionTool using the shared helper.

This is a lightweight example intended for local-colab use. It exposes
`/healthz` and `/invoke` endpoints and demonstrates a simple echo-style
implementation that real tools should replace with model calls.
"""
from __future__ import annotations

import argparse
import logging

from sparkle_motion.tool_entrypoint import create_app
import uvicorn


LOG = logging.getLogger("script_agent.entrypoint")


def invoke_handler(body: dict) -> dict:
    # A tiny example implementation: echo the input and add a small note.
    return {"echo": body, "note": "This is a local ScriptAgent stub."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5001)
    args = p.parse_args()

    app = create_app("script_agent", invoke_handler)
    LOG.info("Starting ScriptAgent on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
