"""Entrypoint for the ScriptAgent FunctionTool.

This runtime instantiates a Google ADK `LlmAgent` eagerly on startup. This is
an explicit project requirement: if the ADK SDK or agent cannot be created the
process will raise and exit rather than silently falling back.
"""
from __future__ import annotations

import argparse
import logging
import os

from sparkle_motion.tool_entrypoint import create_app
import uvicorn


LOG = logging.getLogger("script_agent.entrypoint")


def _create_adk_agent() -> "LlmAgent":
    """Instantiate and return a Google ADK LlmAgent.

    This function raises a RuntimeError if the ADK packages are unavailable or
    if the agent cannot be constructed. The model can be configured with the
    `SCRIPT_AGENT_MODEL` environment variable (defaults to a common Gemini
    model id).
    """
    try:
        # ADK classes (must be present in the environment)
        from google.adk.agents import LlmAgent  # type: ignore
        from google.adk.models.google_llm import Gemini  # type: ignore
    except Exception as e:  # import failure should be fatal per directive
        raise RuntimeError("Failed to import google.adk SDK required for LlmAgent") from e

    model_name = os.environ.get("SCRIPT_AGENT_MODEL", "gemini-2.5-flash-lite")
    try:
        model = Gemini(model=model_name)
    except Exception as e:  # ensure we fail loudly if model construction fails
        raise RuntimeError(f"Failed to construct Gemini model wrapper for '{model_name}'") from e

    try:
        agent = LlmAgent(name="script_agent", model=model)
    except Exception as e:
        raise RuntimeError("Failed to instantiate LlmAgent") from e

    return agent


def invoke_handler(body: dict, *, agent=None) -> dict:
    # Keep a simple echo + attach a note that the ADK agent is available.
    # The agent instance is stored on the app and can be used by more
    # sophisticated handlers if desired.
    info = {"echo": body, "note": "ScriptAgent running with ADK LlmAgent."}
    if agent is not None:
        info["agent_attached"] = True
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5001)
    args = p.parse_args()

    # Eagerly create the ADK agent; failure is fatal as requested.
    agent = _create_adk_agent()

    # Create the FastAPI app and attach the agent for handlers to use.
    app = create_app("script_agent", lambda body: invoke_handler(body, agent=agent))
    # attach for inspection/testability
    app.state.agent = agent

    LOG.info("Starting ScriptAgent (ADK LlmAgent) on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
