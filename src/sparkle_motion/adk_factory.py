from __future__ import annotations

import os
from typing import Any, Optional

from . import adk_helpers
from . import observability


class _DummyAgent:
    def __init__(self, name: str, model_spec: Optional[str] = None) -> None:
        self.name = name
        self.model_spec = model_spec
        self._closed = False

    def info(self) -> dict:
        return {"name": self.name, "model_spec": self.model_spec}

    def close(self) -> None:
        self._closed = True


def _fixture_agent(tool_name: str, model_spec: Optional[str] = None) -> _DummyAgent:
    return _DummyAgent(tool_name, model_spec=model_spec)


def get_agent(tool_name: str, model_spec: Optional[str] = None, mode: str = "per-tool", seed: Optional[int] = None) -> Any:
    """Return an agent for the given tool.

    Behavior:
    - If `ADK_USE_FIXTURE=1` returns a lightweight dummy agent for tests.
    - Otherwise probes the real ADK SDK and attempts to construct an agent.
    - Raises RuntimeError with a clear message on failure (fail‑loud semantics).
    """
    # Fixture/test mode — no real SDK required
    if os.environ.get("ADK_USE_FIXTURE") == "1":
        agent = _fixture_agent(tool_name, model_spec=model_spec)
        # record seed and emit a telemetry event for tests
        try:
            observability.record_seed(seed, tool_name=tool_name)
            observability.emit_agent_event("agent.created", {"tool": tool_name, "model_spec": model_spec, "fixture": True, "seed": seed})
        except Exception:
            pass
        # attach seed to the agent object for test introspection
        try:
            setattr(agent, "seed", seed)
        except Exception:
            pass
        return agent

    # Probe SDK (adk_helpers.probe_sdk raises SystemExit when SDK import fails)
    try:
        adk_mod, client = adk_helpers.probe_sdk()
    except SystemExit as e:
        raise RuntimeError("google.adk SDK not available or import failed") from e

    # Try common SDK agent constructors in a best-effort order
    candidates = [
        getattr(adk_mod, "LlmAgent", None),
        getattr(adk_mod, "Llm", None),
        getattr(adk_mod, "Agent", None),
        getattr(adk_mod, "create_agent", None),
        getattr(adk_mod, "create_llm_agent", None),
    ]

    for cand in [c for c in candidates if c is not None]:
        try:
            # If cand is a callable class/constructor, try a few calling conventions
            if callable(cand):
                try:
                    return cand(model=model_spec)  # common pattern
                except TypeError:
                    try:
                        return cand(model_spec)
                    except TypeError:
                        try:
                            return cand(name=tool_name, model=model_spec)
                        except TypeError:
                            # last-resort: call without args
                            return cand()
        except Exception:
            continue

    # As a last attempt try to use a client helper if present
    client_ctor = None
    if client is not None:
        client_ctor = getattr(client, "create", None) or getattr(client, "open", None)
    if client_ctor and callable(client_ctor):
        try:
            agent = client_ctor(tool_name, model_spec)
            try:
                # record seed and emit telemetry for created agent
                observability.record_seed(seed, tool_name=tool_name)
                observability.emit_agent_event("agent.created", {"tool": tool_name, "model_spec": model_spec, "fixture": False, "seed": seed})
            except Exception:
                pass
            try:
                setattr(agent, "seed", seed)
            except Exception:
                pass
            return agent
        except Exception:
            pass

    # If we reached here, we may still have constructed an agent via the
    # earlier candidate loops; otherwise fail loudly.
    raise RuntimeError("Unable to construct an ADK agent: no known constructor discovered in google.adk")


def create_agent(*args, **kwargs) -> Any:
    """Alias for get_agent to match requested API surface."""
    return get_agent(*args, **kwargs)


def close_agent(agent: Any) -> None:
    """Attempt to cleanly close/shutdown the provided agent instance.

    This will try several common teardown method names and ignore errors.
    """
    if agent is None:
        return

    for name in ("close", "shutdown", "stop", "disconnect", "dispose"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                # Best-effort; do not mask caller errors
                pass
