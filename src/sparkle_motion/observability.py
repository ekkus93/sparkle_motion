from __future__ import annotations

import os
import time
from typing import Any, Optional

from . import adk_helpers, telemetry


def get_session_id() -> str:
    """Return a session identifier.

    Priority:
    - `ADK_SESSION_ID` env var
    - `ADK_RUN_ID` env var
    - fallback to generated local id
    """
    sid = os.environ.get("ADK_SESSION_ID") or os.environ.get("ADK_RUN_ID")
    if sid:
        return sid
    return f"local-{os.getpid()}-{int(time.time())}"


def record_seed(seed: Optional[int], session_id: Optional[str] = None, tool_name: Optional[str] = None) -> None:
    """Record a seed in MemoryService (if available) and emit a telemetry event.

    This is best-effort: when ADK is not present we use local in-memory
    services provided by `adk_helpers` (fixture-friendly). Tests can assert
    that this call emitted a telemetry event via `telemetry.get_events()`.
    """
    sid = session_id or get_session_id()
    try:
        svc = adk_helpers.get_memory_service()
        try:
            svc.store_session_metadata(sid, {"seed": seed, "tool": tool_name})
        except Exception:
            # Best-effort; fall back to telemetry only
            pass
    except Exception:
        # MemoryService not available — ignore
        pass

    try:
        telemetry.emit_event("observability.seed_recorded", {"session_id": sid, "seed": seed, "tool": tool_name})
    except Exception:
        pass


def emit_agent_event(name: str, payload: Optional[dict[str, Any]] = None) -> None:
    """Emit an agent lifecycle event.

    This writes an in-process telemetry event and, if the ADK SDK exposes a
    plausible timeline or events API, attempts to publish a corresponding
    SDK event. SDK emission is best-effort and will not raise on failure.
    """
    payload = payload or {}
    try:
        telemetry.emit_event(name, payload)
    except Exception:
        pass

    # Try to send to ADK timeline/events if SDK available
    try:
        res = adk_helpers.probe_sdk()
    except SystemExit:
        return
    except Exception:
        return
    if not res:
        return
    adk_mod, _ = res

    # common candidate APIs
    candidates = [
        getattr(adk_mod, "timeline", None),
        getattr(adk_mod, "events", None),
        getattr(adk_mod, "Timeline", None),
        getattr(adk_mod, "Events", None),
    ]
    for cand in [c for c in candidates if c is not None]:
        # try a few plausible method names
        for method in ("emit", "create_event", "record", "push"):
            fn = getattr(cand, method, None)
            if not fn:
                continue
            try:
                try:
                    fn(name, payload)
                except TypeError:
                    fn(payload)
                return
            except Exception:
                continue
