from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


_EVENTS: List[Dict[str, Any]] = []


def emit_event(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Record a telemetry event in-process and optionally persist to a log file.

    This is intentionally lightweight for tests and local development. Tests
    can inspect `get_events()` to verify expected emissions.
    """
    ev: Dict[str, Any] = {"name": name, "payload": payload or {}}
    _EVENTS.append(ev)
    log_path = os.environ.get("ADK_TELEMETRY_LOG")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev) + "\n")
        except Exception:
            # Telemetry best-effort: don't raise on logging failures.
            pass


def get_events() -> List[Dict[str, Any]]:
    """Return a copy of recorded events."""
    return list(_EVENTS)


def clear_events() -> None:
    """Clear the in-memory event buffer."""
    _EVENTS.clear()
