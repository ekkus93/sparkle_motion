from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from . import adk_helpers, retry, telemetry

log = logging.getLogger(__name__)


def resume_run(session_id: str, start_stage: Optional[str] = None, memory_service: Optional[Any] = None, max_retries: int = 3) -> Dict[str, Any]:
    """Resume a workflow run from a given session id.

    This is a light-weight helper that reads session metadata from the
    MemoryService, determines a resume point, and emits telemetry. It does not
    attempt to start the WorkflowAgent itself (that's owned by the runner).

    Returns a dict with resume information.
    """
    svc = memory_service
    if svc is None:
        try:
            svc = adk_helpers.get_memory_service()
        except Exception as e:
            raise RuntimeError(f"No MemoryService available to resume run: {e}") from e

    # read metadata
    meta = svc.get_session_metadata(session_id) or {}

    # determine resume stage
    resume_stage = start_stage or meta.get("last_failed_stage") or meta.get("current_stage")

    # record telemetry about resume
    try:
        telemetry.emit_event("workflow.resume_attempt", {"session_id": session_id, "resume_stage": resume_stage})
    except Exception:
        pass

    # implement a best-effort retry of a small recovery action recorded in metadata
    def _recovery_action():
        # If metadata contains a 'recovery_token' we attempt to validate it via SDK
        token = meta.get("recovery_token")
        if token:
            # try SDK helper to validate token; many SDKs will have a method to
            # rehydrate state, but this is best-effort and may be a no-op in tests.
            try:
                adk_mod, _ = adk_helpers.probe_sdk()
                validator = getattr(adk_mod, "validate_recovery_token", None)
                if callable(validator):
                    return validator(token)
            except Exception:
                # best-effort; tests rely on this not raising
                return None
        # nothing to do
        return None

    # run the recovery action with retries
    try:
        retry.retry_call(_recovery_action, attempts=max_retries, backoff_fn=retry.exponential_backoff(base=0.05, factor=2.0, jitter=0.0))
    except Exception as e:
        # emit telemetry and continue; resume may still be attempted
        try:
            telemetry.emit_event("workflow.resume_recovery_failed", {"session_id": session_id, "error": str(e)})
        except Exception:
            pass

    result = {"session_id": session_id, "resume_stage": resume_stage, "status": "ready_to_resume"}
    try:
        telemetry.emit_event("workflow.resume_ready", result)
    except Exception:
        pass

    return result
