from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal, Mapping, MutableMapping, Optional

from . import adk_helpers, telemetry
from .production_agent import (
    ProductionResult,
    StepExecutionRecord,
    StepQueuedError,
    StepRateLimitExceededError,
    execute_plan,
)
from .run_registry import get_run_registry

QueueStatus = Literal["queued", "resuming", "completed", "failed", "abandoned"]


@dataclass
class QueueTicket:
    """Represents a queued production-agent run."""

    ticket_id: str
    plan_id: str
    plan_title: str
    step_id: str
    eta_epoch_s: float
    retry_after_s: float
    ttl_deadline_s: Optional[float]
    attempt: int
    max_attempts: int
    created_at: str
    status: QueueStatus
    message: str
    rate_limit_meta: Dict[str, Any]
    mode: Literal["dry", "run"]
    run_id: str
    plan_payload: Mapping[str, Any] = field(repr=False)

    def eta_seconds(self) -> float:
        return max(0.0, self.eta_epoch_s - time.time())

    def as_payload(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "plan_id": self.plan_id,
            "plan_title": self.plan_title,
            "step_id": self.step_id,
            "eta_epoch_s": self.eta_epoch_s,
            "retry_after_s": self.retry_after_s,
            "ttl_deadline_s": self.ttl_deadline_s,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "status": self.status,
            "message": self.message,
            "rate_limit": self.rate_limit_meta,
            "mode": self.mode,
            "run_id": self.run_id,
        }


Scheduler = Callable[[Callable[[], None]], None]
_EXECUTOR: Callable[..., ProductionResult] = execute_plan
_SCHEDULER: Scheduler


def _default_scheduler(task: Callable[[], None]) -> None:
    thread = threading.Thread(target=task, name="queue-resume", daemon=True)
    thread.start()


_SCHEDULER = _default_scheduler


def set_scheduler(scheduler: Scheduler) -> None:
    global _SCHEDULER
    _SCHEDULER = scheduler


def set_executor(executor: Callable[..., ProductionResult]) -> None:
    global _EXECUTOR
    _EXECUTOR = executor


def reset_scheduler() -> None:
    set_scheduler(_default_scheduler)


def reset_executor() -> None:
    set_executor(execute_plan)


def enqueue_plan(
    *,
    plan_payload: Mapping[str, Any],
    mode: Literal["dry", "run"],
    queued_record: StepExecutionRecord,
    run_id: str,
    max_attempts: int = 3,
) -> QueueTicket:
    if mode != "run":  # dry runs should not queue
        raise ValueError("Queued flow only applies to run mode")
    plan_dict = _coerce_plan_payload(plan_payload)
    eta_epoch_s, retry_after_s, ttl_deadline_s = _derive_eta(queued_record)
    ticket = QueueTicket(
        ticket_id=f"queue-{uuid.uuid4().hex[:12]}",
        plan_id=plan_dict.get("metadata", {}).get("plan_id") or _slugify(plan_dict.get("title")),
        plan_title=plan_dict.get("title", "Untitled Plan"),
        step_id=queued_record.step_id,
        eta_epoch_s=eta_epoch_s,
        retry_after_s=retry_after_s,
        ttl_deadline_s=ttl_deadline_s,
        attempt=1,
        max_attempts=max(1, max_attempts),
        created_at=_now_iso(),
        status="queued",
        message=_format_message(plan_dict, queued_record, eta_epoch_s),
        rate_limit_meta=dict(queued_record.meta.get("rate_limit", {})),
        mode=mode,
        run_id=run_id,
        plan_payload=plan_dict,
    )
    _persist_ticket(ticket)
    _schedule_resume(ticket)
    return ticket


def _schedule_resume(ticket: QueueTicket) -> None:
    delay = max(0.0, ticket.retry_after_s)

    def _task() -> None:
        time.sleep(delay)
        _resume_ticket(ticket)

    _SCHEDULER(_task)


def _resume_ticket(ticket: QueueTicket) -> None:
    registry = get_run_registry()
    _record_event("resuming", ticket)
    ticket.status = "resuming"
    try:
        progress_cb = lambda record: registry.record_step(ticket.run_id, record.as_dict())
        pre_step_hook = registry.pre_step_hook(ticket.run_id)
        _EXECUTOR(
            ticket.plan_payload,
            mode=ticket.mode,
            run_id=ticket.run_id,
            progress_callback=progress_cb,
            pre_step_hook=pre_step_hook,
        )
        ticket.status = "completed"
        ticket.message = f"Plan {ticket.plan_id} resumed and completed"
        _record_event("completed", ticket)
    except StepQueuedError as exc:
        if ticket.attempt >= ticket.max_attempts:
            ticket.status = "abandoned"
            ticket.message = f"Plan {ticket.plan_id} stayed queued after {ticket.attempt} attempts"
            _record_event("abandoned", ticket, extra={"last_record": exc.record.as_dict()})
            return
        ticket.attempt += 1
        ticket.rate_limit_meta = dict(exc.record.meta.get("rate_limit", {}))
        eta_epoch_s, retry_after_s, ttl_deadline_s = _derive_eta(exc.record)
        ticket.eta_epoch_s = eta_epoch_s
        ticket.retry_after_s = retry_after_s
        ticket.ttl_deadline_s = ttl_deadline_s
        ticket.message = _format_message(ticket.plan_payload, exc.record, eta_epoch_s)
        _record_event("requeued", ticket, extra={"attempt": ticket.attempt})
        _schedule_resume(ticket)
    except StepRateLimitExceededError as exc:
        ticket.status = "failed"
        ticket.message = f"Plan {ticket.plan_id} rejected by rate limiter"
        _record_event("failed", ticket, extra={"last_record": exc.record.as_dict()})
    except Exception as exc:
        ticket.status = "failed"
        ticket.message = f"Queue resume failed: {exc}"[:200]
        _record_event("failed", ticket, extra={"error": str(exc)})


def _coerce_plan_payload(plan: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(plan, dict):
        return dict(plan)
    # best-effort conversion for pydantic models
    to_dict = getattr(plan, "model_dump", None) or getattr(plan, "dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise ValueError("Unsupported plan payload type")


def _derive_eta(record: StepExecutionRecord) -> tuple[float, float, Optional[float]]:
    meta = record.meta.get("rate_limit", {}) if record.meta else {}
    eta_epoch_s = float(meta.get("eta_epoch_s") or 0.0)
    retry_after_s = float(meta.get("retry_after_s") or 0.0)
    ttl = meta.get("ttl_deadline_s")
    ttl_deadline_s = float(ttl) if ttl is not None else None
    if eta_epoch_s <= 0.0:
        eta_epoch_s = time.time() + max(retry_after_s, 1.0)
    if retry_after_s <= 0.0:
        retry_after_s = max(eta_epoch_s - time.time(), 1.0)
    return eta_epoch_s, retry_after_s, ttl_deadline_s


def _format_message(plan: Mapping[str, Any], record: StepExecutionRecord, eta_epoch_s: float) -> str:
    eta_dt = datetime.fromtimestamp(eta_epoch_s, tz=timezone.utc)
    return (
        f"Plan {plan.get('metadata', {}).get('plan_id') or _slugify(plan.get('title'))} is queued at step {record.step_id}; "
        f"next slot around {eta_dt.isoformat()}"
    )


def _persist_ticket(ticket: QueueTicket) -> None:
    payload = ticket.as_payload() | {"plan_payload": ticket.plan_payload}
    try:
        svc = adk_helpers.get_memory_service()
        svc.store_session_metadata(ticket.ticket_id, payload)
    except Exception:
        pass
    _record_event("queued", ticket)


def _record_event(status: str, ticket: QueueTicket, *, extra: Optional[MutableMapping[str, Any]] = None) -> None:
    payload: Dict[str, Any] = ticket.as_payload()
    payload["status"] = status
    if extra:
        payload.update(extra)
    try:
        adk_helpers.write_memory_event(run_id=ticket.plan_id, event_type="production_agent.queue", payload=payload)
    except adk_helpers.MemoryWriteError:
        pass
    try:
        telemetry.emit_event("production_agent.queue.status", payload)
    except Exception:
        pass


def _slugify(title: Optional[str]) -> str:
    if not title:
        return "plan"
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in title).strip("-") or "plan"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "enqueue_plan",
    "set_scheduler",
    "set_executor",
    "reset_scheduler",
    "reset_executor",
    "QueueTicket",
]
