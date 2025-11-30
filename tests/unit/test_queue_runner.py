from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Callable, List, Tuple

import pytest

from sparkle_motion import queue_runner
from sparkle_motion.production_agent import (
    StepExecutionRecord,
    StepQueuedError,
    StepRateLimitExceededError,
)


class _MemoryStub:
    def __init__(self) -> None:
        self.metadata: dict[str, dict[str, Any]] = {}

    def store_session_metadata(self, ticket_id: str, payload: dict[str, Any]) -> None:
        self.metadata[ticket_id] = payload


def _build_rate_limit_meta(*, eta_delta: float = 1.0, retry_after: float = 0.05, ttl: float | None = None) -> dict[str, Any]:
    eta = time.time() + eta_delta
    meta: dict[str, Any] = {
        "eta_epoch_s": eta,
        "retry_after_s": retry_after,
    }
    if ttl is not None:
        meta["ttl_deadline_s"] = time.time() + ttl
    return meta


def _make_record(*, rate_limit: dict[str, Any] | None = None, step_id: str = "shot:images") -> StepExecutionRecord:
    now = datetime.now(timezone.utc).isoformat()
    meta = {"rate_limit": rate_limit or _build_rate_limit_meta()}
    return StepExecutionRecord(
        plan_id="plan-alpha",
        step_id=step_id,
        step_type="images",
        status="queued",
        start_time=now,
        end_time=now,
        duration_s=0.1,
        attempts=1,
        meta=meta,
    )


@pytest.fixture
def memory_hooks(monkeypatch: pytest.MonkeyPatch) -> Tuple[_MemoryStub, List[dict[str, Any]], List[Tuple[str, Any]]]:
    stub = _MemoryStub()
    events: List[dict[str, Any]] = []
    telemetry_events: List[Tuple[str, Any]] = []

    monkeypatch.setattr(queue_runner.adk_helpers, "get_memory_service", lambda: stub)

    def _record_event(*, run_id: str | None, event_type: str, payload: dict[str, Any]) -> None:
        events.append({"run_id": run_id, "event_type": event_type, "payload": payload})

    monkeypatch.setattr(queue_runner.adk_helpers, "write_memory_event", _record_event)
    monkeypatch.setattr(queue_runner.telemetry, "emit_event", lambda name, payload: telemetry_events.append((name, payload)))
    return stub, events, telemetry_events


@pytest.fixture
def scheduler_stub() -> List[Callable[[], None]]:
    tasks: List[Callable[[], None]] = []

    def _schedule(task: Callable[[], None]) -> None:
        tasks.append(task)

    queue_runner.set_scheduler(_schedule)
    yield tasks
    queue_runner.reset_scheduler()


@pytest.fixture(autouse=True)
def _reset_executor() -> None:
    queue_runner.reset_executor()
    yield
    queue_runner.reset_executor()


@pytest.fixture
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue_runner.time, "sleep", lambda _: None)


def _status_sequence(events: List[dict[str, Any]]) -> List[str]:
    return [evt["payload"].get("status", "") for evt in events if evt["event_type"] == "production_agent.queue"]


def test_enqueue_plan_persists_ticket_and_records_event(memory_hooks, scheduler_stub) -> None:
    memory_stub, events, _ = memory_hooks
    record = _make_record(rate_limit=_build_rate_limit_meta(eta_delta=0.1, retry_after=0.0))
    plan_payload = {"title": "Epic Plan", "metadata": {"plan_id": "plan-alpha"}}

    ticket = queue_runner.enqueue_plan(plan_payload=plan_payload, mode="run", queued_record=record, run_id="run-test")

    assert ticket.ticket_id in memory_stub.metadata
    stored = memory_stub.metadata[ticket.ticket_id]
    assert stored["plan_payload"]["title"] == "Epic Plan"
    assert scheduler_stub, "enqueue_plan should schedule a resume task"
    assert ticket.status == "queued"
    statuses = _status_sequence(events)
    assert "queued" in statuses


def test_resume_ticket_completes_successfully(memory_hooks, scheduler_stub, fast_sleep) -> None:
    _, events, _ = memory_hooks
    calls: List[Tuple[dict[str, Any], str]] = []

    def _executor(plan_payload: dict[str, Any], *, mode: str, run_id: str, **_: Any) -> None:
        calls.append((plan_payload, mode))

    queue_runner.set_executor(_executor)
    record = _make_record(rate_limit=_build_rate_limit_meta(eta_delta=0.05, retry_after=0.0))
    plan_payload = {"title": "Epic Plan", "metadata": {"plan_id": "plan-alpha"}}

    ticket = queue_runner.enqueue_plan(plan_payload=plan_payload, mode="run", queued_record=record, run_id="run-test")
    assert scheduler_stub, "resume task not scheduled"

    task = scheduler_stub.pop(0)
    task()

    assert calls and calls[0][1] == "run"
    assert ticket.status == "completed"
    statuses = _status_sequence(events)
    assert statuses[:3] == ["queued", "resuming", "completed"]


def test_resume_ticket_requeues_until_abandoned(memory_hooks, scheduler_stub, fast_sleep) -> None:
    _, events, _ = memory_hooks
    records = [
        _make_record(rate_limit=_build_rate_limit_meta(eta_delta=0.01, retry_after=0.0)),
        _make_record(rate_limit=_build_rate_limit_meta(eta_delta=0.02, retry_after=0.0)),
    ]

    def _executor(*_: Any, **__: Any) -> None:
        raise StepQueuedError("still queued", record=records.pop(0))

    queue_runner.set_executor(_executor)
    ticket = queue_runner.enqueue_plan(
        plan_payload={"title": "Epic", "metadata": {"plan_id": "plan-alpha"}},
        mode="run",
        queued_record=_make_record(rate_limit=_build_rate_limit_meta(eta_delta=0.01, retry_after=0.0)),
        run_id="run-test",
        max_attempts=2,
    )

    first_task = scheduler_stub.pop(0)
    first_task()
    assert scheduler_stub, "ticket should be requeued for second attempt"

    second_task = scheduler_stub.pop(0)
    second_task()

    assert ticket.status == "abandoned"
    assert ticket.attempt == 2
    statuses = _status_sequence(events)
    assert statuses.count("resuming") == 2
    assert statuses[-1] == "abandoned"


def test_resume_ticket_marks_failed_when_rate_limit_exceeded(memory_hooks, scheduler_stub, fast_sleep) -> None:
    _, events, _ = memory_hooks
    rate_limit_record = _make_record(rate_limit=_build_rate_limit_meta(eta_delta=0.01, retry_after=0.0))

    def _executor(*_: Any, **__: Any) -> None:
        raise StepRateLimitExceededError("budget exhausted", record=rate_limit_record)

    queue_runner.set_executor(_executor)
    ticket = queue_runner.enqueue_plan(
        plan_payload={"title": "Epic", "metadata": {"plan_id": "plan-alpha"}},
        mode="run",
        queued_record=rate_limit_record,
        run_id="run-test",
    )

    task = scheduler_stub.pop(0)
    task()

    assert ticket.status == "failed"
    assert "rate limiter" in ticket.message
    statuses = _status_sequence(events)
    assert statuses[-1] == "failed"


def test_enqueue_plan_rejects_dry_mode(memory_hooks, scheduler_stub) -> None:
    record = _make_record(rate_limit=_build_rate_limit_meta())

    with pytest.raises(ValueError, match="Queued flow only applies to run mode"):
        queue_runner.enqueue_plan(plan_payload={"title": "Dry"}, mode="dry", queued_record=record, run_id="run-test")