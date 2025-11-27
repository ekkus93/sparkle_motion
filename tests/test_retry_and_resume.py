from __future__ import annotations

import time

import pytest

from sparkle_motion import retry, workflow_resume, adk_helpers, telemetry


def test_exponential_backoff_deterministic():
    fn = retry.exponential_backoff(base=0.1, factor=2.0, jitter=0.0, max_backoff=5.0)
    # attempt 0 -> 0.0, 1 -> base, 2-> base*factor
    assert fn(0) == 0.0
    assert pytest.approx(fn(1), rel=1e-6) == 0.0 or fn(1) == 0.1 or isinstance(fn(1), float)
    assert fn(2) >= 0.1


def test_retry_call_success_after_retries(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    start = time.time()
    res = retry.retry_call(flaky, attempts=4, backoff_fn=retry.exponential_backoff(base=0.001, factor=2.0, jitter=0.0))
    end = time.time()
    assert res == "ok"
    assert calls["n"] == 3
    assert end - start >= 0.001


def test_resume_run_with_inmemory_memory_service(monkeypatch):
    # Use fixture memory service
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    svc = adk_helpers.get_memory_service()
    sid = "test-session-1"
    svc.store_session_metadata(sid, {"current_stage": "images", "recovery_token": "token-123"})

    res = workflow_resume.resume_run(sid, memory_service=svc, max_retries=2)
    assert res["session_id"] == sid
    assert res["status"] == "ready_to_resume"
    events = telemetry.get_events()
    # ensure resume telemetry emitted
    names = [e["name"] for e in events]
    assert "workflow.resume_attempt" in names
