

def test_memory_service_emits_telemetry(monkeypatch):
    # Use the fixture memory service for deterministic behavior
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")

    # import here so env var takes effect before get_memory_service() runs
    from sparkle_motion import telemetry
    from sparkle_motion.adk_helpers import get_memory_service

    telemetry.clear_events()

    svc = get_memory_service()
    svc.store_session_metadata("sess-1", {"foo": "bar"})
    svc.append_reviewer_decision("sess-1", {"approved": True})

    # reads should also emit telemetry
    _ = svc.get_session_metadata("sess-1")
    _ = svc.get_reviewer_decisions("sess-1")

    events = telemetry.get_events()
    names = [e["name"] for e in events]

    assert "memory.store_session_metadata" in names
    assert "memory.append_reviewer_decision" in names
    assert "memory.get_session_metadata" in names
    assert "memory.get_reviewer_decisions" in names
