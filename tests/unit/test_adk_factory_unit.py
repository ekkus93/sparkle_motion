from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sparkle_motion import adk_factory


def _install_memory_writer(monkeypatch):
    events: list[dict[str, object]] = []

    def fake_writer(*, run_id, event_type, payload, **_: object) -> None:
        events.append({"run_id": run_id, "event_type": event_type, "payload": payload})

    monkeypatch.setattr(adk_factory.adk_helpers, "write_memory_event", fake_writer)
    monkeypatch.setattr(adk_factory.observability, "get_session_id", lambda: "test-run")
    return events


@pytest.fixture(autouse=True)
def _reset_factory(monkeypatch):
    monkeypatch.delenv("ADK_USE_FIXTURE", raising=False)
    monkeypatch.setattr(adk_factory, "_agents", {})


def test_safe_probe_sdk_logs_system_exit(monkeypatch):
    events = _install_memory_writer(monkeypatch)

    def boom():
        raise SystemExit("boom")

    monkeypatch.setattr(adk_factory.adk_helpers, "probe_sdk", boom)

    assert adk_factory.safe_probe_sdk() is None

    assert events[-1] == {
        "run_id": "test-run",
        "event_type": "adk_factory.sdk_probe_failure",
        "payload": {"reason": "system_exit", "message": "boom"},
    }


def test_require_adk_fixture_mode_logs_event(monkeypatch):
    events = _install_memory_writer(monkeypatch)
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")

    result = adk_factory.require_adk(allow_fixture=True, tool_name="script_agent", model_spec="qwen")
    assert result == (None, None)

    assert events[-1] == {
        "run_id": "test-run",
        "event_type": "adk_factory.require_adk.fixture_fallback",
        "payload": {"tool": "script_agent", "model_spec": "qwen", "allow_fixture": True},
    }


def test_require_adk_failure_emits_event(monkeypatch):
    events = _install_memory_writer(monkeypatch)
    monkeypatch.setattr(adk_factory, "safe_probe_sdk", lambda: None)

    with pytest.raises(adk_factory.MissingAdkSdkError):
        adk_factory.require_adk(tool_name="images_stage", model_spec="sdxl")

    assert events[-1] == {
        "run_id": "test-run",
        "event_type": "adk_factory.require_adk.failure",
        "payload": {"tool": "images_stage", "model_spec": "sdxl"},
    }


def test_get_agent_caches_and_updates_last_used(monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    first_agent = adk_factory.get_agent("script_agent", model_spec="fixture")
    cfg = adk_factory.AgentConfig(tool_name="script_agent", model_spec="fixture")
    handle = adk_factory._agents[adk_factory._agent_key(cfg)]
    original_last_used = datetime(2025, 1, 1, tzinfo=timezone.utc)
    handle.last_used_at = original_last_used

    next_ts = datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(adk_factory, "_utcnow", lambda: next_ts)

    second_agent = adk_factory.get_agent("script_agent", model_spec="fixture")

    assert first_agent is second_agent
    assert handle.last_used_at == next_ts


def test_create_agent_fixture_emits_memory_event(monkeypatch):
    events = _install_memory_writer(monkeypatch)
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")

    adk_factory.create_agent(tool_name="images_stage", model_spec="stub")

    assert events[-1] == {
        "run_id": "test-run",
        "event_type": "adk_factory.fixture_agent",
        "payload": {"tool": "images_stage", "model_spec": "stub", "mode": "per-tool"},
    }


def test_close_agent_raises_when_shutdown_fails(monkeypatch):
    monkeypatch.setattr(adk_factory.observability, "emit_agent_event", lambda *_, **__: None)

    class BrokenAgent:
        name = "broken"

        def close(self):  # pragma: no cover - invoked via close_agent
            raise RuntimeError("boom")

    cfg = adk_factory.AgentConfig(tool_name="broken")
    handle = adk_factory._AgentHandle(agent=BrokenAgent(), config=cfg, fixture=True)
    monkeypatch.setattr(adk_factory, "_agents", {adk_factory._agent_key(cfg): handle})

    with pytest.raises(adk_factory.AdkAgentLifecycleError) as exc:
        adk_factory.close_agent("broken", suppress_errors=False)

    assert exc.value.tool_name == "broken"
    assert adk_factory._agents == {}


def test_shutdown_reports_first_error_and_clears_registry(monkeypatch):
    emitted: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(adk_factory.observability, "emit_agent_event", lambda name, payload: emitted.append((name, payload)))

    class FakeAgent:
        def __init__(self, name: str) -> None:
            self.name = name

    bad_cfg = adk_factory.AgentConfig(tool_name="bad")
    good_cfg = adk_factory.AgentConfig(tool_name="good")
    bad_handle = adk_factory._AgentHandle(agent=FakeAgent("bad"), config=bad_cfg, fixture=False)
    good_handle = adk_factory._AgentHandle(agent=FakeAgent("good"), config=good_cfg, fixture=False)
    registry = {
        adk_factory._agent_key(bad_cfg): bad_handle,
        adk_factory._agent_key(good_cfg): good_handle,
    }
    monkeypatch.setattr(adk_factory, "_agents", registry)

    def fake_shutdown(agent):
        if agent.name == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(adk_factory, "_shutdown_agent", fake_shutdown)

    with pytest.raises(adk_factory.AdkAgentLifecycleError) as exc:
        adk_factory.shutdown()

    assert exc.value.tool_name == "bad"
    assert adk_factory._agents == {}
    assert emitted == [("agent.closed", {"tool": "good", "model_spec": None})]
