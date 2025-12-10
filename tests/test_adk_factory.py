from __future__ import annotations

import importlib

import pytest


def _reload_adk_factory(monkeypatch, *, fixture_mode: bool) -> object:
    if fixture_mode:
        monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    else:
        monkeypatch.setenv("ADK_USE_FIXTURE", "0")

    ah = importlib.reload(importlib.import_module("sparkle_motion.adk_helpers"))
    if fixture_mode and hasattr(ah, "_InMemoryMemoryService"):
        svc = ah.get_memory_service()
        if hasattr(svc, "clear_memory_events"):
            svc.clear_memory_events()

    mod = importlib.reload(importlib.import_module("sparkle_motion.adk_factory"))
    mod.shutdown()
    return mod


def test_get_agent_fixture_mode(monkeypatch):
    mod = _reload_adk_factory(monkeypatch, fixture_mode=True)
    agent = mod.get_agent("videos_wan", model_spec="wan-2.1")
    assert hasattr(agent, "info")
    info = agent.info()
    assert info["name"] == "videos_wan"


def test_get_agent_missing_sdk(monkeypatch):
    mod = _reload_adk_factory(monkeypatch, fixture_mode=False)
    ah = importlib.import_module("sparkle_motion.adk_helpers")

    def _raise():
        raise SystemExit(1)

    monkeypatch.setattr(ah, "probe_sdk", _raise)

    with pytest.raises(RuntimeError):
        mod.get_agent("videos_wan")

    importlib.reload(ah)


def test_get_agent_reuses_cached_handle(monkeypatch):
    mod = _reload_adk_factory(monkeypatch, fixture_mode=True)
    monkeypatch.setattr(mod.observability, "record_seed", lambda *_, **__: None)
    monkeypatch.setattr(mod.observability, "emit_agent_event", lambda *_, **__: None)

    first = mod.get_agent("script_agent", model_spec="alpha")
    key = next(iter(mod._agents))
    first_last_used = mod._agents[key].last_used_at
    second = mod.get_agent("script_agent", model_spec="alpha")

    assert first is second
    assert mod._agents[key].last_used_at >= first_last_used


def test_fixture_bypass_logs_memory_event(monkeypatch):
    mod = _reload_adk_factory(monkeypatch, fixture_mode=True)
    monkeypatch.setattr(mod.observability, "record_seed", lambda *_, **__: None)
    monkeypatch.setattr(mod.observability, "emit_agent_event", lambda *_, **__: None)
    monkeypatch.setattr(mod.observability, "get_session_id", lambda: "sess-123")

    mod.get_agent("images_stage", model_spec=None)
    svc = mod.adk_helpers.get_memory_service()
    events = svc.list_memory_events("sess-123") if hasattr(svc, "list_memory_events") else []
    assert events, "expected memory events to be recorded"
    last = events[-1]
    assert last["event_type"] == "adk_factory.fixture_agent"
    assert last["payload"] == {"tool": "images_stage", "model_spec": None, "mode": "per-tool"}


def test_close_agent_raises_lifecycle_error_when_unsuppressed(monkeypatch):
    mod = _reload_adk_factory(monkeypatch, fixture_mode=False)
    monkeypatch.setattr(mod.observability, "emit_agent_event", lambda *_, **__: None)

    class BadAgent:
        name = "bad"

        def close(self):
            raise RuntimeError("boom")

    with pytest.raises(mod.AdkAgentLifecycleError) as excinfo:
        mod.close_agent(BadAgent(), suppress_errors=False)

    assert "boom" in str(excinfo.value)
    assert excinfo.value.tool_name == "bad"


def test_shutdown_collects_and_raises_lifecycle_error(monkeypatch):
    mod = _reload_adk_factory(monkeypatch, fixture_mode=False)
    monkeypatch.setattr(mod.observability, "emit_agent_event", lambda *_, **__: None)

    class BadAgent:
        name = "bad"

        def close(self):
            raise RuntimeError("boom")

    class GoodAgent:
        name = "good"

        def __init__(self) -> None:
            self.closed = False

        def close(self):
            self.closed = True

    bad_cfg = mod.AgentConfig(tool_name="bad")
    good_cfg = mod.AgentConfig(tool_name="good")
    bad_agent = BadAgent()
    good_agent = GoodAgent()
    mod._agents["per-tool|bad"] = mod._AgentHandle(agent=bad_agent, config=bad_cfg, fixture=False)
    mod._agents["per-tool|good"] = mod._AgentHandle(agent=good_agent, config=good_cfg, fixture=False)

    with pytest.raises(mod.AdkAgentLifecycleError) as excinfo:
        mod.shutdown()

    assert excinfo.value.tool_name == "bad"
    assert "Failed to close agent" in str(excinfo.value)
    assert "per-tool|bad" not in mod._agents
    assert "per-tool|good" not in mod._agents
    assert good_agent.closed is True
