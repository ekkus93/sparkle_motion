from __future__ import annotations

import importlib
import os


def test_get_agent_fixture_mode(monkeypatch):
    os.environ["ADK_USE_FIXTURE"] = "1"
    mod = importlib.import_module("sparkle_motion.adk_factory")
    agent = mod.get_agent("videos_wan", model_spec="wan-2.1")
    assert hasattr(agent, "info")
    info = agent.info()
    assert info["name"] == "videos_wan"
    del os.environ["ADK_USE_FIXTURE"]


def test_get_agent_missing_sdk(monkeypatch):
    # Ensure real SDK probe fails and we raise RuntimeError
    mod = importlib.import_module("sparkle_motion.adk_factory")
    ah = importlib.import_module("sparkle_motion.adk_helpers")

    def _raise():
        raise SystemExit(1)

    monkeypatch.setattr(ah, "probe_sdk", _raise)
    # Ensure fixture is not set
    if "ADK_USE_FIXTURE" in os.environ:
        del os.environ["ADK_USE_FIXTURE"]

    try:
        try:
            mod.get_agent("videos_wan")
            assert False, "Expected RuntimeError when SDK probe fails"
        except RuntimeError:
            pass
    finally:
        # restore probe to original
        importlib.reload(ah)
