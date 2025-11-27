from __future__ import annotations

import pytest

from sparkle_motion import adk_helpers


def test_probe_sdk_env_aware():
    """Environment-aware test for `probe_sdk()` and `require_adk()`.

    This test does not attempt to simulate the SDK being present or absent;
    instead it asserts that the helper functions reflect the current
    interpreter environment deterministically:
    - If `probe_sdk()` returns `None`, `require_adk()` must raise SystemExit.
    - If `probe_sdk()` returns a module, it should be a usable module object.
    """
    res = adk_helpers.probe_sdk()
    if res is None:
        with pytest.raises(SystemExit):
            adk_helpers.require_adk()
    else:
        adk_mod, client = res
        assert adk_mod is not None
        assert hasattr(adk_mod, "__name__")


def test_get_memory_service_fixture_mode(monkeypatch):
    """Verify the fixture-mode MemoryService is returned when requested.

    This test sets `ADK_USE_FIXTURE=1` and checks that `get_memory_service`
    returns a working in-memory service suitable for unit tests.
    """
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    svc = adk_helpers.get_memory_service()
    assert svc is not None

    # Basic store/get/append semantics
    svc.store_session_metadata("session-1", {"k": "v"})
    meta = svc.get_session_metadata("session-1")
    assert isinstance(meta, dict) and meta.get("k") == "v"

    svc.append_reviewer_decision("session-1", {"decision": "ok"})
    decisions = svc.get_reviewer_decisions("session-1")
    assert isinstance(decisions, list) and any(d.get("decision") == "ok" for d in decisions)
