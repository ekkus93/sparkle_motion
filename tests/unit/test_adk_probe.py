from __future__ import annotations

import sys


from sparkle_motion import adk_helpers


def test_probe_sdk_returns_none_when_sdk_missing(monkeypatch):
    # Ensure google.adk is not importable
    monkeypatch.setitem(sys.modules, 'google.adk', None)
    # Remove any existing 'google.adk' entry if present
    sys.modules.pop('google.adk', None)
    adk_helpers.probe_sdk()
    # ADK probe unit test removed per user request (let runtime fail loudly if
    # `google.adk` is not installed).

    def test_placeholder():
        # Placeholder test to keep test discovery sane. The codebase is expected
        # to fail at runtime if `google.adk` is missing; no simulation is done here.
        assert True
