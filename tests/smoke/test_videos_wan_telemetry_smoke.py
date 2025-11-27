from __future__ import annotations

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.videos_wan.entrypoint import make_app
from sparkle_motion import telemetry


def test_videos_wan_emits_telemetry(monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("VIDEOS_WAN_SEED", "12345")
    telemetry.clear_events()

    app = make_app()
    client = TestClient(app)

    # health/readiness
    r = client.get("/health")
    assert r.status_code == 200
    r = client.get("/ready")
    assert r.status_code == 200

    payload = {"prompt": "hello world"}
    r = client.post("/invoke", json=payload)
    assert r.status_code in (200, 503)

    events = telemetry.get_events()
    names = [e["name"] for e in events]
    # Some agent lifecycle events may be emitted via other paths; ensure
    # that invoke lifecycle telemetry is present which indicates telemetry
    # wiring in the entrypoint is operational in fixture mode.
    assert "invoke.received" in names
    assert "invoke.completed" in names
