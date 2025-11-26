from __future__ import annotations
from fastapi.testclient import TestClient
import os

from sparkle_motion.function_tools.script_agent.entrypoint import app


def test_ready_and_invoke(tmp_path, monkeypatch):
    # Ensure no artificial model load delay and deterministic output
    monkeypatch.setenv("MODEL_LOAD_DELAY", "0")
    monkeypatch.setenv("DETERMINISTIC", "1")
    # Use a temporary artifacts dir
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(app)

    # Startup event should have set ready == True
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json().get("ready") is True

    # Provide a minimal non-empty shots list to satisfy validation
    payload = {"title": "Hardened Test", "shots": [{"id": "s1", "desc": "minimal shot"}]}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["artifact_uri"].startswith("file://") or data["artifact_uri"].startswith("artifact://")
