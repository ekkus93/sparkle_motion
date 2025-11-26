from __future__ import annotations
from fastapi.testclient import TestClient
import os

from sparkle_motion.function_tools.script_agent.entrypoint import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_invoke_endpoint_deterministic(tmp_path):
    # Ensure deterministic artifact path for stable assertions
    os.environ["DETERMINISTIC"] = "1"
    client = TestClient(app)
    payload = {"title": "Test Movie", "shots": []}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["artifact_uri"].startswith("file://") or data["artifact_uri"].startswith("artifact://")
