from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.assemble_ffmpeg.entrypoint import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_invoke_fixture_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clipdata")

    client = TestClient(app)
    payload = {
        "plan_id": "plan-123",
        "clips": [{"uri": str(clip)}],
        "options": {"fixture_only": True},
    }
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["metadata"]["engine"] == "fixture"
    assert data["artifact_uri"].startswith("file://")
    local_path = Path(data["metadata"]["local_path"])
    assert local_path.exists()

