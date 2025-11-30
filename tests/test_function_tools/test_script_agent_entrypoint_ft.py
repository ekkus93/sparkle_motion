from __future__ import annotations
import json
from pathlib import Path

from fastapi.testclient import TestClient

from sparkle_motion import schema_registry
from sparkle_motion.function_tools.script_agent.entrypoint import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_invoke_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(app)
    payload = {"prompt": "test prompt"}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["artifact_uri"].startswith("file://") or data["artifact_uri"].startswith("artifact://")
    assert data["schema_uri"] == schema_registry.movie_plan_schema().uri
    artifact_uri = data["artifact_uri"]
    if artifact_uri.startswith("file://"):
        artifact_path = Path(artifact_uri.replace("file://", ""))
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text(encoding="utf-8"))
        plan = saved.get("validated_plan")
        assert isinstance(plan, dict)
        assert plan.get("shots")
        base_images = plan.get("base_images")
        assert isinstance(base_images, list)
        assert len(base_images) == len(plan["shots"]) + 1
        assert base_images[0]["id"] == "frame_000"
        assert plan["shots"][0]["start_base_image_id"] == "frame_000"
