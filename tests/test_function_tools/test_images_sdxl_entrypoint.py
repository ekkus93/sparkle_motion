from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.images_sdxl.entrypoint import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_invoke_produces_png_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path / "runs"))

    client = TestClient(app)
    payload = {
        "prompt": "fixture prompt",
        "count": 2,
        "width": 64,
        "height": 64,
        "metadata": {"scene": "fixture"},
    }
    response = client.post("/invoke", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["artifacts"]) == 2
    artifact_uri = data["artifact_uri"]
    assert artifact_uri.startswith("file://")

    first_artifact = data["artifacts"][0]
    meta = first_artifact["metadata"]
    assert meta["engine"] == "fixture"
    assert meta["scene"] == "fixture"
    assert meta["prompt"] == payload["prompt"]
    assert meta["width"] == payload["width"]
    assert meta["height"] == payload["height"]
    local_path = Path(meta["source_path"])
    assert local_path.exists()
    blob = local_path.read_bytes()
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")

