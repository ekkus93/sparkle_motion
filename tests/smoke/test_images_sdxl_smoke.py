from __future__ import annotations
from fastapi.testclient import TestClient
from pathlib import Path

from sparkle_motion.function_tools.images_sdxl.entrypoint import app


def test_images_sdxl_smoke(tmp_path, monkeypatch):
    # deterministic mode and artifact dir
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert "status" in r.json()

        r = client.get("/ready")
        assert r.status_code == 200
        # app in test-mode should report ready True
        jr = r.json()
        assert isinstance(jr.get("ready"), bool)

        payload = {"prompt": "a test prompt for sdxl"}
        r = client.post("/invoke", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "success"
        assert data.get("artifact_uri")
        uri = data["artifact_uri"]
        assert uri.startswith("file://")
        path = Path(uri[len("file://"):])
        assert path.exists()
        # check that file contains a serialized version of the request
        text = path.read_text(encoding="utf-8")
        assert "test prompt" in text
