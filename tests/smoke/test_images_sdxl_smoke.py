from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.images_sdxl.entrypoint import app


def test_images_sdxl_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path / "runs"))

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert "status" in r.json()

        r = client.get("/ready")
        assert r.status_code == 200
        jr = r.json()
        assert isinstance(jr.get("ready"), bool)

        payload = {"prompt": "a test prompt for sdxl", "width": 64, "height": 64}
        r = client.post("/invoke", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "success"
        assert data.get("artifact_uri")
        uri = data["artifact_uri"]
        assert uri.startswith("file://")
        path = Path(uri[len("file://"):])
        assert path.exists()
        blob = path.read_bytes()
        assert blob.startswith(b"\x89PNG\r\n\x1a\n")
        assert data["artifacts"][0]["metadata"]["phash"]
