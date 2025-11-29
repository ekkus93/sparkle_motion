from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.lipsync_wav2lip.entrypoint import app

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_invoke_smoke(tmp_path, monkeypatch, deterministic_media_assets: MediaAssets):
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(app)
    payload = {
        "face": {"data_b64": base64.b64encode(deterministic_media_assets.video.read_bytes()).decode("ascii")},
        "audio": {"data_b64": base64.b64encode(deterministic_media_assets.audio.read_bytes()).decode("ascii")},
    }
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["artifact_uri"].startswith("file://") or data["artifact_uri"].startswith("artifact://")

