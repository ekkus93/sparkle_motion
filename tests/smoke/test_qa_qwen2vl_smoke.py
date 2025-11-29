from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.qa_qwen2vl.entrypoint import make_app

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def test_qa_qwen2vl_smoke(monkeypatch, deterministic_media_assets: MediaAssets):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")

    app = make_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

        r = client.get("/ready")
        assert r.status_code == 200
        jr = r.json()
        assert isinstance(jr.get("ready"), bool)

        payload = {
            "prompt": "smoke test qa",
            "frames": [
                {
                    "id": "frame1",
                    "data_b64": base64.b64encode(deterministic_media_assets.image.read_bytes()).decode("ascii"),
                }
            ],
        }
        r = client.post("/invoke", json=payload)
        assert r.status_code in {200, 400, 422, 503}
        if r.status_code == 200:
            data = r.json()
            assert data.get("status") == "success"
            uri = data.get("artifact_uri")
            assert uri
            metadata = data.get("metadata") or {}
            assert metadata.get("frame_ids") == ["frame1"]
            assert "frames_detail" in metadata and len(metadata["frames_detail"]) == 1
            assert metadata.get("policy", {}).get("prompt_match_min") is not None
            assert "options_snapshot" in metadata
