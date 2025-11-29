from __future__ import annotations

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.assemble_ffmpeg.entrypoint import make_app


def test_assemble_ffmpeg_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clipdata")

    app = make_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

        r = client.get("/ready")
        assert r.status_code == 200
        jr = r.json()
        assert isinstance(jr.get("ready"), bool)

        payload = {"clips": [{"uri": str(clip)}], "options": {"fixture_only": True}}
        r = client.post("/invoke", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "success"
        uri = data.get("artifact_uri")
        assert uri
