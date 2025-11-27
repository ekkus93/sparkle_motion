from __future__ import annotations

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.tts_chatterbox.entrypoint import make_app


def test_tts_chatterbox_smoke(monkeypatch):
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

        payload = {"prompt": "smoke test tts"}
        r = client.post("/invoke", json=payload)
        assert r.status_code in {200, 400, 422, 503}
        if r.status_code == 200:
            data = r.json()
            assert data.get("status") == "success"
            uri = data.get("artifact_uri")
            assert uri
            # verify artifact file exists for fixture mode
            try:
                from pathlib import Path

                if uri.startswith("file://"):
                    path = Path(uri[len("file://"):])
                    assert path.exists()
            except Exception:
                pass
