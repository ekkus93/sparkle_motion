from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from sparkle_motion import gpu_utils
from sparkle_motion.function_tools.tts_chatterbox import entrypoint
from tests.unit.utils import assert_managed_artifact_uri


def test_health_endpoint():
    client = TestClient(entrypoint.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_invoke_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("SMOKE_TTS", raising=False)

    client = TestClient(entrypoint.app)
    payload = {"text": "test prompt", "voice_id": "emma", "sample_rate": 24000}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    uri = data["artifact_uri"]
    assert_managed_artifact_uri(uri)
    metadata = data.get("metadata")
    artifact_path = Path(metadata["local_path"])
    assert artifact_path.exists()
    assert metadata["voice_id"] == "emma"
    assert metadata["engine"] in {"fixture", "chatterbox"}
    engine_meta = metadata["engine_metadata"]
    assert engine_meta["mode"] == "fixture"


def test_invoke_returns_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    def _raise_busy(**_: object) -> None:
        raise gpu_utils.GpuBusyError(model_key="tts")

    monkeypatch.setattr(entrypoint.chatterbox_adapter, "synthesize_text", _raise_busy)
    client = TestClient(entrypoint.app)
    resp = client.post("/invoke", json={"prompt": "hello"})
    assert resp.status_code == 503
    assert "busy" in resp.json()["detail"].lower()

