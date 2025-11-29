import base64
from importlib import import_module

import pytest
from fastapi.testclient import TestClient


mod = import_module("sparkle_motion.function_tools.lipsync_wav2lip.entrypoint")
RequestModel = getattr(mod, "RequestModel")
make_app = getattr(mod, "make_app")


def _face_audio_payload() -> dict:
    data = base64.b64encode(b"media").decode("ascii")
    return {
        "face": {"data_b64": data},
        "audio": {"data_b64": data},
    }


def test_request_model_requires_face_and_audio():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel()
    with pytest.raises(ValidationError):
        RequestModel(face={"data_b64": base64.b64encode(b"face").decode("ascii")})
    with pytest.raises(ValidationError):
        RequestModel(audio={"data_b64": base64.b64encode(b"audio").decode("ascii")})


def test_invoke_writes_artifact_and_returns_success(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    app = make_app()
    client = TestClient(app)

    resp = client.post("/invoke", json=_face_audio_payload())
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "success"
    assert j.get("artifact_uri", "").startswith("file://")
