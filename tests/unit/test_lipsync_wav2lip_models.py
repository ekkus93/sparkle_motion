from __future__ import annotations

import base64
from importlib import import_module
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient


mod = import_module("sparkle_motion.function_tools.lipsync_wav2lip.entrypoint")
RequestModel = getattr(mod, "RequestModel")
make_app = getattr(mod, "make_app")

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def _face_audio_payload(assets: MediaAssets) -> dict:
    face_data = base64.b64encode(assets.video.read_bytes()).decode("ascii")
    audio_data = base64.b64encode(assets.audio.read_bytes()).decode("ascii")
    return {
        "face": {"data_b64": face_data},
        "audio": {"data_b64": audio_data},
    }


def test_request_model_requires_face_and_audio():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel()
    with pytest.raises(ValidationError):
        RequestModel(face={"data_b64": base64.b64encode(b"face").decode("ascii")})
    with pytest.raises(ValidationError):
        RequestModel(audio={"data_b64": base64.b64encode(b"audio").decode("ascii")})


def test_invoke_writes_artifact_and_returns_success(monkeypatch, tmp_path, deterministic_media_assets: MediaAssets):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    app = make_app()
    client = TestClient(app)

    resp = client.post("/invoke", json=_face_audio_payload(deterministic_media_assets))
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "success"
    assert j.get("artifact_uri", "").startswith("file://")
