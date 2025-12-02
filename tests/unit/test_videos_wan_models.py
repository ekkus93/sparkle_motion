from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.utils import assert_backend_artifact_uri, expected_artifact_scheme

from sparkle_motion.function_tools.videos_wan.models import VideosWanRequest, VideosWanResponse


mod = import_module("sparkle_motion.function_tools.videos_wan.entrypoint")
make_app = getattr(mod, "make_app")


def test_request_model_requires_prompt():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VideosWanRequest()


def test_invoke_produces_artifact_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    app = make_app()
    client = TestClient(app)

    payload = {
        "prompt": "pilot chunk",
        "num_frames": 12,
        "fps": 6,
        "width": 320,
        "height": 240,
        "metadata": {"shot": "shot-1"},
    }
    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert_backend_artifact_uri(data["artifact_uri"])
    meta = data["metadata"]
    assert meta["shot"] == "shot-1"
    local_path_str = meta.get("local_path")
    if not local_path_str and expected_artifact_scheme() != "filesystem":
        local_path_str = data["artifact_uri"][len("file://"):]
    assert local_path_str, "local_path missing from metadata"
    local_path = Path(local_path_str)
    assert local_path.exists()

    envelope = VideosWanResponse(**data)
    assert envelope.status == "success"
    assert envelope.artifact_uri == data["artifact_uri"]


def test_invoke_missing_frame_path_returns_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    app = make_app()
    client = TestClient(app)

    payload = {
        "prompt": "chunk",
        "start_frame_uri": str(tmp_path / "missing.png"),
    }
    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 400
