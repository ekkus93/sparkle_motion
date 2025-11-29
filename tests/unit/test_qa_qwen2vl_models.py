from __future__ import annotations

import base64
from importlib import import_module
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient


mod = import_module("sparkle_motion.function_tools.qa_qwen2vl.entrypoint")
RequestModel = getattr(mod, "RequestModel")
make_app = getattr(mod, "make_app")

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def test_request_model_requires_prompt():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel(frames=[{"data_b64": _b64(b"frame"), "id": "f1"}])


def test_request_model_requires_frames():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel(prompt="ok", frames=[])


def test_invoke_writes_artifact_and_returns_success(
    monkeypatch, tmp_path, deterministic_media_assets: MediaAssets
):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    app = make_app()
    client = TestClient(app)

    resp = client.post(
        "/invoke",
        json={
            "prompt": "smoke",
            "frames": [
                {
                    "id": "frame1",
                    "data_b64": _b64(deterministic_media_assets.image.read_bytes()),
                }
            ],
        },
    )
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "success"
    assert j.get("artifact_uri", "").startswith("file://")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
