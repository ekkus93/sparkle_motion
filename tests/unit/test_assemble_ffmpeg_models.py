from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from sparkle_motion.function_tools.assemble_ffmpeg import entrypoint, models

RequestModel = models.AssembleRequest
OptionsModel = models.AssembleOptions
ResponseModel = models.AssembleResponse
make_app = entrypoint.make_app

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def test_request_model_requires_clips():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel(clips=[])


def test_request_model_defaults_for_clip():
    req = RequestModel(clips=[{"uri": "file:///tmp/fake.mp4"}])
    clip = req.clips[0]
    assert clip.start_s == 0.0
    assert clip.end_s is None
    assert clip.metadata is None
    assert req.audio is None


def test_options_model_validates_ranges():
    from pydantic import ValidationError

    opts = OptionsModel()
    assert opts.video_codec == "libx264"
    assert opts.timeout_s == 120.0

    with pytest.raises(ValidationError):
        OptionsModel(timeout_s=0)


def test_response_model_round_trip(tmp_path):
    payload = ResponseModel(status="success", artifact_uri="file:///tmp/out.mp4", request_id="abc", metadata={"local_path": str(tmp_path)})
    dumped = payload.model_dump()
    assert dumped["status"] == "success"
    assert dumped["metadata"]["local_path"] == str(tmp_path)


def test_invoke_returns_metadata(monkeypatch, tmp_path, deterministic_media_assets: MediaAssets):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    clip = tmp_path / "clip.mp4"
    shutil.copyfile(deterministic_media_assets.video, clip)

    app = make_app()
    client = TestClient(app)

    payload = {
        "clips": [{"uri": str(clip)}],
        "options": {"fixture_only": True},
        "plan_id": "unit-plan",
        "metadata": {"test": True},
    }
    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "success"
    assert j.get("metadata", {}).get("engine") == "fixture"
    local_path = Path(j["metadata"]["local_path"])
    assert local_path.exists()
