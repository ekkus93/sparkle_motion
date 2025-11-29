import pytest
from fastapi.testclient import TestClient
from importlib import import_module
from pathlib import Path


mod = import_module("sparkle_motion.function_tools.assemble_ffmpeg.entrypoint")
RequestModel = getattr(mod, "RequestModel")
make_app = getattr(mod, "make_app")


def test_request_model_requires_clips():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel(clips=[])


def test_invoke_returns_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clipdata")

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
