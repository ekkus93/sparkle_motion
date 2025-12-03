from __future__ import annotations

import base64
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from tests.unit.utils import assert_backend_artifact_uri, artifact_local_path

if TYPE_CHECKING:
    from tests.conftest import MediaAssets

# Parametrized list of tool entrypoint modules to test
MODULES = [
    "sparkle_motion.function_tools.script_agent.entrypoint",
    "sparkle_motion.function_tools.images_sdxl.entrypoint",
    "sparkle_motion.function_tools.videos_wan.entrypoint",
    "sparkle_motion.function_tools.tts_chatterbox.entrypoint",
    "sparkle_motion.function_tools.lipsync_wav2lip.entrypoint",
    "sparkle_motion.function_tools.assemble_ffmpeg.entrypoint",
]


@pytest.mark.parametrize("module_path", MODULES)
def test_entrypoint_contract(
    module_path: str,
    monkeypatch,
    tmp_path: Path,
    deterministic_media_assets: MediaAssets,
):
    """Shared test: ensure each entrypoint exposes RequestModel, can respond to
    /health, /ready and /invoke and writes an artifact under ARTIFACTS_DIR.
    """
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    mod = importlib.import_module(module_path)
    # RequestModel may be present as attribute
    assert hasattr(mod, "RequestModel"), f"{module_path} missing RequestModel"
    assert hasattr(mod, "make_app") or hasattr(mod, "app"), f"{module_path} missing app factory"

    app = mod.make_app() if hasattr(mod, "make_app") else getattr(mod, "app")
    client = TestClient(app)

    # /health
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

    # /ready shape
    r = client.get("/ready")
    assert r.status_code == 200
    ready_json = r.json()
    assert "ready" in ready_json and "shutting_down" in ready_json

    payload = _build_payload(module_path, tmp_path, deterministic_media_assets)
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j.get("status") == "success"
    uri = j.get("artifact_uri")
    assert_backend_artifact_uri(uri)
    meta = j.get("artifact_metadata") or j.get("metadata")
    local_path = artifact_local_path(uri, meta)
    if local_path:
        assert local_path.exists()


@pytest.mark.parametrize("module_path", MODULES)
def test_entrypoint_missing_prompt_returns_400(module_path: str, monkeypatch, tmp_path: Path):
    """Ensure endpoints validate required fields and return 400 when `prompt` is missing."""
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    mod = importlib.import_module(module_path)
    app = mod.make_app() if hasattr(mod, "make_app") else getattr(mod, "app")
    client = TestClient(app)

    missing_payload = _build_missing_payload(module_path)
    r = client.post("/invoke", json=missing_payload)
    assert r.status_code in (400, 422), f"unexpected status: {r.status_code} / {r.text}"


def _build_payload(module_path: str, tmp_path: Path, assets: MediaAssets) -> dict:
    if module_path.endswith("assemble_ffmpeg.entrypoint"):
        clip = tmp_path / "assemble_clip.mp4"
        clip.write_bytes(assets.video.read_bytes())
        return {"clips": [{"uri": str(clip)}], "options": {"fixture_only": True}}
    if module_path.endswith("videos_wan.entrypoint"):
        return {
            "prompt": "param test",
            "num_frames": 8,
            "fps": 4,
            "width": 320,
            "height": 240,
            "metadata": {"suite": "param"},
        }
    if module_path.endswith("lipsync_wav2lip.entrypoint"):
        return {
            "face": _frame_payload_from_path(assets.video, frame_id=None),
            "audio": _frame_payload_from_path(assets.audio, frame_id=None),
            "metadata": {"suite": "param"},
        }
    return {"prompt": "param test"}


def _build_missing_payload(module_path: str) -> dict:
    if module_path.endswith("assemble_ffmpeg.entrypoint"):
        return {"clips": []}
    return {}


def _frame_payload_from_path(path: Path, frame_id: str | None = "frame") -> dict:
    return _frame_payload_from_bytes(path.read_bytes(), frame_id)


def _frame_payload_from_bytes(data: bytes, frame_id: str | None = "frame") -> dict:
    payload = {"data_b64": base64.b64encode(data).decode("ascii")}
    if frame_id is not None:
        payload["id"] = frame_id
    return payload
