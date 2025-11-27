import os
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


MODULES = [
    "sparkle_motion.function_tools.assemble_ffmpeg.entrypoint",
    "sparkle_motion.function_tools.videos_wan.entrypoint",
    "sparkle_motion.function_tools.qa_qwen2vl.entrypoint",
    "sparkle_motion.function_tools.images_sdxl.entrypoint",
    "sparkle_motion.function_tools.tts_chatterbox.entrypoint",
    "sparkle_motion.function_tools.lipsync_wav2lip.entrypoint",
]


@pytest.mark.parametrize("module_path", MODULES)
def test_tool_entrypoint_smoke(tmp_path: Path, monkeypatch, module_path: str):
    # Use deterministic artifact naming and fixture mode for fast, reliable tests
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

    mod = importlib.import_module(module_path)
    # prefer factory when available
    app = getattr(mod, "make_app", None)
    if callable(app):
        app = app()
    else:
        app = getattr(mod, "app")

    client = TestClient(app)

    # health
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

    # ready endpoint should return the readiness shape; tools may report False
    r = client.get("/ready")
    assert r.status_code == 200
    ready_json = r.json()
    assert "ready" in ready_json and "shutting_down" in ready_json

    # invoke
    payload = {"prompt": "smoke test prompt"}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j.get("status") == "success"
    uri = j.get("artifact_uri")
    assert uri and uri.startswith("file://")

    local_path = uri[len("file://") :]
    assert Path(local_path).exists()
