import importlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Parametrized list of tool entrypoint modules to test
MODULES = [
    "sparkle_motion.function_tools.script_agent.entrypoint",
    "sparkle_motion.function_tools.images_sdxl.entrypoint",
    "sparkle_motion.function_tools.videos_wan.entrypoint",
    "sparkle_motion.function_tools.tts_chatterbox.entrypoint",
    "sparkle_motion.function_tools.lipsync_wav2lip.entrypoint",
    "sparkle_motion.function_tools.assemble_ffmpeg.entrypoint",
    "sparkle_motion.function_tools.qa_qwen2vl.entrypoint",
]


@pytest.mark.parametrize("module_path", MODULES)
def test_entrypoint_contract(module_path: str, monkeypatch, tmp_path: Path):
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

    # /invoke
    r = client.post("/invoke", json={"prompt": "param test"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j.get("status") == "success"
    uri = j.get("artifact_uri")
    assert uri and uri.startswith("file://")
    local = uri[len("file://"):]
    assert Path(local).exists()


@pytest.mark.parametrize("module_path", MODULES)
def test_entrypoint_missing_prompt_returns_400(module_path: str, monkeypatch, tmp_path: Path):
    """Ensure endpoints validate required fields and return 400 when `prompt` is missing."""
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    mod = importlib.import_module(module_path)
    app = mod.make_app() if hasattr(mod, "make_app") else getattr(mod, "app")
    client = TestClient(app)

    # missing prompt should lead to a 400 Bad Request (or 422 Unprocessable Entity
    # depending on FastAPI/Pydantic validation behavior). Accept either.
    r = client.post("/invoke", json={})
    assert r.status_code in (400, 422), f"unexpected status: {r.status_code} / {r.text}"
