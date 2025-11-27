from __future__ import annotations
import os
import importlib
from fastapi.testclient import TestClient


FUNCTION_TOOL_MODULES = [
    "sparkle_motion.function_tools.images_sdxl.entrypoint",
    "sparkle_motion.function_tools.assemble_ffmpeg.entrypoint",
    "sparkle_motion.function_tools.videos_wan.entrypoint",
    "sparkle_motion.function_tools.qa_qwen2vl.entrypoint",
    "sparkle_motion.function_tools.script_agent.entrypoint",
    "sparkle_motion.function_tools.lipsync_wav2lip.entrypoint",
    "sparkle_motion.function_tools.tts_chatterbox.entrypoint",
]


def test_function_tools_basic_smoke(monkeypatch):
    # Put tools into deterministic/fixture mode so they don't try to call external ADK
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    # ensure artifacts dir is writable in CI/temp
    monkeypatch.setenv("ARTIFACTS_DIR", os.environ.get("ARTIFACTS_DIR", "artifacts/test_smoke"))

    for mod_path in FUNCTION_TOOL_MODULES:
        mod = importlib.import_module(mod_path)
        # Expect modules to expose either `app` or `make_app()`
        if hasattr(mod, "app"):
            app = getattr(mod, "app")
        elif hasattr(mod, "make_app"):
            app = mod.make_app()
        else:
            raise AssertionError(f"module {mod_path} exposes no app")

        client = TestClient(app)

        # health
        r = client.get("/health")
        assert r.status_code == 200
        j = r.json()
        assert isinstance(j, dict) and j.get("status") in ("ok", "healthy", None) or "status" in j

        # ready - should be present and boolean
        r = client.get("/ready")
        assert r.status_code == 200
        j = r.json()
        assert isinstance(j, dict)
        assert "ready" in j

        # invoke - basic payload
        payload = {"prompt": "unit-test prompt"}
        r = client.post("/invoke", json=payload)
        assert r.status_code == 200, f"invoke failed for {mod_path}: {r.text}"
        j = r.json()
        assert isinstance(j, dict)
        # expect an artifact_uri pointing to a file:// for fixture mode
        uri = j.get("artifact_uri") or j.get("artifactUri")
        assert uri is not None and uri.startswith("file://"), f"unexpected artifact uri for {mod_path}: {uri}"
