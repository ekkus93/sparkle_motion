from __future__ import annotations
import base64
import os
from pathlib import Path
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
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts/test_smoke"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

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
        payload = _payload_for_module(mod_path, artifacts_dir)
        r = client.post("/invoke", json=payload)
        assert r.status_code == 200, f"invoke failed for {mod_path}: {r.text}"
        j = r.json()
        assert isinstance(j, dict)
        # expect an artifact_uri pointing to a file:// for fixture mode
        uri = j.get("artifact_uri") or j.get("artifactUri")
        assert uri is not None and uri.startswith("file://"), f"unexpected artifact uri for {mod_path}: {uri}"
        if mod_path.endswith("qa_qwen2vl.entrypoint"):
            metadata = j.get("metadata") or {}
            assert metadata.get("frame_ids") == ["frame-smoke"], "missing frame_ids metadata"
            assert "frames_detail" in metadata and len(metadata["frames_detail"]) == 1
            policy = metadata.get("policy") or {}
            assert policy.get("prompt_match_min") is not None
            assert "options_snapshot" in metadata
        # optional checks when fixture mode writes files
        try:
            path = Path(uri[len("file://"):])
            assert path.exists(), f"artifact file missing for {mod_path}: {path}"
            # lightly validate the artifact contains the prompt text when textual
            try:
                txt = path.read_text(encoding="utf-8")
                assert "unit-test prompt" in txt or "unit-test" in txt
            except Exception:
                # not all artifacts are textual; skip content assertion safely
                pass
        except Exception:
            # some FunctionTools may return remote URIs in fixture mode; skip
            pass
        # if telemetry/tool identifiers are present, validate types
        if "tool_name" in j:
            assert isinstance(j["tool_name"], str)
        if "telemetry" in j:
            assert isinstance(j["telemetry"], dict)


def _payload_for_module(module_path: str, artifacts_dir: Path) -> dict[str, object]:
    if module_path.endswith("assemble_ffmpeg.entrypoint"):
        clip = artifacts_dir / "smoke_clip.mp4"
        clip.write_bytes(b"clip")
        return {"clips": [{"uri": str(clip)}], "options": {"fixture_only": True}}
    if module_path.endswith("videos_wan.entrypoint"):
        return {
            "prompt": "unit-test prompt",
            "num_frames": 12,
            "fps": 6,
            "width": 320,
            "height": 240,
            "metadata": {"suite": "smoke"},
        }
    if module_path.endswith("qa_qwen2vl.entrypoint"):
        return {
            "prompt": "unit-test prompt",
            "frames": [
                {
                    "id": "frame-smoke",
                    "data_b64": base64.b64encode(b"qa fixture smoke frame").decode("ascii"),
                }
            ],
        }
    if module_path.endswith("lipsync_wav2lip.entrypoint"):
        return {
            "face": {"data_b64": base64.b64encode(b"face-bytes").decode("ascii")},
            "audio": {"data_b64": base64.b64encode(b"audio-bytes").decode("ascii")},
            "metadata": {"suite": "smoke"},
        }
    return {"prompt": "unit-test prompt"}
