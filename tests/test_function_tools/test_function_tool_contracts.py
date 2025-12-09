"""Contract tests that compare FunctionTool responses to documented samples."""

from __future__ import annotations

import json
from pathlib import Path
import base64

from fastapi.testclient import TestClient
import pytest

from sparkle_motion.function_tools.script_agent.entrypoint import app as script_app
from sparkle_motion.function_tools.production_agent.entrypoint import app as production_app
from sparkle_motion.function_tools.assemble_ffmpeg.entrypoint import app as assemble_app
from sparkle_motion.function_tools.images_sdxl.entrypoint import app as images_app
from sparkle_motion.function_tools.tts_chatterbox.entrypoint import app as tts_app
from sparkle_motion.function_tools.lipsync_wav2lip.entrypoint import app as lipsync_app
from sparkle_motion.function_tools.videos_wan.entrypoint import app as videos_app

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "samples" / "function_tools"


def _load_sample(name: str) -> dict[str, object]:
    path = SAMPLES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_script_agent_response_matches_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(script_app)
    resp = client.post("/invoke", json={"prompt": "contract sample"})
    assert resp.status_code == 200
    data = resp.json()

    sample = _load_sample("script_agent_response.sample.json")
    normalized = dict(data)
    normalized["artifact_uri"] = "__ARTIFACT_URI__"
    normalized["request_id"] = "__REQUEST_ID__"

    assert normalized == sample


def test_production_agent_response_matches_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_plan = {
        "title": "Test Film",
        "metadata": {"plan_id": "plan-entrypoint"},
        "base_images": [
            {"id": "frame_000", "prompt": "hero start"},
            {"id": "frame_001", "prompt": "hero end"},
        ],
        "shots": [
            {
                "id": "shot-1",
                "duration_sec": 2,
                "visual_description": "A hero poses",
                "start_base_image_id": "frame_000",
                "end_base_image_id": "frame_001",
                "dialogue": [],
                "is_talking_closeup": False,
            }
        ],
        "dialogue_timeline": [
            {"type": "silence", "start_time_sec": 0.0, "duration_sec": 2.0},
        ],
        "render_profile": {"video": {"model_id": "wan-fixture"}},
    }

    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))

    client = TestClient(production_app)
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "dry"})
    assert resp.status_code == 200
    data = resp.json()

    sample = _load_sample("production_agent_response.sample.json")
    normalized = json.loads(json.dumps(data))
    normalized["request_id"] = "__REQUEST_ID__"
    normalized["run_id"] = "__RUN_ID__"
    for step in normalized.get("steps") or []:
        step["start_time"] = "__START_TIME__"
        step["end_time"] = "__END_TIME__"
        step["duration_s"] = "__DURATION_S__"

    assert normalized == sample


def test_assemble_ffmpeg_response_matches_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deterministic_media_assets,
) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(assemble_app)
    payload = {
        "plan_id": "plan-123",
        "clips": [{"uri": str(deterministic_media_assets.video)}],
        "options": {"fixture_only": True},
    }

    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    sample = _load_sample("assemble_ffmpeg_response.sample.json")
    normalized = json.loads(json.dumps(data))
    normalized["artifact_uri"] = "__ARTIFACT_URI__"
    normalized["request_id"] = "__REQUEST_ID__"
    metadata = normalized.get("metadata")
    assert metadata is not None
    metadata["digest"] = "__DIGEST__"
    metadata["request_id"] = "__REQUEST_ID__"
    metadata["local_path"] = "__LOCAL_PATH__"

    assert normalized == sample


def test_images_sdxl_response_matches_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(images_app)
    payload = {
        "prompt": "cinematic dusk cityscape",
        "plan_id": "plan-123",
        "run_id": "run-456",
        "metadata": {"plan_id": "plan-123"},
        "seed": 42,
        "count": 1,
    }

    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    sample = _load_sample("images_sdxl_response.sample.json")
    normalized = json.loads(json.dumps(data))
    normalized["request_id"] = "__REQUEST_ID__"
    normalized["artifact_uri"] = "__ARTIFACT_URI__"

    artifacts = normalized.get("artifacts") or []
    assert len(artifacts) == 1
    artifacts[0]["artifact_uri"] = "__ARTIFACT_URI__"
    metadata = artifacts[0].get("metadata", {})
    metadata["request_id"] = "__REQUEST_ID__"
    metadata["local_path"] = "__LOCAL_PATH__"
    metadata["source_path"] = "__SOURCE_PATH__"
    metadata["modified_ts"] = "__MODIFIED_TS__"

    assert normalized == sample


def test_tts_chatterbox_response_matches_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(tts_app)
    payload = {
        "text": "Hello world from fixture",
        "voice_id": "emma",
        "language": "en-US",
        "plan_id": "plan-123",
        "run_id": "run-456",
        "step_id": "step-1",
    }

    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    sample = _load_sample("tts_chatterbox_response.sample.json")
    normalized = json.loads(json.dumps(data))
    normalized["artifact_uri"] = "__ARTIFACT_URI__"
    normalized["request_id"] = "__REQUEST_ID__"
    metadata = normalized.get("metadata")
    assert metadata is not None
    metadata["request_id"] = "__REQUEST_ID__"
    metadata["local_path"] = "__LOCAL_PATH__"
    engine_meta = metadata.get("engine_metadata")
    assert isinstance(engine_meta, dict)
    engine_meta["seed"] = "__SEED__"

    assert normalized == sample


def test_lipsync_wav2lip_response_matches_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deterministic_media_assets,
) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    face_b64 = base64.b64encode(deterministic_media_assets.video.read_bytes()).decode("ascii")
    audio_b64 = base64.b64encode(deterministic_media_assets.audio.read_bytes()).decode("ascii")

    client = TestClient(lipsync_app)
    payload = {
        "face": {"data_b64": face_b64},
        "audio": {"data_b64": audio_b64},
        "plan_id": "plan-123",
        "run_id": "run-456",
        "step_id": "step-1",
        "movie_title": "Test Film",
    }

    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    sample = _load_sample("lipsync_wav2lip_response.sample.json")
    normalized = json.loads(json.dumps(data))
    normalized["artifact_uri"] = "__ARTIFACT_URI__"
    normalized["request_id"] = "__REQUEST_ID__"
    metadata = normalized.get("metadata", {})
    metadata["face_video"] = "__FACE_PATH__"
    metadata["audio"] = "__AUDIO_PATH__"
    metadata["request_id"] = "__REQUEST_ID__"
    metadata["duration_s"] = "__DURATION__"

    assert normalized == sample


def test_videos_wan_response_matches_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(videos_app)
    payload = {
        "prompt": "A serene lake under moonlight",
        "plan_id": "plan-789",
        "run_id": "run-001",
        "step_id": "step-002",
        "seed": 12345,
        "num_frames": 32,
        "fps": 16,
        "width": 512,
        "height": 512,
        "metadata": {"shot_id": "shot-3"},
        "options": {"fixture_only": True, "guidance_scale": 4.5},
    }

    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    sample = _load_sample("videos_wan_response.sample.json")
    normalized = json.loads(json.dumps(data))
    normalized["artifact_uri"] = "__ARTIFACT_URI__"
    normalized["request_id"] = "__REQUEST_ID__"
    metadata = normalized.get("metadata")
    assert isinstance(metadata, dict)
    metadata["request_id"] = "__REQUEST_ID__"
    metadata["local_path"] = "__LOCAL_PATH__"

    assert normalized == sample

