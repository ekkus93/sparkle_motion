from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient
import pytest

from sparkle_motion import schema_registry
import sparkle_motion.function_tools.production_agent.entrypoint as production_entrypoint
from sparkle_motion.function_tools.production_agent.entrypoint import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_plan() -> dict:
    return {
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


def test_invoke_dry_mode(client: TestClient, sample_plan: dict) -> None:
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "dry"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["simulation_report"]
    assert data["artifact_uris"] == []
    assert data["run_id"]
    assert data["schema_uri"] == schema_registry.movie_plan_schema().uri


def test_invoke_run_mode_writes_artifact(
    client: TestClient,
    sample_plan: dict,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "run"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifact_uris"]
    assert data["steps"]
    assert data["run_id"]
    assert data["schema_uri"] == schema_registry.movie_plan_schema().uri


def test_status_and_control_endpoints(client: TestClient, sample_plan: dict, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_TTS", "1")
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "run", "qa_mode": "skip"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    status_resp = client.get("/status", params={"run_id": run_id})
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["run_id"] == run_id
    assert status_data["steps"], "status should include recorded steps"
    assert status_data["qa_mode"] == "skip"
    assert status_data["metadata"].get("qa_mode") == "skip"
    assert status_data["render_profile"]["video"]["model_id"] == "wan-fixture"
    assert status_data["timeline"] == status_data["log"]
    assert status_data["timeline"], "timeline history should be populated"
    assert status_data["timeline"][0]["qa_mode"] == "skip"

    artifacts_resp = client.get("/artifacts", params={"run_id": run_id})
    assert artifacts_resp.status_code == 200
    artifacts_data = artifacts_resp.json()
    assert artifacts_data["run_id"] == run_id
    assert isinstance(artifacts_data["artifacts"], list)
    assert all("artifact_uri" in item and "stage_id" in item for item in artifacts_data["artifacts"])
    dialogue_entries = [item for item in artifacts_data["artifacts"] if item["artifact_type"] == "tts_timeline_audio"]
    assert dialogue_entries, "dialogue audio manifest entries should be exposed"
    assert dialogue_entries[0]["stage_id"] == "dialogue_audio"
    assert dialogue_entries[0]["name"].endswith("tts_timeline.wav")

    pause_resp = client.post("/control/pause", json={"run_id": run_id})
    assert pause_resp.status_code == 200
    resume_resp = client.post("/control/resume", json={"run_id": run_id})
    assert resume_resp.status_code == 200
    stop_resp = client.post("/control/stop", json={"run_id": run_id})
    assert stop_resp.status_code == 200


def test_artifacts_endpoint_returns_video_final_manifest(
    client: TestClient,
    sample_plan: dict,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "run"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    artifacts_resp = client.get("/artifacts", params={"run_id": run_id, "stage": "qa_publish"})
    assert artifacts_resp.status_code == 200
    data = artifacts_resp.json()
    entries = data["artifacts"]
    assert entries, "video_final manifest should be present"
    final_entry = entries[0]
    assert final_entry["artifact_type"] == "video_final"
    assert final_entry["stage_id"] == "qa_publish"
    assert isinstance(final_entry["qa_passed"], bool)
    assert isinstance(final_entry["playback_ready"], bool)
    assert final_entry["checksum_sha256"]
    if final_entry["storage_hint"] == "adk":
        assert final_entry["download_url"], "download_url required for adk storage"
    else:
        assert final_entry["local_path"], "local_path required for local storage"


def test_artifacts_endpoint_rejects_invalid_video_final_manifest(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get_artifacts(run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "stage": "qa_publish",
                "artifact_type": "video_final",
                "name": "video_final.mp4",
                "artifact_uri": "artifact://sparkle/video_final",
                "media_type": "video/mp4",
                "local_path": "/tmp/video_final.mp4",
                "storage_hint": "local",
                "mime_type": "video/mp4",
                "size_bytes": 0,  # invalid
                "duration_s": 1.0,
                "frame_rate": 24.0,
                "resolution_px": "1280x720",
                "checksum_sha256": "a" * 64,
                "qa_report_uri": "artifact://sparkle/qa_report",
                "qa_passed": True,
                "qa_mode": "full",
                "playback_ready": True,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(production_entrypoint.registry, "get_artifacts", _fake_get_artifacts)
    resp = client.get("/artifacts", params={"run_id": "invalid-run", "stage": "qa_publish"})
    assert resp.status_code == 500
    assert "size_bytes" in resp.json()["detail"]


def test_artifacts_endpoint_errors_when_video_final_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get_artifacts(run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "stage": "qa_publish",
                "artifact_type": "qa_report",
                "name": "qa_report.json",
                "artifact_uri": "artifact://sparkle/qa_report",
                "media_type": "application/json",
                "local_path": "/tmp/qa_report.json",
                "storage_hint": "local",
                "mime_type": "application/json",
                "size_bytes": 128,
                "duration_s": None,
                "frame_rate": None,
                "resolution_px": None,
                "checksum_sha256": None,
                "qa_report_uri": None,
                "qa_passed": True,
                "qa_mode": "full",
                "playback_ready": True,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(production_entrypoint.registry, "get_artifacts", _fake_get_artifacts)
    resp = client.get("/artifacts", params={"run_id": "missing-final", "stage": "qa_publish"})
    assert resp.status_code == 409
    assert "video_final" in resp.json()["detail"]