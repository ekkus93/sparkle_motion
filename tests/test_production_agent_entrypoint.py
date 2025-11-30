from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from sparkle_motion import schema_registry
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
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "run"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    status_resp = client.get("/status", params={"run_id": run_id})
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["run_id"] == run_id
    assert status_data["steps"], "status should include recorded steps"

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