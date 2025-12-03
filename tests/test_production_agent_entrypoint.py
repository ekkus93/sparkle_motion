from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient
import pytest

from sparkle_motion import adk_helpers, schema_registry
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


def test_invoke_plan_uri_with_validated_plan(client: TestClient, sample_plan: dict, tmp_path) -> None:
    artifact_path = tmp_path / "plan_artifact.json"
    artifact_payload = {
        "request": {"title": sample_plan["title"]},
        "validated_plan": sample_plan,
    }
    artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    resp = client.post(
        "/invoke",
        json={"plan_uri": f"file://{artifact_path}", "mode": "dry"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["run_id"]


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
    assert status_data["render_profile"]["video"]["model_id"] == "wan-fixture"
    assert status_data["timeline"] == status_data["log"]
    assert status_data["timeline"], "timeline history should be populated"

    artifacts_resp = client.get("/artifacts", params={"run_id": run_id})
    assert artifacts_resp.status_code == 200
    artifacts_data = artifacts_resp.json()
    assert artifacts_data["run_id"] == run_id
    assert isinstance(artifacts_data["artifacts"], list)
    assert isinstance(artifacts_data.get("stages"), list)
    assert artifacts_data["total_artifacts"] == len(artifacts_data["artifacts"]) == sum(section["count"] for section in artifacts_data["stages"])
    dialogue_stage = next(section for section in artifacts_data["stages"] if section["stage_id"] == "dialogue_audio")
    assert dialogue_stage["count"] == len(dialogue_stage["artifacts"])
    assert "preview" in dialogue_stage and dialogue_stage["preview"].get("audio")
    media_summary = dialogue_stage.get("media_summary", {})
    assert "audio" in media_summary
    assert media_summary["audio"]["count"] >= 1
    timeline_entries = [item for item in dialogue_stage["artifacts"] if item["artifact_type"] == "tts_timeline_audio"]
    assert timeline_entries, "dialogue audio manifest entries should be exposed"
    assert timeline_entries[0]["stage_id"] == "dialogue_audio"
    assert timeline_entries[0]["name"].endswith("tts_timeline.wav")

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

    artifacts_resp = client.get("/artifacts", params={"run_id": run_id, "stage": "finalize"})
    assert artifacts_resp.status_code == 200
    data = artifacts_resp.json()
    assert data["run_id"] == run_id
    assert len(data["stages"]) == 1
    entries = data["stages"][0]["artifacts"]
    assert entries, "video_final manifest should be present"
    assert data["artifacts"] == entries, "stage-filtered response should flatten to the same entries"
    final_entry = entries[0]
    assert final_entry["artifact_type"] == "video_final"
    assert final_entry["stage_id"] == "finalize"
    assert isinstance(final_entry["playback_ready"], bool)
    assert final_entry["checksum_sha256"]
    if final_entry["storage_hint"] == "adk":
        assert final_entry["download_url"], "download_url required for adk storage"
    else:
        assert final_entry["local_path"], "local_path required for local storage"
    stage_section = data["stages"][0]
    assert stage_section["preview"].get("video"), "video preview metadata should be available"


def test_artifacts_endpoint_stage_filter_isolated(
    client: TestClient,
    sample_plan: dict,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "run"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    artifacts_resp = client.get("/artifacts", params={"run_id": run_id, "stage": "dialogue_audio"})
    assert artifacts_resp.status_code == 200
    data = artifacts_resp.json()
    assert len(data["stages"]) == 1
    stage_section = data["stages"][0]
    assert stage_section["stage_id"] == "dialogue_audio"
    assert data["artifacts"] == stage_section["artifacts"]
    assert stage_section["count"] == len(stage_section["artifacts"])
    assert "size_bytes_total" in stage_section
    assert set(stage_section["preview"].keys()) == {"image", "audio", "video", "other"}


def test_artifacts_endpoint_rejects_invalid_video_final_manifest(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get_artifacts(run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "stage": "finalize",
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
                "playback_ready": True,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(production_entrypoint.registry, "get_artifacts", _fake_get_artifacts)
    resp = client.get("/artifacts", params={"run_id": "invalid-run", "stage": "finalize"})
    assert resp.status_code == 500
    assert "size_bytes" in resp.json()["detail"]


def test_artifacts_endpoint_rejects_missing_download_for_adk(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get_artifacts(run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "stage": "finalize",
                "artifact_type": "video_final",
                "name": "video_final.mp4",
                "artifact_uri": "artifact://sparkle/video_final",
                "media_type": "video/mp4",
                "local_path": "/tmp/video_final.mp4",
                "storage_hint": "adk",
                "mime_type": "video/mp4",
                "size_bytes": 1024,
                "duration_s": 1.0,
                "frame_rate": 24.0,
                "resolution_px": "1280x720",
                "checksum_sha256": "a" * 64,
                "playback_ready": True,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(production_entrypoint.registry, "get_artifacts", _fake_get_artifacts)
    resp = client.get("/artifacts", params={"run_id": "adk-missing", "stage": "finalize"})
    assert resp.status_code == 500
    assert "download_url" in resp.json()["detail"]


def test_artifacts_endpoint_accepts_filesystem_video_final_manifest(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_artifacts(run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "stage": "finalize",
                "artifact_type": "video_final",
                "name": "video_final.mp4",
                "artifact_uri": "artifact+fs://foo",
                "media_type": "video/mp4",
                "local_path": "/tmp/video_final.mp4",
                "storage_hint": "filesystem",
                "mime_type": "video/mp4",
                "size_bytes": 2048,
                "duration_s": 1.5,
                "frame_rate": 24.0,
                "resolution_px": "1280x720",
                "checksum_sha256": "a" * 64,
                "playback_ready": True,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(production_entrypoint.registry, "get_artifacts", _fake_get_artifacts)
    resp = client.get("/artifacts", params={"run_id": "fs-run", "stage": "finalize"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifacts"], "filesystem manifests should be accepted"


def test_artifacts_endpoint_errors_when_video_final_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get_artifacts(run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {
                "stage": "finalize",
                "artifact_type": "stage_manifest",
                "name": "stage_manifest.json",
                "artifact_uri": "artifact://sparkle/stage_manifest",
                "media_type": "application/json",
                "local_path": "/tmp/qa_report.json",
                "storage_hint": "local",
                "mime_type": "application/json",
                "size_bytes": 128,
                "duration_s": None,
                "frame_rate": None,
                "resolution_px": None,
                "checksum_sha256": None,
                "playback_ready": True,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(production_entrypoint.registry, "get_artifacts", _fake_get_artifacts)
    resp = client.get("/artifacts", params={"run_id": "missing-final", "stage": "finalize"})
    assert resp.status_code == 409
    assert "video_final" in resp.json()["detail"]


def test_artifacts_endpoint_reads_filesystem_store_when_registry_empty(
    client: TestClient,
    sample_plan: dict,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs_root = tmp_path / "fs"
    fs_index = fs_root / "index.db"
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("ARTIFACTS_BACKEND", "filesystem")
    monkeypatch.setenv("ARTIFACTS_FS_ROOT", str(fs_root))
    monkeypatch.setenv("ARTIFACTS_FS_INDEX", str(fs_index))
    monkeypatch.setenv("ARTIFACTS_FS_ALLOW_INSECURE", "1")
    adk_helpers._reset_filesystem_store_for_tests()

    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "run"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    registry = production_entrypoint.registry
    with registry._lock:  # type: ignore[attr-defined]
        state = registry._runs.get(run_id)  # type: ignore[attr-defined]
        assert state is not None
        state.artifacts.clear()

    artifacts_resp = client.get("/artifacts", params={"run_id": run_id, "stage": "plan_intake"})
    assert artifacts_resp.status_code == 200
    artifacts_data = artifacts_resp.json()
    assert artifacts_data["artifacts"], "filesystem artifacts should materialize even when registry memory is empty"

    status_resp = client.get("/status", params={"run_id": run_id})
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["artifact_counts"].get("plan_intake", 0) >= 1


def test_artifacts_endpoint_validates_video_final_on_aggregate(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_grouped(run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "finalize": [
                {
                    "run_id": run_id,
                    "stage": "finalize",
                    "artifact_type": "stage_manifest",
                    "name": "stage_manifest.json",
                    "artifact_uri": "artifact://sparkle/stage_manifest",
                    "media_type": "application/json",
                    "local_path": "/tmp/qa_report.json",
                    "storage_hint": "local",
                    "mime_type": "application/json",
                    "size_bytes": 128,
                    "playback_ready": True,
                    "metadata": {},
                }
            ]
        }

    monkeypatch.setattr(production_entrypoint.registry, "get_artifacts_by_stage", _fake_grouped)
    resp = client.get("/artifacts", params={"run_id": "aggregate-missing"})
    assert resp.status_code == 409
    assert "video_final" in resp.json()["detail"]