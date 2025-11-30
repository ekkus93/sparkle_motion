from __future__ import annotations
import json
from pathlib import Path

from fastapi.testclient import TestClient

from sparkle_motion import schema_registry
from sparkle_motion.schemas import MoviePlan
from sparkle_motion.function_tools.script_agent import entrypoint as script_entrypoint
from sparkle_motion.function_tools.script_agent.entrypoint import app


def test_ready_and_invoke(tmp_path, monkeypatch):
    # Ensure no artificial model load delay and deterministic output
    monkeypatch.setenv("MODEL_LOAD_DELAY", "0")
    monkeypatch.setenv("DETERMINISTIC", "1")
    # Use a temporary artifacts dir
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(app)

    # Startup event should have set ready == True
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json().get("ready") is True

    # Provide a minimal non-empty shots list to satisfy validation
    payload = {"title": "Hardened Test", "shots": [{"id": "s1", "desc": "minimal shot"}]}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["artifact_uri"].startswith("file://") or data["artifact_uri"].startswith("artifact://")
    assert data["schema_uri"] == schema_registry.movie_plan_schema().uri


def test_invoke_rebuilds_base_images(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LOAD_DELAY", "0")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    canonical_payload = {
        "title": "Canonical Test",
        "base_images": [
            {"id": "legacy_a", "prompt": "legacy start"},
            {"id": "legacy_b", "prompt": "legacy bridge"},
            {"id": "legacy_c", "prompt": "legacy end"},
        ],
        "shots": [
            {
                "id": "shot-1",
                "duration_sec": 2,
                "visual_description": "Scene one",
                "start_base_image_id": "legacy_a",
                "end_base_image_id": "legacy_b",
            },
            {
                "id": "shot-2",
                "duration_sec": 3,
                "visual_description": "Scene two",
                "start_base_image_id": "legacy_b",
                "end_base_image_id": "legacy_c",
            },
        ],
        "dialogue_timeline": [
            {"type": "silence", "start_time_sec": 0.0, "duration_sec": 5.0},
        ],
        "render_profile": {"video": {"model_id": "wan-fixture"}, "metadata": {}},
    }
    canonical_plan = MoviePlan.model_validate(canonical_payload)
    monkeypatch.setattr(script_entrypoint, "_generate_movie_plan", lambda *args, **kwargs: canonical_plan)

    client = TestClient(app)
    resp = client.post("/invoke", json={"prompt": "ignored"})
    assert resp.status_code == 200
    artifact_uri = resp.json()["artifact_uri"]
    assert artifact_uri.startswith("file://")
    artifact_path = Path(artifact_uri.replace("file://", ""))
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    plan = saved["validated_plan"]
    base_images = plan["base_images"]
    assert [img["id"] for img in base_images] == ["frame_000", "frame_001", "frame_002"]
    assert [img["prompt"] for img in base_images] == [
        "legacy start",
        "legacy bridge",
        "legacy end",
    ]
    assert plan["shots"][0]["start_base_image_id"] == "frame_000"
    assert plan["shots"][0]["end_base_image_id"] == "frame_001"
    assert plan["shots"][1]["start_base_image_id"] == "frame_001"
    assert plan["shots"][1]["end_base_image_id"] == "frame_002"
