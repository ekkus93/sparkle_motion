from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from sparkle_motion.function_tools.production_agent.entrypoint import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_plan() -> dict:
    return {
        "title": "Test Film",
        "metadata": {"plan_id": "plan-entrypoint"},
        "shots": [
            {
                "id": "shot-1",
                "duration_sec": 2,
                "visual_description": "A hero poses",
                "start_frame_prompt": "hero start",
                "end_frame_prompt": "hero end",
                "dialogue": [],
                "is_talking_closeup": False,
            }
        ],
    }


def test_invoke_dry_mode(client: TestClient, sample_plan: dict) -> None:
    resp = client.post("/invoke", json={"plan": sample_plan, "mode": "dry"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["simulation_report"]
    assert data["artifact_uris"] == []


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