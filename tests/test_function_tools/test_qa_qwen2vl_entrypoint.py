from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Dict

import pytest
from fastapi.testclient import TestClient

from sparkle_motion.function_tools.qa_qwen2vl import adapter
from sparkle_motion.function_tools.qa_qwen2vl import entrypoint as qa_entrypoint
from sparkle_motion.function_tools.qa_qwen2vl.entrypoint import make_app
from sparkle_motion.schemas import QAReport, QAReportPerShot

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def test_health_endpoint():
    client = TestClient(make_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


@pytest.fixture()
def qa_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    report = QAReport(
        movie_title="unit",
        per_shot=[
            QAReportPerShot(
                shot_id="frame_0000",
                prompt_match=1.0,
                finger_issues=False,
                finger_issue_ratio=0.0,
            )
        ],
        summary="All frames good",
        decision="approve",
        issues_found=0,
        aggregate_prompt_match=1.0,
    )
    artifact_path = tmp_path / "qa_report.json"
    artifact_path.write_text("{}", encoding="utf-8")
    calls: Dict[str, Any] = {}

    def _fake_inspect(frames, prompts, *, opts=None):
        calls["frames"] = list(frames)
        calls["prompts"] = list(prompts)
        calls["opts"] = dict(opts or {})
        return adapter.QAInspectionResult(
            report=report,
            artifact_path=artifact_path,
            artifact_uri="file://stub/qa_report.json",
            metadata={"engine": "stub"},
            decision=report.decision or "pending",
            human_task_id=None,
        )

    monkeypatch.setattr(adapter, "inspect_frames", _fake_inspect)

    app = make_app()
    with TestClient(app) as client:
        yield client, calls


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_invoke_happy_path_returns_report(qa_client, deterministic_media_assets: MediaAssets):
    client, calls = qa_client
    payload = {
        "prompt": "spotlight scene",
        "frames": [
            {
                "id": "frame1",
                "data_b64": _b64(deterministic_media_assets.image.read_bytes()),
            }
        ],
        "metadata": {"plan_id": "plan-123"},
    }
    response = client.post("/invoke", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["decision"] == "approve"
    assert data["report"]["summary"] == "All frames good"
    assert data["artifact_uri"].startswith("file://")
    assert calls["prompts"] == ["spotlight scene"]
    assert calls["frames"] == [deterministic_media_assets.image.read_bytes()]
    assert calls["opts"]["frame_ids"] == ["frame1"]


def test_invoke_requires_prompt(qa_client):
    client, _ = qa_client
    payload = {
        "frames": [
            {
                "id": "frame1",
                "data_b64": _b64(b"missing"),
            }
        ]
    }
    response = client.post("/invoke", json=payload)
    assert response.status_code == 400


def test_invoke_rejects_invalid_base64(qa_client):
    client, _ = qa_client
    payload = {
        "prompt": "ok",
        "frames": [
            {
                "id": "frame1",
                "data_b64": "@@ not base64 @@",
            }
        ],
    }
    response = client.post("/invoke", json=payload)
    assert response.status_code == 400
    assert "data_base64" in response.json()["detail"]


def test_invoke_rejects_missing_file(qa_client, tmp_path):
    client, _ = qa_client
    nonexistent = tmp_path / "missing.png"
    payload = {
        "prompt": "ok",
        "frames": [
            {
                "id": "frame1",
                "uri": nonexistent.as_uri(),
            }
        ],
    }
    response = client.post("/invoke", json=payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]


def test_invoke_respects_download_limits(qa_client):
    client, _ = qa_client
    oversized = b"x" * 2048
    payload = {
        "prompt": "ok",
        "frames": [
            {
                "id": "frame1",
                "data_b64": _b64(oversized),
            }
        ],
        "options": {"max_download_bytes": 1024},
    }
    response = client.post("/invoke", json=payload)
    assert response.status_code == 400
    assert "max_download_bytes" in response.json()["detail"]


def test_invoke_fetches_http_uri(monkeypatch, qa_client):
    client, calls = qa_client
    fetch_calls: Dict[str, Any] = {}

    def _fake_fetch(uri: str, idx: int, *, max_bytes: int, timeout_s: float) -> bytes:
        fetch_calls["uri"] = uri
        fetch_calls["idx"] = idx
        fetch_calls["max_bytes"] = max_bytes
        fetch_calls["timeout_s"] = timeout_s
        return b"http-bytes"

    monkeypatch.setattr(qa_entrypoint, "_fetch_remote_bytes", _fake_fetch)

    payload = {
        "prompt": "remote",
        "frames": [
            {
                "id": "frame-http",
                "uri": "https://example.com/frame.png",
            }
        ],
    }
    response = client.post("/invoke", json=payload)
    assert response.status_code == 200
    assert calls["frames"] == [b"http-bytes"]
    assert calls["opts"]["frame_ids"] == ["frame-http"]
    assert fetch_calls["uri"].startswith("https://example.com")

