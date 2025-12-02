from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from sparkle_motion import adk_helpers
from sparkle_motion.function_tools.qa_qwen2vl import adapter


@pytest.fixture(autouse=True)
def _silence_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter.telemetry, "emit_event", lambda *_, **__: None)
    monkeypatch.setattr(adk_helpers, "write_memory_event", lambda **__: None)


@pytest.fixture()
def helper_backend(tmp_path: Path) -> Iterator[Dict[str, Any]]:
    published: List[Dict[str, Any]] = []
    human_requests: List[Dict[str, Any]] = []

    def _publisher(**kwargs: Any) -> Dict[str, Any]:
        published.append(kwargs)
        return {
            "uri": f"artifact://{kwargs['local_path'].name}",
            "storage": "local",
            "metadata": kwargs["metadata"],
            "run_id": kwargs.get("run_id"),
        }

    def _human_request(*, payload: Dict[str, Any], dry_run: bool = False) -> str:
        human_requests.append({"payload": payload, "dry_run": dry_run})
        return f"task-{len(human_requests):04d}"

    backend = adk_helpers.HelperBackend(publish=_publisher, request_human_input=_human_request)
    with adk_helpers.set_backend(backend):
        yield {"published": published, "requests": human_requests, "artifacts_dir": tmp_path}


def _force_real_engine(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    monkeypatch.setattr(adapter, "_decide_engine", lambda *_: adapter.EngineDecision(True, "forced-test"))

    def _fake_run(*_: Any, **__: Any) -> tuple[list[str], dict[str, Any]]:
        return [response], {"path": "test"}

    monkeypatch.setattr(adapter, "_run_qwen", _fake_run)


def test_inspect_frames_parses_structured_response(monkeypatch: pytest.MonkeyPatch, helper_backend: Dict[str, Any]) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(helper_backend["artifacts_dir"]))
    response = json.dumps(
        {
            "prompt_match": 0.91,
            "finger_issues": True,
            "finger_issue_ratio": 0.62,
            "artifact_notes": ["edge glow"],
            "missing_audio_detected": False,
        }
    )
    _force_real_engine(monkeypatch, response)

    result = adapter.inspect_frames([b"frame-bytes"], ["A hero shot"], opts={"plan_id": "plan-qa", "run_id": "run-qa"})

    assert result.metadata["engine"] == "qwen2vl"
    assert result.decision == "regenerate"
    shot = result.report.per_shot[0]
    assert shot.prompt_match == pytest.approx(0.91)
    assert shot.finger_issues is True
    assert shot.finger_issue_ratio == pytest.approx(0.62, rel=1e-5)
    assert shot.artifact_notes == ["edge glow"]

    published = helper_backend["published"]
    assert published and published[0]["metadata"]["decision"] == "regenerate"
    assert adk_helpers.is_artifact_uri(result.metadata["artifact_uri"])


def test_structured_payload_inside_markdown_escalates(monkeypatch: pytest.MonkeyPatch, helper_backend: Dict[str, Any]) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(helper_backend["artifacts_dir"]))
    response = """
    model analysis
    ```json
    {"prompt_match_score": 0.35, "issues": ["possible violence"], "safety_violation": true}
    ```
    """.strip()
    _force_real_engine(monkeypatch, response)

    result = adapter.inspect_frames([b"frame"], ["Battle"], opts={"plan_id": "plan", "run_id": "run"})

    assert result.decision == "escalate"
    assert result.report.per_shot[0].safety_violation is True
    requests = helper_backend["requests"]
    assert requests and requests[0]["payload"]["reason"].lower().startswith("escalation")


def test_frame_ids_and_metadata_snapshot(monkeypatch: pytest.MonkeyPatch, helper_backend: Dict[str, Any]) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(helper_backend["artifacts_dir"]))
    opts = {"frame_ids": ["shot-1", "shot-2"], "fixture_seed": 123, "plan_id": "plan", "metadata": {"foo": "bar"}}

    result = adapter.inspect_frames([b"a", b"b"], ["P1", "P2"], opts=opts)

    assert [shot.shot_id for shot in result.report.per_shot] == ["shot-1", "shot-2"]
    assert result.metadata["frame_ids"] == ["shot-1", "shot-2"]
    assert result.metadata["options_snapshot"]["fixture_seed"] == 123
    assert result.metadata["engine_details"]["fixture_seed"] == 123


def test_real_engine_failure_records_fallback(monkeypatch: pytest.MonkeyPatch, helper_backend: Dict[str, Any]) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(helper_backend["artifacts_dir"]))

    def _boom(*_: Any, **__: Any) -> tuple[list[str], dict[str, Any]]:
        raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "_run_qwen", _boom)
    monkeypatch.setattr(adapter, "_decide_engine", lambda *_: adapter.EngineDecision(True, "forced"))

    result = adapter.inspect_frames([b"a"], ["prompt"], opts={"plan_id": "plan"})

    assert result.metadata["engine"] == "qwen_fixture"
    assert result.metadata["fallback_reason"].startswith("real_engine_error:RuntimeError")