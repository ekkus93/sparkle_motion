from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import pytest

import sparkle_motion.production_agent as production_agent

from sparkle_motion.images_agent import RateLimitExceeded, RateLimitQueued
from sparkle_motion.production_agent import (
    ProductionResult,
    StepExecutionRecord,
    StepQueuedError,
    StepRateLimitExceededError,
    execute_plan,
)
from sparkle_motion.ratelimit import RateLimitDecision
from sparkle_motion.schemas import DialogueLine, MoviePlan, ShotSpec


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMOKE_ADAPTERS", raising=False)
    monkeypatch.delenv("SMOKE_TTS", raising=False)
    monkeypatch.delenv("SMOKE_LIPSYNC", raising=False)


@pytest.fixture(autouse=True)
def _stub_videos_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_render(
        start_frames: Iterable[Any],
        end_frames: Iterable[Any],
        prompt: str,
        opts: Optional[Mapping[str, Any]] = None,
        *,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        _adapter: Optional[Any] = None,
    ) -> Dict[str, Any]:
        options = dict(opts or {})
        output_path_value = options.get("output_path")
        if output_path_value:
            target = Path(output_path_value)
        else:
            base = Path(options.get("output_dir") or os.getcwd())
            base.mkdir(parents=True, exist_ok=True)
            target = base / f"{options.get('shot_id', 'clip')}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"FAKE_VIDEO")
        if on_progress:
            event = {
                "plan_id": options.get("plan_id"),
                "step_id": options.get("step_id"),
                "chunk_index": 0,
                "progress": 0.5,
            }
            on_progress(event)
        return {
            "uri": f"file://{target.name}",
            "metadata": {"source_path": str(target), "prompt": prompt, "frames": len(list(start_frames))},
        }

    monkeypatch.setattr("sparkle_motion.videos_agent.render_video", _fake_render)


@pytest.fixture
def sample_plan(tmp_path: Path) -> MoviePlan:
    return MoviePlan(
        title="Test Film",
        metadata={"plan_id": "plan-123"},
        shots=[
            ShotSpec(
                id="shot-1",
                visual_description="A hero poses",
                duration_sec=3,
                dialogue=[
                    DialogueLine(character_id="hero", text="Hello there!"),
                ],
                start_frame_prompt="hero start",
                end_frame_prompt="hero end",
                is_talking_closeup=True,
            ),
            ShotSpec(
                id="shot-2",
                visual_description="Establishing shot",
                duration_sec=5,
                dialogue=[],
                start_frame_prompt="establish",
                end_frame_prompt="sunset",
                is_talking_closeup=False,
            ),
        ],
    )


@pytest.mark.parametrize("mode", ["dry", "run"])
def test_execute_plan_modes(tmp_path: Path, sample_plan: MoviePlan, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    if mode == "run":
        monkeypatch.setenv("SMOKE_ADAPTERS", "1")
        monkeypatch.setenv("SMOKE_TTS", "1")
        monkeypatch.setenv("SMOKE_LIPSYNC", "1")
    result = execute_plan(sample_plan, mode=mode)
    assert isinstance(result, ProductionResult)
    if mode == "dry":
        assert result.steps == []
        assert result.simulation_report is not None
    else:
        assert len(result.steps) >= 3
        assert result.simulation_report is None
        assert "production_agent_final_movie" in result[0]["uri"]


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"SMOKE_ADAPTERS": "1"},
        {"SMOKE_TTS": "true"},
        {"SMOKE_LIPSYNC": "yes"},
    ],
)
def test_gate_flags(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path, flags: Dict[str, str]) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    for key, value in flags.items():
        monkeypatch.setenv(key, value)
    result = execute_plan(sample_plan, mode="run")
    statuses = {record.step_id.split(":")[-1]: record.status for record in result.steps}
    adapters_enabled = os.getenv("SMOKE_ADAPTERS") not in {None, "", "0", "false"}
    expected = "succeeded" if adapters_enabled else "simulated"
    assert statuses["images"] == expected
    assert statuses["video"] == expected


class FakeError(Exception):
    pass


def test_retry_behavior(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    attempts: List[int] = []

    def fail_once() -> Path:
        attempts.append(1)
        if len(attempts) < 2:
            from sparkle_motion.production_agent import StepTransientError

            raise StepTransientError("retry me")
        return tmp_path / "ok.bin"

    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_ADAPTERS", "1")
    monkeypatch.setattr("sparkle_motion.production_agent._render_frames", lambda *args, **kwargs: fail_once())
    result = execute_plan(sample_plan, mode="run")
    image_records = [r for r in result.steps if r.step_type == "images"]
    assert image_records[0].attempts == 2


def test_render_video_clip_passes_metadata(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    shot = sample_plan.shots[0]
    plan_id = "plan-meta"
    run_id = "run-456"
    monkeypatch.setenv("VIDEOS_AGENT_DEFAULT_FPS", "12")
    expected_path = tmp_path / "video" / f"{shot.id}.mp4"

    captured: Dict[str, Any] = {}

    def _capture(
        start_frames: Iterable[bytes],
        end_frames: Iterable[bytes],
        prompt: str,
        opts: Optional[Mapping[str, Any]] = None,
        *,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        _adapter: Optional[Any] = None,
    ) -> Dict[str, Any]:
        captured["start_frames"] = list(start_frames)
        captured["end_frames"] = list(end_frames)
        captured["prompt"] = prompt
        captured["opts"] = dict(opts or {})
        return {"uri": f"file://{expected_path}", "metadata": {"source_path": str(expected_path)}}

    monkeypatch.setattr("sparkle_motion.videos_agent.render_video", _capture)

    result_path = production_agent._render_video_clip(shot, tmp_path, plan_id, run_id)

    assert result_path == expected_path
    assert captured["start_frames"][0] == shot.start_frame_prompt.encode("utf-8")
    assert captured["end_frames"][0] == shot.end_frame_prompt.encode("utf-8")
    assert captured["prompt"] == shot.visual_description
    opts = captured["opts"]
    assert opts["plan_id"] == plan_id
    assert opts["run_id"] == run_id
    assert opts["shot_id"] == shot.id
    assert opts["output_path"].endswith(f"{shot.id}.mp4")
    expected_frames = int(round(shot.duration_sec * 12))
    assert opts["num_frames"] == expected_frames


def test_progress_callback_invoked(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    seen: List[StepExecutionRecord] = []

    def on_progress(record: StepExecutionRecord) -> None:
        seen.append(record)

    execute_plan(sample_plan, mode="run", progress_callback=on_progress)
    assert seen
    assert all(isinstance(record, StepExecutionRecord) for record in seen)


def test_video_progress_forwarded(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_ADAPTERS", "1")
    seen: List[StepExecutionRecord] = []

    def on_progress(record: StepExecutionRecord) -> None:
        seen.append(record)

    execute_plan(sample_plan, mode="run", progress_callback=on_progress)

    progress_records = [
        record
        for record in seen
        if record.step_type == "video" and record.status == "running" and "videos_agent_progress" in record.meta
    ]
    assert progress_records, "Expected chunk progress records from videos_agent"
    event_meta = progress_records[0].meta["videos_agent_progress"]
    assert event_meta["chunk_index"] == 0
    assert event_meta["progress"] == 0.5


def _enable_full_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_ADAPTERS", "1")
    monkeypatch.setenv("SMOKE_TTS", "1")
    monkeypatch.setenv("SMOKE_LIPSYNC", "1")


def test_rate_limit_queue_sets_record(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    decision = RateLimitDecision(status="queued", tokens=2, retry_after_s=3.0, eta_epoch_s=10.0, ttl_deadline_s=20.0, reason="rate")

    def _raise_queue(*_: object, **__: object) -> Path:
        raise RateLimitQueued("queued", decision)

    monkeypatch.setattr("sparkle_motion.production_agent._render_frames", _raise_queue)

    with pytest.raises(StepQueuedError) as excinfo:
        execute_plan(sample_plan, mode="run")

    record = excinfo.value.record
    assert record.status == "queued"
    assert record.step_id.endswith(":images")
    rl = record.meta.get("rate_limit")
    assert rl and rl["status"] == "queued"


def test_rate_limit_exceeded_sets_record(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    decision = RateLimitDecision(status="rejected", tokens=2, retry_after_s=0.5, eta_epoch_s=5.0, ttl_deadline_s=15.0, reason="maxed")

    def _raise_exceeded(*_: object, **__: object) -> Path:
        raise RateLimitExceeded("rejected", decision)

    monkeypatch.setattr("sparkle_motion.production_agent._render_frames", _raise_exceeded)

    with pytest.raises(StepRateLimitExceededError) as excinfo:
        execute_plan(sample_plan, mode="run")

    record = excinfo.value.record
    assert record.status == "failed"
    rl = record.meta.get("rate_limit")
    assert rl and rl["status"] == "rejected"
