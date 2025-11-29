from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pytest

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


def test_progress_callback_invoked(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    seen: List[StepExecutionRecord] = []

    def on_progress(record: StepExecutionRecord) -> None:
        seen.append(record)

    execute_plan(sample_plan, mode="run", progress_callback=on_progress)
    assert seen
    assert all(isinstance(record, StepExecutionRecord) for record in seen)


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
