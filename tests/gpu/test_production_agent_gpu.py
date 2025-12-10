from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Dict, List

import pytest

from sparkle_motion import production_agent, queue_runner
from sparkle_motion.function_tools.production_agent import entrypoint as production_entrypoint
from sparkle_motion.function_tools.production_agent.models import ProductionAgentRequest
from sparkle_motion.production_agent import StepExecutionError, StepTransientError
from sparkle_motion.run_registry import RunRegistry, get_run_registry
from sparkle_motion.ratelimit import RateLimitDecision
from sparkle_motion.schemas import (
    BaseImageSpec,
    CharacterSpec,
    DialogueLine,
    DialogueTimelineDialogue,
    DialogueTimelineSilence,
    MoviePlan,
    RenderProfile,
    RenderProfileVideo,
    ShotSpec,
)

from . import helpers

pytestmark = pytest.mark.gpu


def _timeline_for_shots(shots: List[ShotSpec]) -> List[DialogueTimelineDialogue | DialogueTimelineSilence]:
    timeline: List[DialogueTimelineDialogue | DialogueTimelineSilence] = []
    cursor = 0.0
    for shot in shots:
        spoken = next((line for line in shot.dialogue if line.text.strip()), None)
        if spoken:
            timeline.append(
                DialogueTimelineDialogue(
                    character_id=spoken.character_id,
                    text=spoken.text,
                    start_time_sec=cursor,
                    duration_sec=shot.duration_sec,
                )
            )
        else:
            timeline.append(
                DialogueTimelineSilence(
                    start_time_sec=cursor,
                    duration_sec=shot.duration_sec,
                )
            )
        cursor += shot.duration_sec
    return timeline


def _build_gpu_plan() -> MoviePlan:
    shots = [
        ShotSpec(
            id="shot-1",
            visual_description="Close-up of the explorer securing their helmet inside the rover.",
            duration_sec=3.5,
            dialogue=[DialogueLine(character_id="hero", text="Systems check complete.")],
            start_base_image_id="frame_000",
            end_base_image_id="frame_001",
            motion_prompt="Slow handheld push-in",
            is_talking_closeup=True,
        ),
        ShotSpec(
            id="shot-2",
            visual_description="Wide tracking shot of the rover cresting the canyon ridge at sunrise.",
            duration_sec=4.5,
            dialogue=[],
            start_base_image_id="frame_001",
            end_base_image_id="frame_002",
            motion_prompt="Stabilized drone pan",
            is_talking_closeup=False,
        ),
    ]
    base_images = [
        BaseImageSpec(id="frame_000", prompt="Interior rover cockpit, astronaut adjusting controls"),
        BaseImageSpec(id="frame_001", prompt="Rover doorway opening to Martian canyon"),
        BaseImageSpec(id="frame_002", prompt="Wide Martian canyon sunrise, rover silhouette"),
    ]
    plan_id = "gpu-production-agent"
    return MoviePlan(
        title="GPU Production Agent Demo",
        metadata={"plan_id": plan_id},
        characters=[CharacterSpec(id="hero", name="Avery")],
        base_images=base_images,
        shots=shots,
        dialogue_timeline=_timeline_for_shots(shots),
        render_profile=RenderProfile(video=RenderProfileVideo(model_id="wan2.1/flf2v"), metadata={}),
    )


def _configure_runtime(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: Path,
    *,
    plan: MoviePlan,
    run_id: str,
    resume: bool = False,
) -> tuple[RunRegistry, Path, Path, str]:
    helpers.ensure_real_adapter(
        monkeypatch,
        flags=[
            "SMOKE_ADAPTERS",
            "SMOKE_IMAGES",
            "SMOKE_VIDEOS",
            "SMOKE_TTS",
            "SMOKE_LIPSYNC",
            "SMOKE_ASSEMBLE",
        ],
        disable_keys=[
            "ADK_USE_FIXTURE",
            "IMAGES_SDXL_FIXTURE_ONLY",
            "VIDEOS_WAN_FIXTURE_ONLY",
            "LIPSYNC_WAV2LIP_FIXTURE_ONLY",
            "ASSEMBLE_FFMPEG_FIXTURE_ONLY",
        ],
    )

    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"
    helpers.set_env(
        monkeypatch,
        {
            "IMAGES_SDXL_FIXTURE_ONLY": "0",
            "VIDEOS_WAN_FIXTURE_ONLY": "0",
            "LIPSYNC_WAV2LIP_FIXTURE_ONLY": "0",
            "ASSEMBLE_FFMPEG_FIXTURE_ONLY": "0",
            "ARTIFACTS_DIR": str(artifacts_root),
            "SPARKLE_LOCAL_RUNS_ROOT": str(runs_root),
            "SPARKLE_RUN_ID": run_id,
        },
    )

    plan_id = plan.metadata.get("plan_id", plan.title)
    registry = get_run_registry()
    if not resume:
        registry.discard_run(run_id)
        registry.start_run(run_id=run_id, plan_id=plan_id, plan_title=plan.title, mode="run")
    return registry, runs_root, artifacts_root, plan_id


@pytest.mark.gpu
def test_production_agent_full_run_real_adapters(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    plan = _build_gpu_plan()
    run_id = "gpu-production-agent-run"
    registry, runs_root, _, plan_id = _configure_runtime(monkeypatch, tmp_path, plan=plan, run_id=run_id)

    result = production_agent.execute_plan(plan, mode="run", run_id=run_id)

    assert isinstance(result, production_agent.ProductionResult)
    finalize_record = next(record for record in result.steps if record.step_type == "finalize")
    assert finalize_record.status == "succeeded"

    finalize_entries = registry.get_artifacts(run_id, stage="finalize")
    video_entry = next(entry for entry in finalize_entries if entry["artifact_type"] == "video_final")
    video_path = Path(video_entry["local_path"])
    assert video_path.exists(), "Final video must be materialized on disk"
    assert video_path.stat().st_size > 50 * 1024, "Real adapters should create a non-trivial MP4"

    expected_path = runs_root / run_id / plan_id / "final" / f"{plan_id}-video_final.mp4"
    assert video_path == expected_path

    assert video_entry["checksum_sha256"]
    assert video_entry["duration_s"] == pytest.approx(sum(shot.duration_sec for shot in plan.shots), rel=0.05)


@pytest.mark.gpu
def test_production_agent_resume_after_failure(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    plan = _build_gpu_plan()
    run_id = "gpu-production-agent-resume"
    registry, runs_root, _, plan_id = _configure_runtime(monkeypatch, tmp_path, plan=plan, run_id=run_id)

    original_render = production_agent._render_video_clip
    failure_state: Dict[str, bool] = {"should_fail": True}

    def _flaky_render(*args: Any, **kwargs: Any) -> Path:
        if failure_state["should_fail"]:
            raise StepTransientError("Simulated WAN transient failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr(production_agent, "_render_video_clip", _flaky_render)

    with pytest.raises(StepExecutionError):
        production_agent.execute_plan(plan, mode="run", run_id=run_id)

    failure_state["should_fail"] = False
    _configure_runtime(monkeypatch, tmp_path, plan=plan, run_id=run_id, resume=True)

    result = production_agent.execute_plan(plan, mode="run", run_id=run_id, resume=True)

    assert isinstance(result, production_agent.ProductionResult)
    finalize_record = next(record for record in result.steps if record.step_type == "finalize")
    assert finalize_record.status == "succeeded"

    resumed_steps = {record.step_id for record in result.steps if record.meta.get("resume")}
    expected_resumed = {f"{plan_id}:dialogue_audio", "shot-1:images", "shot-1:tts"}
    assert expected_resumed.issubset(resumed_steps)

    final_entries = registry.get_artifacts(run_id, stage="finalize")
    video_entry = next(entry for entry in final_entries if entry["artifact_type"] == "video_final")
    video_path = Path(video_entry["local_path"])
    assert video_path.exists(), "Final video must be materialized on disk after resume"
    assert video_path.stat().st_size > 50 * 1024

    expected_path = runs_root / run_id / plan_id / "final" / f"{plan_id}-video_final.mp4"
    assert video_path == expected_path
    assert video_entry["checksum_sha256"]
    assert video_entry["duration_s"] == pytest.approx(sum(shot.duration_sec for shot in plan.shots), rel=0.05)


@pytest.mark.gpu
def test_production_agent_rate_limit_handling(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    plan = _build_gpu_plan()
    plan_id = plan.metadata.get("plan_id", plan.title)

    helpers.ensure_real_adapter(
        monkeypatch,
        flags=[
            "SMOKE_ADAPTERS",
            "SMOKE_IMAGES",
            "SMOKE_VIDEOS",
            "SMOKE_TTS",
            "SMOKE_LIPSYNC",
            "SMOKE_ASSEMBLE",
        ],
        disable_keys=[
            "ADK_USE_FIXTURE",
            "IMAGES_SDXL_FIXTURE_ONLY",
            "VIDEOS_WAN_FIXTURE_ONLY",
            "LIPSYNC_WAV2LIP_FIXTURE_ONLY",
            "ASSEMBLE_FFMPEG_FIXTURE_ONLY",
        ],
    )

    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"
    runs_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    helpers.set_env(
        monkeypatch,
        {
            "IMAGES_SDXL_FIXTURE_ONLY": "0",
            "VIDEOS_WAN_FIXTURE_ONLY": "0",
            "LIPSYNC_WAV2LIP_FIXTURE_ONLY": "0",
            "ASSEMBLE_FFMPEG_FIXTURE_ONLY": "0",
            "ARTIFACTS_DIR": str(artifacts_root),
            "SPARKLE_LOCAL_RUNS_ROOT": str(runs_root),
        },
    )

    original_render_frames = production_agent._render_frames
    state = {"queued": False}
    decision = RateLimitDecision(
        status="queued",
        tokens=1,
        retry_after_s=0.0,
        eta_epoch_s=time.time() + 1.0,
        ttl_deadline_s=time.time() + 60.0,
        reason="gpu-test",
    )

    def _rate_limited_render(*args: Any, **kwargs: Any) -> production_agent.StepResult:
        if not state["queued"]:
            state["queued"] = True
            raise production_agent.RateLimitQueued("images stage queued", decision)
        return original_render_frames(*args, **kwargs)

    monkeypatch.setattr(production_agent, "_render_frames", _rate_limited_render)

    scheduled_tasks: List[Callable[[], None]] = []

    def _capture_scheduler(task: Callable[[], None]) -> None:
        scheduled_tasks.append(task)

    monkeypatch.setattr(queue_runner, "_SCHEDULER", _capture_scheduler)

    class _MemoryStub:
        def __init__(self) -> None:
            self.payloads: Dict[str, Dict[str, Any]] = {}

        def store_session_metadata(self, ticket_id: str, payload: Dict[str, Any]) -> None:
            self.payloads[ticket_id] = dict(payload)

    memory_stub = _MemoryStub()
    monkeypatch.setattr(queue_runner.adk_helpers, "get_memory_service", lambda: memory_stub)

    enqueued: List[queue_runner.QueueTicket] = []

    def _enqueue_proxy(**kwargs: Any) -> queue_runner.QueueTicket:
        ticket = queue_runner.enqueue_plan(**kwargs)
        enqueued.append(ticket)
        return ticket

    monkeypatch.setattr(production_entrypoint, "enqueue_plan", _enqueue_proxy)

    request = ProductionAgentRequest(plan=plan, mode="run")
    response = production_entrypoint.invoke(request)

    assert response["status"] == "queued"
    queue_info = response.get("queue")
    assert queue_info is not None
    assert queue_info["ticket_id"].startswith("queue-")
    assert queue_info["step_id"].endswith(":images")
    assert enqueued and enqueued[0].ticket_id == queue_info["ticket_id"]
    assert scheduled_tasks, "Queue resume task must be scheduled"

    run_id = response["run_id"]
    registry = get_run_registry()
    status_payload = registry.get_status(run_id)
    queued_steps = [record for record in status_payload.get("steps", []) if record.get("status") == "queued"]
    assert queued_steps, "Queued step record must be persisted"
    assert queued_steps[0]["step_id"] == queue_info["step_id"]

    resume_task = scheduled_tasks.pop()
    resume_task()

    ticket = enqueued[0]
    assert ticket.status == "completed"

    finalize_entries = registry.get_artifacts(run_id, stage="finalize")
    video_entry = next(entry for entry in finalize_entries if entry["artifact_type"] == "video_final")
    video_path = Path(video_entry["local_path"])
    assert video_path.exists(), "Final video should be produced after dequeue"
    assert video_path.stat().st_size > 50 * 1024

    expected_path = runs_root / run_id / plan_id / "final" / f"{plan_id}-video_final.mp4"
    assert video_path == expected_path

    steps = registry.get_status(run_id)["steps"]
    assert any(step["step_type"] == "finalize" and step["status"] == "succeeded" for step in steps)
