from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pytest

import sparkle_motion.production_agent as production_agent
from sparkle_motion import tts_agent

from sparkle_motion.images_agent import RateLimitExceeded, RateLimitQueued
from sparkle_motion.production_agent import (
    ProductionResult,
    StepExecutionError,
    StepExecutionRecord,
    StepQueuedError,
    StepRateLimitExceededError,
    execute_plan,
)
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

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMOKE_ADAPTERS", raising=False)
    monkeypatch.delenv("SMOKE_TTS", raising=False)
    monkeypatch.delenv("SMOKE_LIPSYNC", raising=False)


@pytest.fixture(autouse=True)
def _stub_videos_agent(
    monkeypatch: pytest.MonkeyPatch, deterministic_media_assets: MediaAssets
) -> None:
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
        shutil.copyfile(deterministic_media_assets.video, target)
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


@pytest.fixture(autouse=True)
def _stub_tts_agent(
    monkeypatch: pytest.MonkeyPatch, deterministic_media_assets: MediaAssets
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def _fake_synthesize(text: str, voice_config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        output_dir = Path(kwargs.get("output_dir") or Path.cwd())
        output_dir.mkdir(parents=True, exist_ok=True)
        step_label = (kwargs.get("step_id") or "tts").replace(":", "_")
        dest = output_dir / f"{step_label}.wav"
        shutil.copyfile(deterministic_media_assets.audio, dest)
        voice_id = (voice_config or {}).get("voice_id", "default")
        voice_payload = {
            "voice_id": voice_id,
            "provider_id": "fixture-local",
            "provider_voice_id": f"{voice_id}-voice",
            "description": f"{voice_id} voice",
            "display_name": voice_id.title(),
            "language_codes": ["en"],
            "features": ["fixture"],
            "sample_rate": 22050,
            "bit_depth": 16,
            "watermarking": False,
            "estimated_cost_usd_per_1k_chars": 0.0,
            "estimated_latency_s": 0.1,
            "quality_score": 0.5,
        }
        metadata = {
            "source_path": str(dest),
            "model_id": "fixture-tts",
            "provider_id": "fixture-local",
            "voice_id": voice_id,
            "duration_s": 0.5,
            "sample_rate": 22050,
            "bit_depth": 16,
            "watermarked": False,
            "score_breakdown": {"quality": 1.0},
            "voice_metadata": voice_payload,
        }
        calls.append(
            {
                "text": text,
                "voice_config": voice_config,
                "plan_id": kwargs.get("plan_id"),
                "step_id": kwargs.get("step_id"),
                "run_id": kwargs.get("run_id"),
            }
        )
        return {
            "uri": dest.as_uri(),
            "storage": "local",
            "artifact_type": "tts_audio",
            "metadata": metadata,
        }

    monkeypatch.setattr("sparkle_motion.tts_agent.synthesize", _fake_synthesize)
    return calls


def _timeline_for_shots(shots: Sequence[ShotSpec]) -> List[DialogueTimelineDialogue | DialogueTimelineSilence]:
    entries: List[DialogueTimelineDialogue | DialogueTimelineSilence] = []
    current = 0.0
    for shot in shots:
        spoken_line = next((line for line in shot.dialogue if line.text.strip()), None)
        if spoken_line is not None:
            entries.append(
                DialogueTimelineDialogue(
                    character_id=spoken_line.character_id,
                    text=spoken_line.text,
                    start_time_sec=current,
                    duration_sec=shot.duration_sec,
                )
            )
        else:
            entries.append(
                DialogueTimelineSilence(
                    start_time_sec=current,
                    duration_sec=shot.duration_sec,
                )
            )
        current += shot.duration_sec
    return entries


@pytest.fixture
def sample_plan() -> MoviePlan:
    shots = [
        ShotSpec(
            id="shot-1",
            visual_description="A hero poses",
            duration_sec=3,
            dialogue=[
                DialogueLine(character_id="hero", text="Hello there!"),
            ],
            start_base_image_id="frame_000",
            end_base_image_id="frame_001",
            is_talking_closeup=True,
        ),
        ShotSpec(
            id="shot-2",
            visual_description="Establishing shot",
            duration_sec=5,
            dialogue=[],
            start_base_image_id="frame_001",
            end_base_image_id="frame_002",
            is_talking_closeup=False,
        ),
    ]
    base_images = [
        BaseImageSpec(id="frame_000", prompt="Hero start frame"),
        BaseImageSpec(id="frame_001", prompt="Hero end frame / establishing intro"),
        BaseImageSpec(id="frame_002", prompt="Sunset skyline"),
    ]
    timeline = _timeline_for_shots(shots)
    return MoviePlan(
        title="Test Film",
        metadata={"plan_id": "plan-123"},
        characters=[CharacterSpec(id="hero", name="Hero")],
        base_images=base_images,
        shots=shots,
        dialogue_timeline=timeline,
        render_profile=RenderProfile(video=RenderProfileVideo(model_id="wan-fixture"), metadata={}),
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
    base_images = {img.id: img for img in sample_plan.base_images}

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

    result_path = production_agent._render_video_clip(shot, tmp_path, plan_id, run_id, base_images)

    assert result_path == expected_path
    start_prompt = base_images[shot.start_base_image_id].prompt
    end_prompt = base_images[shot.end_base_image_id].prompt
    assert captured["start_frames"][0] == start_prompt.encode("utf-8")
    assert captured["end_frames"][0] == end_prompt.encode("utf-8")
    expected_prompt = " | ".join(part for part in [shot.visual_description, start_prompt, end_prompt] if part)
    assert captured["prompt"] == expected_prompt
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


def test_tts_step_records_metadata(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    result = execute_plan(sample_plan, mode="run")
    tts_records = [record for record in result.steps if record.step_type == "tts"]
    assert tts_records, "Expected TTS stage execution"
    record = tts_records[0]
    assert record.model_id == "fixture-tts"
    tts_meta = record.meta["tts"]
    assert tts_meta["provider_id"] == "fixture-local"
    assert record.artifact_uri and record.artifact_uri.startswith("file://")
    assert record.meta["lines"] == len(sample_plan.shots[0].dialogue)
    assert tts_meta["lines_synthesized"] == len(sample_plan.shots[0].dialogue)
    assert len(tts_meta["line_artifacts"]) == len(sample_plan.shots[0].dialogue)
    assert record.meta["dialogue_paths"]
    assert tts_meta["voice_metadata"]["provider_id"] == "fixture-local"
    assert "voice_metadata" in tts_meta["line_artifacts"][0]


def test_voice_profile_forwarded_to_tts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _stub_tts_agent: List[Dict[str, Any]],
) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    shots = [
        ShotSpec(
            id="shot-voice",
            visual_description="Conversation",
            duration_sec=3,
            dialogue=[
                DialogueLine(character_id="hero", text="It is my turn to speak."),
                DialogueLine(character_id="narrator", text="Now I will reply."),
            ],
            start_base_image_id="frame_000",
            end_base_image_id="frame_001",
            is_talking_closeup=True,
        )
    ]
    plan = MoviePlan(
        title="Voices",
        metadata={"plan_id": "plan-voices"},
        characters=[
            CharacterSpec(id="hero", name="Hero", voice_profile={"voice_id": "hero_voice"}),
            CharacterSpec(id="narrator", name="Narrator", voice_profile={"voice_id": "narrator_voice"}),
        ],
        base_images=[
            BaseImageSpec(id="frame_000", prompt="hero start"),
            BaseImageSpec(id="frame_001", prompt="hero end"),
        ],
        shots=shots,
        dialogue_timeline=_timeline_for_shots(shots),
        render_profile=RenderProfile(video=RenderProfileVideo(model_id="wan-fixture")),
    )

    execute_plan(plan, mode="run")
    assert len(_stub_tts_agent) == 2, "Expected one synthesis call per dialogue line"
    first_voice = _stub_tts_agent[0]["voice_config"]
    second_voice = _stub_tts_agent[1]["voice_config"]
    assert first_voice and first_voice.get("voice_id") == "hero_voice"
    assert second_voice and second_voice.get("voice_id") == "narrator_voice"


def test_multiple_dialogue_lines_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    shots = [
        ShotSpec(
            id="shot-multi",
            visual_description="Back-and-forth",
            duration_sec=3,
            dialogue=[
                DialogueLine(character_id="a", text="First line."),
                DialogueLine(character_id="b", text="Second line."),
            ],
            start_base_image_id="frame_100",
            end_base_image_id="frame_101",
            is_talking_closeup=True,
        )
    ]
    plan = MoviePlan(
        title="Two Lines",
        metadata={"plan_id": "plan-multi"},
        base_images=[
            BaseImageSpec(id="frame_100", prompt="start"),
            BaseImageSpec(id="frame_101", prompt="end"),
        ],
        shots=shots,
        dialogue_timeline=_timeline_for_shots(shots),
        render_profile=RenderProfile(video=RenderProfileVideo(model_id="wan-fixture")),
    )

    result = execute_plan(plan, mode="run")
    tts_records = [record for record in result.steps if record.step_type == "tts" and record.step_id.endswith(":tts")]
    assert tts_records, "Expected TTS stage execution"
    record = tts_records[0]
    dialogue_paths = record.meta["dialogue_paths"]
    assert len(dialogue_paths) == 2
    assert len(record.meta["tts"]["line_artifacts"]) == 2
    for artifact in record.meta["tts"]["line_artifacts"]:
        assert artifact["voice_metadata"]["language_codes"] == ["en"]


def test_tts_policy_violation_raises(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "sparkle_motion.tts_agent.synthesize",
        lambda *_, **__: (_ for _ in ()).throw(tts_agent.TTSPolicyViolation("blocked")),
    )
    records: List[StepExecutionRecord] = []

    with pytest.raises(StepExecutionError):
        execute_plan(sample_plan, mode="run", progress_callback=records.append)

    assert records, "Expected progress records when policy error occurs"
    assert records[-1].step_type == "tts"
    assert records[-1].status == "failed"
    assert records[-1].error_type == "TTSPolicyViolation"


def test_tts_quota_error_surfaces(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)

    def _raise_quota(*_: object, **__: object) -> None:
        raise tts_agent.TTSQuotaExceeded("quota hit")

    monkeypatch.setattr("sparkle_motion.tts_agent.synthesize", _raise_quota)
    records: List[StepExecutionRecord] = []

    with pytest.raises(StepExecutionError):
        execute_plan(sample_plan, mode="run", progress_callback=records.append)

    assert records[-1].step_type == "tts"
    assert records[-1].status == "failed"
    assert records[-1].error_type == "TTSQuotaExceeded"
