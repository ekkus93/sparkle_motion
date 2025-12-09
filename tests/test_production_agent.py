from __future__ import annotations

from datetime import datetime
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pytest

import sparkle_motion.production_agent as production_agent
from sparkle_motion import adk_helpers, tts_stage

from sparkle_motion.images_stage import RateLimitExceeded, RateLimitQueued
from sparkle_motion.production_agent import (
    ProductionResult,
    ProductionAgentConfig,
    StepExecutionError,
    StepExecutionRecord,
    StepQueuedError,
    StepRateLimitExceededError,
    execute_plan,
)
from sparkle_motion.ratelimit import RateLimitDecision
from sparkle_motion.run_registry import get_run_registry, _reset_filesystem_store_for_tests
from sparkle_motion.dialogue_timeline import DialogueTimelineBuild
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
def _stub_videos_stage(
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

    monkeypatch.setattr("sparkle_motion.videos_stage.render_video", _fake_render)


@pytest.fixture(autouse=True)
def _stub_tts_stage(
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

    monkeypatch.setattr("sparkle_motion.tts_stage.synthesize", _fake_synthesize)
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
    assert result.steps, "expected plan_intake step to be recorded"
    first_step = result.steps[0]
    assert first_step.step_type == "plan_intake"
    assert first_step.status == "succeeded"
    base_map = first_step.meta.get("base_image_map")
    assert isinstance(base_map, dict) and base_map, "plan_intake should record base image map"
    if mode == "dry":
        assert result.simulation_report is not None
    else:
        assert result.simulation_report is None
        assert "production_agent_final_movie" in result[0]["uri"]


def test_plan_intake_uses_base_image_local_path(
    monkeypatch: pytest.MonkeyPatch,
    sample_plan: MoviePlan,
    tmp_path: Path,
    deterministic_media_assets: "MediaAssets",
) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    source = tmp_path / "inputs" / "frame_local.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(deterministic_media_assets.image, source)
    sample_plan.base_images[0].local_path = str(source)
    sample_plan.base_images[0].mime_type = "image/png"

    result = production_agent._run_plan_intake(sample_plan, plan_id="plan-assets", run_id="run-assets", output_dir=tmp_path)
    asset = result.base_image_assets[sample_plan.base_images[0].id]
    assert asset.path and asset.path.exists()
    assert asset.path.suffix == ".png"
    assert asset.payload_bytes == source.read_bytes()
    mapped = result.run_context.base_image_map[sample_plan.base_images[0].id]
    assert mapped == asset.path.as_posix()


def test_plan_intake_builds_stage_manifests(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    result = production_agent._run_plan_intake(sample_plan, plan_id="plan-manifest", run_id="run-manifest", output_dir=tmp_path)
    manifests = result.stage_manifests
    assert manifests, "expected plan_intake to build stage manifest entries"
    artifact_types = {entry.artifact_type for entry in manifests}
    assert {"plan_run_context", "movie_plan", "dialogue_timeline"}.issubset(artifact_types)
    rc_entry = next(entry for entry in manifests if entry.artifact_type == "plan_run_context")
    assert rc_entry.local_path and rc_entry.local_path.endswith("run_context.json")


def test_record_stage_manifest_entries_publish_events(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    result = production_agent._run_plan_intake(sample_plan, plan_id="plan-events", run_id="run-events", output_dir=tmp_path)
    captured: List[Dict[str, Any]] = []
    registry = get_run_registry()
    registry.discard_run("run-events")
    registry.start_run(run_id="run-events", plan_id="plan-events", plan_title=sample_plan.title, mode="run")

    def _fake_write_memory_event(*, run_id: Optional[str], event_type: str, payload: Mapping[str, Any], ts: Optional[datetime] = None) -> None:
        captured.append({"run_id": run_id, "event_type": event_type, "payload": dict(payload)})

    monkeypatch.setattr(production_agent.adk_helpers, "write_memory_event", _fake_write_memory_event)
    production_agent._record_stage_manifest_entries(run_id="run-events", manifests=result.stage_manifests)
    manifest_events = [item for item in captured if item["event_type"] == "production_agent.stage_manifest"]
    assert len(manifest_events) == len(result.stage_manifests)
    assert all(event["payload"].get("stage_id") == "plan_intake" for event in manifest_events)
    stage_artifacts = registry.get_artifacts("run-events", stage="plan_intake")
    assert len(stage_artifacts) == len(result.stage_manifests)
    assert {entry["artifact_type"] for entry in stage_artifacts} == {manifest.artifact_type for manifest in result.stage_manifests}
    registry.discard_run("run-events")


def test_record_stage_manifest_entries_filesystem_backend(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    fs_root = tmp_path / "fs"
    fs_index = fs_root / "index.db"
    output_dir = tmp_path / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(output_dir))
    monkeypatch.setenv("ARTIFACTS_BACKEND", "filesystem")
    monkeypatch.setenv("ARTIFACTS_FS_ROOT", str(fs_root))
    monkeypatch.setenv("ARTIFACTS_FS_INDEX", str(fs_index))
    monkeypatch.setenv("ARTIFACTS_FS_ALLOW_INSECURE", "1")
    adk_helpers._reset_filesystem_store_for_tests()

    result = production_agent._run_plan_intake(sample_plan, plan_id="plan-fs", run_id="run-fs", output_dir=output_dir)
    registry = get_run_registry()
    registry.discard_run("run-fs")
    registry.start_run(run_id="run-fs", plan_id="plan-fs", plan_title=sample_plan.title, mode="run")

    production_agent._record_stage_manifest_entries(run_id="run-fs", manifests=result.stage_manifests)

    artifacts = registry.get_artifacts("run-fs", stage="plan_intake")
    assert artifacts, "expected plan_intake manifests to be recorded"
    assert all(entry["artifact_uri"].startswith("artifact+fs://") for entry in artifacts)
    assert all(entry.get("storage_hint") == "filesystem" for entry in artifacts)
    for entry in artifacts:
        local_path = entry.get("local_path")
        assert local_path, "filesystem manifest should expose local_path"
        assert local_path.startswith(str(fs_root)), "local_path should point to filesystem root"
        assert Path(local_path).exists(), "filesystem copy should exist on disk"
    registry.discard_run("run-fs")


def test_execute_plan_filesystem_backend_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    sample_plan: MoviePlan,
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    fs_root = tmp_path / "fs"
    fs_index = fs_root / "index.db"
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ARTIFACTS_BACKEND", "filesystem")
    monkeypatch.setenv("ARTIFACTS_FS_ROOT", str(fs_root))
    monkeypatch.setenv("ARTIFACTS_FS_INDEX", str(fs_index))
    monkeypatch.setenv("ARTIFACTS_FS_ALLOW_INSECURE", "1")
    monkeypatch.setenv("SMOKE_ADAPTERS", "1")
    monkeypatch.setenv("SMOKE_TTS", "1")
    monkeypatch.setenv("SMOKE_LIPSYNC", "1")
    adk_helpers._reset_filesystem_store_for_tests()
    _reset_filesystem_store_for_tests()

    run_id = "run-fs-end-to-end"
    registry = get_run_registry()
    registry.discard_run(run_id)

    result = execute_plan(sample_plan, mode="run", run_id=run_id)
    assert isinstance(result, ProductionResult)
    assert result.steps, "expected stage execution records"

    stage_groups = registry.get_artifacts_by_stage(run_id)
    assert stage_groups, "expected artifacts to be grouped by stage"
    assert "plan_intake" in stage_groups

    for entries in stage_groups.values():
        for entry in entries:
            assert entry["artifact_uri"].startswith("artifact+fs://"), "filesystem URIs should use artifact+fs scheme"
            local_path = entry.get("local_path")
            if local_path:
                assert local_path.startswith(str(fs_root)), "local_path should point inside filesystem root"
                assert Path(local_path).exists(), "filesystem artifact should exist on disk"

    manifest_rows = registry.list_artifacts(run_id)
    assert manifest_rows, "list_artifacts should surface manifest rows"
    assert {row["run_id"] for row in manifest_rows} == {run_id}
    assert all(row["artifact_uri"].startswith("artifact+fs://") for row in manifest_rows)
    assert {row["storage_hint"] for row in manifest_rows} <= {"filesystem"}

    plan_manifests = [row for row in manifest_rows if row["stage_id"] == "plan_intake"]
    assert plan_manifests, "plan_intake manifests should be present"
    for row in plan_manifests:
        local_path = row.get("local_path")
        if local_path:
            assert Path(local_path).exists()

    registry.discard_run(run_id)


def test_finalize_stage_emits_video_final_manifest(
    monkeypatch: pytest.MonkeyPatch,
    sample_plan: MoviePlan,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_ADAPTERS", "1")
    monkeypatch.setenv("SMOKE_TTS", "1")
    monkeypatch.setenv("SMOKE_LIPSYNC", "1")
    run_id = "run-finalize"
    registry = get_run_registry()
    registry.discard_run(run_id)
    registry.start_run(run_id=run_id, plan_id="plan-qa", plan_title=sample_plan.title, mode="run")
    result = execute_plan(sample_plan, mode="run", run_id=run_id)
    artifacts = registry.get_artifacts(run_id, stage="finalize")
    assert artifacts, "expected finalize manifest entry"
    final_entry = artifacts[-1]
    assert final_entry["artifact_type"] == "video_final"
    assert final_entry["local_path"]
    assert Path(final_entry["local_path"]).exists()
    assert final_entry["checksum_sha256"] and len(final_entry["checksum_sha256"]) == 64
    expected_duration = sum(shot.duration_sec for shot in sample_plan.shots)
    assert pytest.approx(expected_duration) == final_entry["duration_s"]
    assert any(artifact.get("artifact_type") == "video_final" for artifact in result)
    registry.discard_run(run_id)


def test_shot_and_assemble_stage_manifests_recorded(
    monkeypatch: pytest.MonkeyPatch,
    sample_plan: MoviePlan,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_ADAPTERS", "1")
    monkeypatch.setenv("SMOKE_TTS", "1")
    monkeypatch.setenv("SMOKE_LIPSYNC", "1")
    run_id = "run-shot-manifests"
    registry = get_run_registry()
    registry.discard_run(run_id)
    registry.start_run(run_id=run_id, plan_id="plan-shot", plan_title=sample_plan.title, mode="run")
    execute_plan(sample_plan, mode="run", run_id=run_id)
    frames_entries = registry.get_artifacts(run_id, stage="shot-1:images")
    assert frames_entries, "expected shot frames manifest"
    assert frames_entries[-1]["artifact_type"] == "shot_frames"
    assert frames_entries[-1]["local_path"].endswith("frames.json")
    video_entries = registry.get_artifacts(run_id, stage="shot-1:video")
    assert video_entries, "expected shot video manifest"
    assert video_entries[-1]["artifact_type"] == "shot_video"
    assert video_entries[-1]["duration_s"] == pytest.approx(sample_plan.shots[0].duration_sec)
    tts_entries = registry.get_artifacts(run_id, stage="shot-1:tts")
    assert tts_entries, "expected shot tts manifest"
    assert tts_entries[-1]["artifact_type"] == "shot_dialogue_audio"
    assert tts_entries[-1]["local_path"].endswith("tts_summary.json")
    assert tts_entries[-1]["metadata"].get("dialogue_paths")
    lipsync_entries = registry.get_artifacts(run_id, stage="shot-1:lipsync")
    assert lipsync_entries, "expected shot lipsync manifest"
    assert lipsync_entries[-1]["artifact_type"] == "shot_lipsync_video"
    assemble_entries = registry.get_artifacts(run_id, stage="assemble")
    assert assemble_entries, "expected assemble manifest"
    assert assemble_entries[-1]["artifact_type"] == "assembly_plan"
    registry.discard_run(run_id)


def test_pipeline_stage_statuses_and_manifests(
    monkeypatch: pytest.MonkeyPatch,
    sample_plan: MoviePlan,
    tmp_path: Path,
) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    run_id = "run-stage-coverage"
    plan_id = sample_plan.metadata.get("plan_id", "plan-stage-coverage")
    registry = get_run_registry()
    registry.discard_run(run_id)
    registry.start_run(run_id=run_id, plan_id=plan_id, plan_title=sample_plan.title, mode="run")

    result = execute_plan(sample_plan, mode="run", run_id=run_id)

    def _assert_step(step_type: str, *, predicate: Optional[Callable[[StepExecutionRecord], bool]] = None) -> StepExecutionRecord:
        record = next(
            (
                record
                for record in result.steps
                if record.step_type == step_type and (predicate(record) if predicate else True)
            ),
            None,
        )
        assert record is not None, f"expected StepExecutionRecord for {step_type}"
        assert record.status == "succeeded", f"expected {step_type} to succeed"
        return record

    _assert_step("plan_intake")
    _assert_step("dialogue_audio")
    first_shot = sample_plan.shots[0].id
    _assert_step("lipsync", predicate=lambda record: record.step_id.startswith(f"{first_shot}:"))
    _assert_step("assemble")
    _assert_step("finalize")

    def _assert_stage(stage: str) -> List[Mapping[str, Any]]:
        entries = registry.get_artifacts(run_id, stage=stage)
        assert entries, f"expected manifest entries for {stage}"
        return entries

    plan_entries = _assert_stage("plan_intake")
    assert any(entry["artifact_type"] == "movie_plan" for entry in plan_entries)

    dialogue_entries = _assert_stage("dialogue_audio")
    assert any(entry["artifact_type"] == "dialogue_timeline_audio" for entry in dialogue_entries)

    lipsync_stage = f"{first_shot}:lipsync"
    lipsync_entries = _assert_stage(lipsync_stage)
    assert any(entry["artifact_type"] == "shot_lipsync_video" for entry in lipsync_entries)

    assemble_entries = _assert_stage("assemble")
    assert any(entry["artifact_type"] == "assembly_plan" for entry in assemble_entries)

    finalize_entries = _assert_stage("finalize")
    assert any(entry["artifact_type"] == "video_final" for entry in finalize_entries)

    registry.discard_run(run_id)


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
    tts_enabled = os.getenv("SMOKE_TTS") not in {None, "", "0", "false"}
    lipsync_enabled = os.getenv("SMOKE_LIPSYNC") not in {None, "", "0", "false"}
    expected = "succeeded" if adapters_enabled else "simulated"
    assert statuses["plan_intake"] == "succeeded"
    assert statuses["images"] == expected
    assert statuses["video"] == expected
    assert statuses["tts"] == ("succeeded" if tts_enabled else "simulated")
    if "lipsync" in statuses:
        assert statuses["lipsync"] == ("succeeded" if lipsync_enabled else "simulated")
    assert "qa_base_images" not in statuses
    assert "qa_video" not in statuses


def test_run_registry_tracks_status_metadata(
    monkeypatch: pytest.MonkeyPatch,
    sample_plan: MoviePlan,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("SMOKE_ADAPTERS", "1")
    monkeypatch.setenv("SMOKE_TTS", "1")
    monkeypatch.setenv("SMOKE_LIPSYNC", "1")
    run_id = "run-status-metadata"
    plan_id = "plan-status"
    registry = get_run_registry()
    registry.discard_run(run_id)
    registry.start_run(
        run_id=run_id,
        plan_id=plan_id,
        plan_title=sample_plan.title,
        mode="run",
        render_profile={"video": {"model_id": "wan-fixture"}},
        run_metadata={"plan_id": plan_id},
    )
    progress_handler = registry.build_progress_handler(run_id)
    pre_step_hook = registry.pre_step_hook(run_id)
    execute_plan(
        sample_plan,
        mode="run",
        run_id=run_id,
        progress_callback=progress_handler,
        pre_step_hook=pre_step_hook,
    )
    status = registry.get_status(run_id)
    assert status["run_id"] == run_id
    assert status["metadata"].get("plan_id") == plan_id
    assert status["timeline"], "timeline entries should be captured"
    artifacts = registry.get_artifacts(run_id, stage="finalize")
    final_entry = next(entry for entry in artifacts if entry["artifact_type"] == "video_final")
    assert isinstance(final_entry["playback_ready"], bool)
    registry.discard_run(run_id)


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
    monkeypatch.setenv("VIDEOS_STAGE_DEFAULT_FPS", "12")
    expected_path = tmp_path / "video" / f"{shot.id}.mp4"
    base_images = {img.id: img for img in sample_plan.base_images}
    base_image_assets = {
        img.id: production_agent._BaseImageAsset(spec=img, path=None, payload_bytes=img.prompt.encode("utf-8"))
        for img in sample_plan.base_images
    }

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

    monkeypatch.setattr("sparkle_motion.videos_stage.render_video", _capture)

    result_path = production_agent._render_video_clip(shot, tmp_path, plan_id, run_id, base_images, base_image_assets)

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


def test_render_frames_persists_continuity_assets(sample_plan: MoviePlan, tmp_path: Path) -> None:
    shot = sample_plan.shots[0]
    base_images = {img.id: img for img in sample_plan.base_images}
    base_image_assets = {
        img.id: production_agent._BaseImageAsset(spec=img, path=None, payload_bytes=img.prompt.encode("utf-8"))
        for img in sample_plan.base_images
    }

    result = production_agent._render_frames(shot, tmp_path, base_images, base_image_assets)

    assert result.path is not None
    assert result.path.exists()
    expected_dir = tmp_path / "frames" / shot.id
    assert result.path.parent == expected_dir

    start_asset = base_image_assets[shot.start_base_image_id]
    end_asset = base_image_assets[shot.end_base_image_id]
    for asset, role in ((start_asset, "start"), (end_asset, "end")):
        assert asset.path is not None and asset.path.exists()
        payload = json.loads(asset.path.read_text())
        assert payload["shot_id"] == shot.id
        assert payload["role"] == role
        assert payload["base_image_id"].startswith("frame_")

    assert end_asset.payload_bytes != start_asset.payload_bytes
    assert json.loads(result.path.read_text())["end_frame_path"] == end_asset.path.as_posix()


def test_video_stage_reuses_base_image_payloads(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    captured: List[Dict[str, Any]] = []

    def _capture(
        start_frames: Iterable[bytes],
        end_frames: Iterable[bytes],
        prompt: str,
        opts: Optional[Mapping[str, Any]] = None,
        *,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        _adapter: Optional[Any] = None,
    ) -> Dict[str, Any]:
        options = dict(opts or {})
        target = Path(options.get("output_path") or tmp_path / "video" / f"{options.get('shot_id')}.mp4")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        captured.append(
            {
                "shot_id": options.get("shot_id"),
                "start": [bytes(chunk) for chunk in start_frames],
                "end": [bytes(chunk) for chunk in end_frames],
            }
        )
        return {"uri": f"file://{target}", "metadata": {"source_path": str(target), "prompt": prompt}}

    monkeypatch.setattr("sparkle_motion.videos_stage.render_video", _capture)

    execute_plan(sample_plan, mode="run")

    assert len(captured) == len(sample_plan.shots)
    assert captured[0]["end"], "expected end frame payload"
    assert captured[0]["end"][0] == captured[1]["start"][0]
    first_end_payload = captured[0]["end"][0]
    assert b'"role": "end"' in first_end_payload
    assert sample_plan.shots[0].id.encode() in first_end_payload


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
        if record.step_type == "video" and record.status == "running" and "videos_stage_progress" in record.meta
    ]
    assert progress_records, "Expected chunk progress records from videos_stage"
    event_meta = progress_records[0].meta["videos_stage_progress"]
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


def test_dialogue_stage_builds_timeline_audio(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    result = execute_plan(sample_plan, mode="run")
    dialogue_records = [record for record in result.steps if record.step_type == "dialogue_audio"]
    assert dialogue_records, "Expected dialogue/audio stage execution"
    record = dialogue_records[0]
    timeline_audio = Path(record.meta["timeline_audio_path"])
    summary_path = Path(record.meta["timeline_summary_path"])
    assert timeline_audio.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["entry_count"] == len(sample_plan.dialogue_timeline)
    assert record.meta["entry_count"] == len(sample_plan.dialogue_timeline)
    assert record.meta["line_artifacts"], "Line artifacts should capture per-entry metadata"
    assert "timeline_offsets" in record.meta
    assert summary["timeline_offsets"]
    assert len(summary["timeline_offsets"]) == summary["entry_count"]
    first_entry = summary["lines"][0]
    assert pytest.approx(first_entry["start_time_actual_s"], rel=1e-3) == first_entry["start_time_sec"]
    expected_end = first_entry["start_time_sec"] + first_entry["duration_sec"]
    assert pytest.approx(first_entry["end_time_actual_s"], rel=1e-3) == expected_end
    assert first_entry["timeline_padding_s"] >= 0.0
    total_plan_duration = sum(line.get("duration_sec", 0.0) for line in summary["lines"])
    assert pytest.approx(summary["timeline_audio"]["duration_s"], rel=1e-3) == total_plan_duration


def test_run_dialogue_stage_uses_timeline_builder(
    monkeypatch: pytest.MonkeyPatch,
    sample_plan: MoviePlan,
    tmp_path: Path,
) -> None:
    captured: Dict[str, Any] = {}

    class _FakeBuilder:
        def __init__(
            self,
            *,
            synthesizer: Any,
            voice_resolver: Any,
            timeline_subdir: str = "audio/timeline",
            timeline_audio_filename: str = "tts_timeline.wav",
            summary_filename: str = "dialogue_timeline_audio.json",
        ) -> None:
            captured["builder_init"] = {
                "timeline_subdir": timeline_subdir,
                "timeline_audio_filename": timeline_audio_filename,
                "summary_filename": summary_filename,
            }
            self._summary_filename = summary_filename
            self._timeline_filename = timeline_audio_filename

        def build(self, plan: MoviePlan, *, plan_id: str, run_id: str, output_dir: Path) -> DialogueTimelineBuild:
            captured["build_args"] = {"plan_id": plan_id, "run_id": run_id, "plan_title": plan.title}
            line_dir = output_dir / "lines"
            line_dir.mkdir(parents=True, exist_ok=True)
            line_path = line_dir / "line.wav"
            line_path.write_bytes(b"00")
            timeline_path = output_dir / self._timeline_filename
            timeline_path.parent.mkdir(parents=True, exist_ok=True)
            timeline_path.write_bytes(b"11")
            summary_path = output_dir / self._summary_filename
            summary_payload = {"entry_count": 1, "lines": [{"index": 0, "text": "stub"}]}
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
            offsets = {0: {"start_time_s": 0.0, "end_time_s": 1.0, "written_duration_s": 1.0}}
            return DialogueTimelineBuild(
                line_entries=[{"index": 0, "text": "stub", "start_time_sec": 0.0, "duration_sec": 1.0}],
                line_paths=[line_path],
                summary_path=summary_path,
                summary_payload=summary_payload,
                timeline_audio_path=timeline_path,
                total_duration_s=1.0,
                sample_rate=22050,
                channels=1,
                sample_width=2,
                timeline_offsets=offsets,
            )

    monkeypatch.setattr(production_agent, "DialogueTimelineBuilder", _FakeBuilder)
    result = production_agent._run_dialogue_stage(
        sample_plan,
        plan_id="plan-builder",
        run_id="run-builder",
        output_dir=tmp_path,
        voice_profiles=production_agent._character_voice_map(sample_plan),
    )

    assert captured["build_args"]["plan_id"] == "plan-builder"
    assert result.line_entries[0]["text"] == "stub"
    assert result.summary_path.exists()
    assert result.timeline_audio_path.exists()


def test_dialogue_stage_reflects_plan_edits(
    sample_plan: MoviePlan,
    tmp_path: Path,
) -> None:
    sample_plan.shots[0].duration_sec = 4.0
    sample_plan.shots[0].dialogue[0].text = "Edited line"
    sample_plan.dialogue_timeline[0].text = "Edited line"
    sample_plan.dialogue_timeline[0].duration_sec = 4.0
    if len(sample_plan.dialogue_timeline) > 1:
        sample_plan.dialogue_timeline[1].start_time_sec = 4.0

    result = production_agent._run_dialogue_stage(
        sample_plan,
        plan_id="plan-edit",
        run_id="run-edit",
        output_dir=tmp_path,
        voice_profiles=production_agent._character_voice_map(sample_plan),
    )

    summary = json.loads(result.summary_path.read_text())
    first_line = summary["lines"][0]
    assert first_line["text"] == "Edited line"
    assert pytest.approx(first_line["duration_sec"], rel=1e-3) == 4.0
    assert pytest.approx(summary["lines"][1]["start_time_sec"], rel=1e-3) == 4.0


def test_run_dialogue_stage_returns_stage_manifests(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    voice_profiles = production_agent._character_voice_map(sample_plan)
    result = production_agent._run_dialogue_stage(
        sample_plan,
        plan_id="plan-dialogue",
        run_id="run-dialogue",
        output_dir=tmp_path,
        voice_profiles=voice_profiles,
    )
    assert result.timeline_audio_path.exists()
    assert result.summary_path.exists()
    artifact_types = {manifest.artifact_type for manifest in result.stage_manifests}
    assert {"dialogue_timeline_audio", "tts_timeline_audio"}.issubset(artifact_types)


def test_dialogue_stage_trims_and_pads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)
    shots = [
        ShotSpec(
            id="shot-trim",
            visual_description="Quick close-up",
            duration_sec=0.2,
            dialogue=[DialogueLine(character_id="hero", text="Hi")],
            start_base_image_id="frame_a",
            end_base_image_id="frame_b",
            is_talking_closeup=True,
        ),
        ShotSpec(
            id="shot-pad",
            visual_description="Long speech",
            duration_sec=1.5,
            dialogue=[DialogueLine(character_id="hero", text="This line runs long")],
            start_base_image_id="frame_b",
            end_base_image_id="frame_c",
            is_talking_closeup=True,
        ),
    ]
    timeline = [
        DialogueTimelineDialogue(character_id="hero", text="Hi", start_time_sec=0.0, duration_sec=0.2),
        DialogueTimelineDialogue(character_id="hero", text="This line runs long", start_time_sec=0.2, duration_sec=1.5),
    ]
    plan = MoviePlan(
        title="Trim Pad",
        metadata={"plan_id": "plan-trim-pad"},
        characters=[CharacterSpec(id="hero", name="Hero")],
        base_images=[
            BaseImageSpec(id="frame_a", prompt="start"),
            BaseImageSpec(id="frame_b", prompt="mid"),
            BaseImageSpec(id="frame_c", prompt="end"),
        ],
        shots=shots,
        dialogue_timeline=timeline,
        render_profile=RenderProfile(video=RenderProfileVideo(model_id="wan-fixture")),
    )
    voice_profiles = production_agent._character_voice_map(plan)
    result = production_agent._run_dialogue_stage(
        plan,
        plan_id="plan-trim-pad",
        run_id="run-trim-pad",
        output_dir=tmp_path,
        voice_profiles=voice_profiles,
    )
    assert pytest.approx(result.total_duration_s, rel=1e-3) == 1.7
    first_entry = result.line_entries[0]
    second_entry = result.line_entries[1]
    assert first_entry["timeline_trimmed_s"] > 0.0
    assert pytest.approx(first_entry["duration_actual_s"], rel=1e-3) == 0.2
    assert second_entry["timeline_padding_s"] > 0.0
    assert pytest.approx(second_entry["duration_actual_s"], rel=1e-3) == 1.5


def test_voice_profile_forwarded_to_tts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _stub_tts_stage: List[Dict[str, Any]],
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
    synthesized_voices = [call.get("voice_config", {}) for call in _stub_tts_stage]
    voice_ids = {config.get("voice_id") for config in synthesized_voices if config}
    assert "hero_voice" in voice_ids
    assert "narrator_voice" in voice_ids


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

def test_tts_quota_error_surfaces(monkeypatch: pytest.MonkeyPatch, sample_plan: MoviePlan, tmp_path: Path) -> None:
    _enable_full_execution(monkeypatch, tmp_path)

    def _raise_quota(*_: object, **__: object) -> None:
        raise tts_stage.TTSQuotaExceeded("quota hit")

    monkeypatch.setattr("sparkle_motion.tts_stage.synthesize", _raise_quota)
    records: List[StepExecutionRecord] = []

    with pytest.raises(StepExecutionError):
        execute_plan(sample_plan, mode="run", progress_callback=records.append)

    assert records[-1].step_type in {"dialogue_audio", "tts"}
    assert records[-1].status == "failed"
    assert records[-1].error_type == "TTSQuotaExceeded"
