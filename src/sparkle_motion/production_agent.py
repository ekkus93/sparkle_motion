from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

from . import adk_helpers, observability, telemetry, videos_stage, tts_stage, schema_registry
from .dialogue_timeline import (
    DialogueTimelineBuilder,
    DialogueTimelineError,
    DialogueSynthesizer,
    build_dialogue_timeline,
)
from .run_registry import ArtifactEntry, get_run_registry
from .images_stage import RateLimitExceeded, RateLimitQueued
from .ratelimit import RateLimitDecision
from .schemas import BaseImageSpec, DialogueLine, MoviePlan, ShotSpec, RunContext, StageEvent, StageManifest
from .utils.env import filesystem_backend_enabled


class ProductionAgentError(RuntimeError):
    """Base error for production agent failures."""


class StepExecutionError(ProductionAgentError):
    """Raised when a step exhausts retries without success."""


class StepTransientError(StepExecutionError):
    """Raised for retryable errors (mostly used in tests)."""


@dataclass(frozen=True)
class ProductionAgentConfig:
    """Runtime knobs for the production agent."""

    artifact_type: str = "production_agent_final_movie"
    adapters_flag: str = "SMOKE_ADAPTERS"
    tts_flag: str = "SMOKE_TTS"
    lipsync_flag: str = "SMOKE_LIPSYNC"
    max_attempts: int = 2
    backoff_base_seconds: float = 0.4

    def retry_delay(self, attempt: int) -> float:
        return self.backoff_base_seconds * (2 ** (attempt - 1))


@dataclass
class StepExecutionRecord:
    plan_id: str
    step_id: str
    step_type: str
    status: Literal["queued", "running", "succeeded", "failed", "skipped", "simulated"]
    start_time: str
    end_time: str
    duration_s: float
    attempts: int
    model_id: Optional[str] = None
    device: Optional[str] = None
    memory_hint_mb: Optional[int] = None
    logs_uri: Optional[str] = None
    artifact_uri: Optional[str] = None
    error_type: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "step_type": self.step_type,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "attempts": self.attempts,
            "model_id": self.model_id,
            "device": self.device,
            "memory_hint_mb": self.memory_hint_mb,
            "logs_uri": self.logs_uri,
            "artifact_uri": self.artifact_uri,
            "error_type": self.error_type,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class StepResult:
    path: Optional[Path] = None
    paths: Optional[Sequence[Path]] = None
    artifact_uri: Optional[str] = None
    model_id: Optional[str] = None
    device: Optional[str] = None
    memory_hint_mb: Optional[int] = None
    logs_uri: Optional[str] = None
    meta: Optional[Mapping[str, Any]] = None


StepActionReturn = Optional[Union[Path, StepResult, str]]


class StepRateLimitError(StepExecutionError):
    """Base error for rate-limit conditions with attached record."""

    def __init__(self, message: str, *, record: StepExecutionRecord) -> None:
        super().__init__(message)
        self.record = record


class StepQueuedError(StepRateLimitError):
    """Raised when a step is queued due to rate limiting."""


class StepRateLimitExceededError(StepRateLimitError):
    """Raised when a step exhausts rate-limit budget immediately."""


@dataclass(frozen=True)
class SimulationStep:
    step_id: str
    step_type: str
    estimated_runtime_s: float
    estimated_gpu_memory_mb: int
    simulated_artifact_uri: str


@dataclass(frozen=True)
class SimulationReport:
    plan_id: str
    steps: Sequence[SimulationStep]
    resource_summary: Dict[str, float]


class ProductionResult(list):
    """List of published artifacts plus execution metadata."""

    def __init__(
        self,
        artifacts: Iterable[adk_helpers.ArtifactRef],
        *,
        steps: Optional[Sequence[StepExecutionRecord]] = None,
        simulation_report: Optional[SimulationReport] = None,
    ) -> None:
        super().__init__(artifacts)
        self.steps: List[StepExecutionRecord] = list(steps or [])
        self.simulation_report = simulation_report


@dataclass
class _ShotArtifacts:
    shot_id: str
    frames_path: Optional[Path] = None
    dialogue_paths: List[Path] = field(default_factory=list)
    video_path: Optional[Path] = None
    lipsync_path: Optional[Path] = None


@dataclass(frozen=True)
class _BaseImageAsset:
    spec: BaseImageSpec
    path: Optional[Path]
    payload_bytes: bytes


@dataclass(frozen=True)
class _PlanIntakeResult:
    base_image_lookup: Dict[str, BaseImageSpec]
    base_image_assets: Dict[str, _BaseImageAsset]
    run_context: RunContext
    run_context_path: Optional[Path]
    plan_path: Optional[Path]
    dialogue_timeline_path: Optional[Path]
    schema_meta: Dict[str, Dict[str, str]]
    stage_manifests: List[StageManifest] = field(default_factory=list)


@dataclass(frozen=True)
class _DialogueStageResult:
    line_entries: List[Dict[str, Any]]
    line_paths: List[Path]
    summary_path: Path
    timeline_audio_path: Path
    total_duration_s: float
    sample_rate: int
    channels: int
    sample_width: int
    timeline_offsets: Dict[int, Dict[str, Any]]
    stage_manifests: List[StageManifest]


class _ResumeSnapshot:
    def __init__(self, *, enabled: bool, plan_id: str, run_id: str, output_dir: Path) -> None:
        self.enabled = enabled
        self.plan_id = plan_id
        self.run_id = run_id
        self.output_dir = output_dir

    def resume_dialogue_stage(self) -> Optional[Tuple[StepExecutionRecord, StepResult, _DialogueStageResult]]:
        if not self.enabled:
            return None
        summary_path = self.output_dir / "audio" / "timeline" / "dialogue_timeline_audio.json"
        payload = _load_json_object(summary_path)
        if payload is None:
            return None
        timeline_meta = payload.get("timeline_audio") or {}
        timeline_path = _resolve_existing_path(timeline_meta.get("path"), base=summary_path.parent)
        if timeline_path is None or not timeline_path.exists():
            return None
        line_entries_raw = payload.get("lines") or []
        line_entries = [dict(entry) for entry in line_entries_raw if isinstance(entry, Mapping)]
        line_paths: List[Path] = []
        for entry in line_entries:
            candidate = _resolve_existing_path(entry.get("local_path"), base=summary_path.parent)
            if candidate is not None:
                line_paths.append(candidate)
        offsets_payload = payload.get("timeline_offsets") or {}
        timeline_offsets: Dict[int, Dict[str, Any]] = {}
        if isinstance(offsets_payload, Mapping):
            for key, value in offsets_payload.items():
                try:
                    timeline_offsets[int(key)] = dict(value)
                except (TypeError, ValueError):
                    continue
        total_duration = float(timeline_meta.get("duration_s", 0.0))
        sample_rate = int(timeline_meta.get("sample_rate", 22050))
        channels = int(timeline_meta.get("channels", 1))
        sample_width = int(timeline_meta.get("sample_width_bytes", 2))
        stage_result = _DialogueStageResult(
            line_entries=line_entries,
            line_paths=line_paths,
            summary_path=summary_path,
            timeline_audio_path=timeline_path,
            total_duration_s=total_duration,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            timeline_offsets=timeline_offsets,
            stage_manifests=[],
        )
        metadata: Dict[str, Any] = {
            "stage": "dialogue_audio",
            "entry_count": len(line_entries),
            "timeline_summary_path": summary_path.as_posix(),
            "timeline_audio_path": timeline_path.as_posix(),
            "resume": True,
        }
        step_result = StepResult(
            path=timeline_path,
            paths=tuple([*line_paths, timeline_path]),
            artifact_uri=timeline_path.as_posix(),
            meta=metadata,
        )
        record = self._resume_record(
            step_id=f"{self.plan_id}:dialogue_audio",
            step_type="dialogue_audio",
            artifact_path=timeline_path,
            meta=metadata,
        )
        return record, step_result, stage_result

    def resume_shot_images(
        self,
        shot: ShotSpec,
        *,
        base_images: Mapping[str, BaseImageSpec],
        base_image_assets: MutableMapping[str, _BaseImageAsset],
    ) -> Optional[Tuple[StepExecutionRecord, StepResult]]:
        if not self.enabled:
            return None
        summary_path = self.output_dir / "frames" / shot.id / "frames.json"
        payload = _load_json_object(summary_path)
        if payload is None:
            return None
        start_id = payload.get("start_base_image_id")
        end_id = payload.get("end_base_image_id")
        start_path = _resolve_existing_path(payload.get("start_frame_path"), base=summary_path.parent)
        end_path = _resolve_existing_path(payload.get("end_frame_path"), base=summary_path.parent)
        if start_id and start_path and start_path.exists():
            _hydrate_base_image_asset(base_image_assets, base_images, start_id, start_path)
        if end_id and end_path and end_path.exists():
            _hydrate_base_image_asset(base_image_assets, base_images, end_id, end_path)
        meta = {
            "shot_id": shot.id,
            "start_frame_path": start_path.as_posix() if start_path else payload.get("start_frame_path"),
            "end_frame_path": end_path.as_posix() if end_path else payload.get("end_frame_path"),
            "resume": True,
        }
        step_result = StepResult(path=summary_path, meta=meta)
        record = self._resume_record(
            step_id=f"{shot.id}:images",
            step_type="images",
            artifact_path=summary_path,
            meta=meta,
        )
        return record, step_result

    def resume_shot_tts(self, shot: ShotSpec) -> Optional[Tuple[StepExecutionRecord, StepResult, List[Path]]]:
        if not self.enabled:
            return None
        summary_path = self.output_dir / "audio" / shot.id / "tts_summary.json"
        payload = _load_json_object(summary_path)
        if payload is None:
            return None
        dialogue_paths = _resolve_existing_paths(payload.get("dialogue_paths"), base=summary_path.parent)
        line_entries = payload.get("lines") or []
        tts_meta: Dict[str, Any] = {
            "line_artifacts": line_entries,
            "lines_synthesized": len(line_entries),
            "dialogue_paths": [path.as_posix() for path in dialogue_paths],
            "total_duration_s": payload.get("total_duration_s"),
            "provider_id": payload.get("provider_id"),
            "voice_id": payload.get("voice_id"),
        }
        if payload.get("voice_metadata"):
            tts_meta["voice_metadata"] = payload["voice_metadata"]
        meta = {
            "tts": tts_meta,
            "dialogue_paths": [path.as_posix() for path in dialogue_paths],
            "lines": len(line_entries),
            "summary_path": summary_path.as_posix(),
            "resume": True,
        }
        artifact_uri = None
        if line_entries:
            first = line_entries[0]
            artifact_uri = first.get("artifact_uri")
        if artifact_uri is None and dialogue_paths:
            artifact_uri = dialogue_paths[0].as_posix()
        step_result = StepResult(
            path=summary_path,
            paths=tuple(dialogue_paths),
            artifact_uri=artifact_uri,
            meta=meta,
        )
        record = self._resume_record(
            step_id=f"{shot.id}:tts",
            step_type="tts",
            artifact_path=summary_path,
            meta=meta,
        )
        return record, step_result, dialogue_paths

    def resume_shot_video(self, shot: ShotSpec) -> Optional[Tuple[StepExecutionRecord, StepResult, Path]]:
        if not self.enabled:
            return None
        video_path = self.output_dir / "video" / f"{shot.id}.mp4"
        if not video_path.exists():
            return None
        meta = {"shot_id": shot.id, "duration_sec": shot.duration_sec, "resume": True}
        step_result = StepResult(path=video_path, artifact_uri=video_path.as_posix(), meta=meta)
        record = self._resume_record(
            step_id=f"{shot.id}:video",
            step_type="video",
            artifact_path=video_path,
            meta=meta,
        )
        return record, step_result, video_path

    def resume_shot_lipsync(self, shot: ShotSpec) -> Optional[Tuple[StepExecutionRecord, StepResult, Path]]:
        if not self.enabled:
            return None
        lipsync_path = self.output_dir / "lipsync" / f"{shot.id}.mp4"
        if not lipsync_path.exists():
            return None
        meta = {"shot_id": shot.id, "resume": True}
        step_result = StepResult(path=lipsync_path, artifact_uri=lipsync_path.as_posix(), meta=meta)
        record = self._resume_record(
            step_id=f"{shot.id}:lipsync",
            step_type="lipsync",
            artifact_path=lipsync_path,
            meta=meta,
        )
        return record, step_result, lipsync_path

    def resume_assemble_stage(self, plan: MoviePlan) -> Optional[Tuple[StepExecutionRecord, StepResult]]:
        if not self.enabled:
            return None
        assemble_path = self.output_dir / "final" / f"{self.plan_id}-assembly.json"
        if not assemble_path.exists():
            return None
        metadata = {
            "stage": "assemble",
            "shot_count": len(plan.shots),
            "resume": True,
        }
        step_result = StepResult(
            path=assemble_path,
            artifact_uri=assemble_path.as_posix(),
            meta=metadata,
        )
        record = self._resume_record(
            step_id=f"{self.plan_id}:assemble",
            step_type="assemble",
            artifact_path=assemble_path,
            meta=metadata,
        )
        return record, step_result

    def resume_finalize_stage(
        self,
        plan: MoviePlan,
        dialogue_result: Optional[_DialogueStageResult],
    ) -> Optional[Tuple[StepExecutionRecord, StepResult]]:
        if not self.enabled:
            return None
        final_path = self.output_dir / "final" / f"{self.plan_id}-video_final.mp4"
        if not final_path.exists():
            return None
        assemble_path = self.output_dir / "final" / f"{self.plan_id}-assembly.json"
        timeline_audio_path = dialogue_result.timeline_audio_path if dialogue_result else None
        total_duration = (
            dialogue_result.total_duration_s
            if dialogue_result
            else sum(shot.duration_sec for shot in plan.shots)
        )
        frame_rate = float(plan.render_profile.video.max_fps or _default_video_fps())
        resolution = _resolve_render_resolution(plan)
        checksum = _hash_file_path(final_path)
        size_bytes = final_path.stat().st_size if final_path.exists() else None
        metadata = {
            "stage": "finalize",
            "artifact_type": "video_final",
            "media_type": "video/mp4",
            "mime_type": "video/mp4",
            "local_path": final_path.as_posix(),
            "storage_hint": "local",
            "size_bytes": size_bytes,
            "duration_s": total_duration,
            "frame_rate": frame_rate,
            "resolution_px": resolution,
            "checksum_sha256": checksum,
            "playback_ready": final_path.exists(),
            "timeline_audio": timeline_audio_path.as_posix() if timeline_audio_path else None,
            "assemble_path": assemble_path.as_posix() if assemble_path.exists() else None,
            "resume": True,
        }
        step_result = StepResult(
            path=final_path,
            paths=(final_path,),
            artifact_uri=final_path.as_posix(),
            meta=metadata,
        )
        record = self._resume_record(
            step_id=f"{self.plan_id}:finalize",
            step_type="finalize",
            artifact_path=final_path,
            meta=metadata,
        )
        return record, step_result

    def _resume_record(
        self,
        *,
        step_id: str,
        step_type: str,
        artifact_path: Optional[Path],
        meta: Mapping[str, Any],
    ) -> StepExecutionRecord:
        artifact_uri = artifact_path.as_posix() if artifact_path else None
        meta_payload = dict(meta)
        meta_payload.setdefault("resume", True)
        now = _now_iso()
        return StepExecutionRecord(
            plan_id=self.plan_id,
            step_id=step_id,
            step_type=step_type,
            status="succeeded",
            start_time=now,
            end_time=now,
            duration_s=0.0,
            attempts=0,
            artifact_uri=artifact_uri,
            meta=meta_payload,
        )


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _resolve_existing_path(candidate: Optional[str], *, base: Path) -> Optional[Path]:
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = base / candidate
    if path.exists():
        return path
    return None


def _resolve_existing_paths(values: Optional[Iterable[Any]], *, base: Path) -> List[Path]:
    paths: List[Path] = []
    if not values:
        return paths
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = _resolve_existing_path(value, base=base)
        if candidate is not None:
            paths.append(candidate)
    return paths


def _hydrate_base_image_asset(
    assets: MutableMapping[str, _BaseImageAsset],
    base_images: Mapping[str, BaseImageSpec],
    image_id: str,
    asset_path: Path,
) -> None:
    spec = base_images.get(image_id)
    if spec is None or not asset_path.exists():
        return
    assets[image_id] = _BaseImageAsset(spec=spec, path=asset_path, payload_bytes=asset_path.read_bytes())


def execute_plan(
    plan: MoviePlan | Mapping[str, Any],
    *,
    mode: Literal["dry", "run"] = "dry",
    progress_callback: Optional[Callable[[StepExecutionRecord], None]] = None,
    config: Optional[ProductionAgentConfig] = None,
    run_id: Optional[str] = None,
    pre_step_hook: Optional[Callable[[str], None]] = None,
    resume: bool = False,
) -> ProductionResult:
    """Execute or simulate a MoviePlan."""

    cfg = config or ProductionAgentConfig()
    model = _coerce_plan(plan)
    _validate_plan(model)
    plan_id = _plan_identifier(model)
    provided_run_id = run_id
    if resume and mode != "run":
        raise ProductionAgentError("resume support requires run mode")
    run_id = provided_run_id or observability.get_session_id()
    if resume and provided_run_id is None:
        raise ProductionAgentError("resume support requires explicit run_id")

    def _execute_plan_intake_stage(output_dir: Optional[Path]) -> tuple[StepExecutionRecord, _PlanIntakeResult]:
        holder: Dict[str, _PlanIntakeResult] = {}

        def _plan_intake_action() -> StepResult:
            result = _run_plan_intake(model, plan_id=plan_id, run_id=run_id, output_dir=output_dir)
            holder["result"] = result
            if output_dir is not None and result.run_context_path is None:
                raise ProductionAgentError("plan intake failed to persist run_context")
            meta: Dict[str, Any] = {
                "artifact_type": "plan_run_context",
                "media_type": "application/json",
                "plan_uri": str(result.plan_path) if result.plan_path else None,
                "dialogue_timeline_uri": result.run_context.dialogue_timeline_uri,
                "base_image_map": dict(result.run_context.base_image_map),
                "schema_uris": result.schema_meta,
            }
            meta = {key: value for key, value in meta.items() if value is not None}
            _record_stage_manifest_entries(run_id=run_id, manifests=result.stage_manifests)
            meta.setdefault("stage_manifest_count", len(result.stage_manifests))
            artifact_path = result.run_context_path
            artifact_uri = artifact_path.as_posix() if artifact_path else None
            return StepResult(path=artifact_path, artifact_uri=artifact_uri, meta=meta)

        plan_record, _ = _run_step(
            plan_id=plan_id,
            run_id=run_id,
            step_id=f"{plan_id}:plan_intake",
            step_type="plan_intake",
            gate_flag=None,
            cfg=cfg,
            action=_plan_intake_action,
            meta={"stage": "plan_intake"},
            progress_callback=progress_callback,
            pre_step_hook=pre_step_hook,
        )
        plan_result = holder.get("result")
        if plan_result is None:
            raise ProductionAgentError("plan intake stage did not return a result")
        return plan_record, plan_result

    if mode == "dry":
        plan_record, plan_intake_result = _execute_plan_intake_stage(output_dir=None)
        report = _simulate_execution_report(model)
        _record_summary_event(run_id, plan_id, "dry", [plan_record], simulation=report)
        return ProductionResult([], steps=[plan_record], simulation_report=report)

    output_dir = _resolve_output_dir(run_id, plan_id)
    resume_snapshot = _ResumeSnapshot(enabled=resume, plan_id=plan_id, run_id=run_id, output_dir=output_dir)
    plan_record, plan_intake_result = _execute_plan_intake_stage(output_dir)
    records: List[StepExecutionRecord] = [plan_record]

    def _append_resume_record(record: StepExecutionRecord) -> None:
        _emit_step_record(run_id, record, progress_callback)
        records.append(record)

    shot_artifacts: List[_ShotArtifacts] = []

    base_images = plan_intake_result.base_image_lookup
    base_image_assets = plan_intake_result.base_image_assets
    voice_profiles = _character_voice_map(model)

    dialogue_stage_holder: Dict[str, _DialogueStageResult] = {}
    final_delivery_holder: Dict[str, adk_helpers.ArtifactRef] = {}
    finalize_result_holder: Dict[str, StepResult] = {}

    def _dialogue_stage_action() -> StepResult:
        result = _run_dialogue_stage(
            model,
            plan_id=plan_id,
            run_id=run_id,
            output_dir=output_dir,
            voice_profiles=voice_profiles,
        )
        dialogue_stage_holder["result"] = result
        _record_stage_manifest_entries(run_id=run_id, manifests=result.stage_manifests)
        meta_payload = {
            "stage": "dialogue_audio",
            "entry_count": len(result.line_entries),
            "line_artifacts": result.line_entries,
            "dialogue_paths": [path.as_posix() for path in result.line_paths],
            "timeline_audio_path": result.timeline_audio_path.as_posix(),
            "timeline_summary_path": result.summary_path.as_posix(),
            "total_duration_s": result.total_duration_s,
            "sample_rate": result.sample_rate,
            "channels": result.channels,
            "sample_width_bytes": result.sample_width,
            "timeline_offsets": result.timeline_offsets,
        }
        return StepResult(
            path=result.timeline_audio_path,
            paths=tuple(result.line_paths + [result.timeline_audio_path]),
            artifact_uri=result.timeline_audio_path.as_posix(),
            meta=meta_payload,
        )

    resumed_dialogue = resume_snapshot.resume_dialogue_stage()
    if resumed_dialogue:
        dialogue_record, _, dialogue_result = resumed_dialogue
        dialogue_stage_holder["result"] = dialogue_result
        _append_resume_record(dialogue_record)
    else:
        try:
            dialogue_record, _ = _run_step(
                plan_id=plan_id,
                run_id=run_id,
                step_id=f"{plan_id}:dialogue_audio",
                step_type="dialogue_audio",
                gate_flag=cfg.tts_flag,
                cfg=cfg,
                action=_dialogue_stage_action,
                meta={"stage": "dialogue_audio", "entry_count": len(model.dialogue_timeline)},
                progress_callback=progress_callback,
                pre_step_hook=pre_step_hook,
            )
            records.append(dialogue_record)
        except StepRateLimitError as exc:
            records.append(exc.record)
            _record_summary_event(run_id, plan_id, "run", records)
            raise

    try:
        for shot in model.shots:
            artifacts = _execute_shot(
                shot,
                plan_id=plan_id,
                run_id=run_id,
                output_dir=output_dir,
                cfg=cfg,
                progress_callback=progress_callback,
                pre_step_hook=pre_step_hook,
                records=records,
                voice_profiles=voice_profiles,
                base_images=base_images,
                base_image_assets=base_image_assets,
                resume_snapshot=resume_snapshot,
            )
            shot_artifacts.append(artifacts)
    except StepRateLimitError:
        _record_summary_event(run_id, plan_id, "run", records)
        raise

    final_step_result: Optional[StepResult] = None
    assemble_resume = resume_snapshot.resume_assemble_stage(model)
    if assemble_resume:
        assemble_record, final_step_result = assemble_resume
        _append_resume_record(assemble_record)
    else:
        try:
            assemble_record, final_step_result = _run_step(
                plan_id=plan_id,
                run_id=run_id,
                step_id=f"{plan_id}:assemble",
                step_type="assemble",
                gate_flag=None,
                cfg=cfg,
                progress_callback=progress_callback,
                pre_step_hook=pre_step_hook,
                action=lambda: _assemble_plan(model, shot_artifacts, output_dir),
                meta={"shot_count": len(model.shots)},
            )
            records.append(assemble_record)
        except StepRateLimitError as exc:
            records.append(exc.record)
            _record_summary_event(run_id, plan_id, "run", records)
            raise

    if final_step_result is None or final_step_result.path is None:
        raise ProductionAgentError("assemble stage did not produce an artifact path")

    assert final_step_result is not None

    manifest = _build_assemble_stage_manifest(
        run_id=run_id,
        plan_id=plan_id,
        plan_title=model.title,
        shot_count=len(model.shots),
        assemble_path=final_step_result.path,
    )
    _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])

    def _finalize_action() -> StepResult:
        dialogue_result = dialogue_stage_holder.get("result")
        assemble_path = final_step_result.path
        if assemble_path is None:
            raise ProductionAgentError("finalize stage missing assemble artifact path")
        final_artifact_ref, step_result = _run_final_delivery_stage(
            plan=model,
            plan_id=plan_id,
            run_id=run_id,
            output_dir=output_dir,
            shot_artifacts=shot_artifacts,
            dialogue_result=dialogue_result,
            assemble_path=assemble_path,
        )
        final_delivery_holder["artifact"] = final_artifact_ref
        if step_result.path:
            manifest = _build_finalize_stage_manifest(
                run_id=run_id,
                plan_id=plan_id,
                shot_count=len(model.shots),
                video_path=step_result.path,
                step_result=step_result,
            )
            _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])
        return step_result

    finalize_resume = resume_snapshot.resume_finalize_stage(model, dialogue_stage_holder.get("result"))
    if finalize_resume:
        finalize_record, finalize_step_result = finalize_resume
        _append_resume_record(finalize_record)
        finalize_result_holder["result"] = finalize_step_result
        if finalize_step_result.path:
            manifest = _build_finalize_stage_manifest(
                run_id=run_id,
                plan_id=plan_id,
                shot_count=len(model.shots),
                video_path=finalize_step_result.path,
                step_result=finalize_step_result,
            )
            _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])
            final_metadata = dict(finalize_step_result.meta or {})
            final_metadata.setdefault("plan_id", plan_id)
            final_metadata.setdefault("run_id", run_id)
            final_metadata.setdefault("stage", "finalize")
            final_artifact_ref = adk_helpers.publish_artifact(
                local_path=finalize_step_result.path,
                artifact_type="video_final",
                media_type=final_metadata.get("media_type") or final_metadata.get("mime_type") or "video/mp4",
                metadata=final_metadata,
                run_id=run_id,
            )
            final_delivery_holder["artifact"] = final_artifact_ref
    else:
        try:
            finalize_record, finalize_step_result = _run_step(
                plan_id=plan_id,
                run_id=run_id,
                step_id=f"{plan_id}:finalize",
                step_type="finalize",
                gate_flag=None,
                cfg=cfg,
                progress_callback=progress_callback,
                pre_step_hook=pre_step_hook,
                action=_finalize_action,
                meta={"stage": "finalize", "shot_count": len(model.shots)},
            )
            records.append(finalize_record)
            finalize_result_holder["result"] = finalize_step_result
        except StepRateLimitError as exc:
            records.append(exc.record)
            _record_summary_event(run_id, plan_id, "run", records)
            raise

    final_artifact_ref = final_delivery_holder.get("artifact")
    finalize_step_result = finalize_result_holder.get("result")
    artifact_ref: Optional[adk_helpers.ArtifactRef] = None
    if finalize_step_result and finalize_step_result.path:
        final_metadata = dict(finalize_step_result.meta or {})
        final_metadata.setdefault("plan_id", plan_id)
        final_metadata.setdefault("run_id", run_id)
        final_metadata.setdefault("stage", "finalize")
        artifact_ref = adk_helpers.publish_artifact(
            local_path=finalize_step_result.path,
            artifact_type=cfg.artifact_type,
            media_type=final_metadata.get("media_type") or "video/mp4",
            metadata=final_metadata,
            run_id=run_id,
        )

    telemetry_payload = {
        "plan_id": plan_id,
        "shot_count": len(model.shots),
    }
    if artifact_ref:
        telemetry_payload["artifact_uri"] = artifact_ref.get("uri")
    elif final_artifact_ref:
        telemetry_payload["artifact_uri"] = final_artifact_ref.get("uri")

    telemetry.emit_event(
        "production_agent.execute_plan.completed",
        telemetry_payload,
    )
    _record_summary_event(run_id, plan_id, "run", records)
    artifacts: List[adk_helpers.ArtifactRef] = []
    if artifact_ref:
        artifacts.append(artifact_ref)
    if final_artifact_ref:
        artifacts.append(final_artifact_ref)
    return ProductionResult(artifacts, steps=records)


def _coerce_plan(plan: MoviePlan | Mapping[str, Any]) -> MoviePlan:
    if isinstance(plan, MoviePlan):
        return plan
    try:
        if hasattr(MoviePlan, "model_validate"):
            return MoviePlan.model_validate(plan)  # type: ignore[attr-defined]
        return MoviePlan.parse_obj(plan)  # type: ignore[attr-defined]
    except ValidationError as exc:  # pragma: no cover - defensive
        raise ProductionAgentError(f"Invalid plan payload: {exc}") from exc


def _validate_plan(plan: MoviePlan) -> None:
    if not plan.shots:
        raise ProductionAgentError("MoviePlan must contain at least one shot")


def _build_base_image_lookup(plan: MoviePlan) -> Dict[str, BaseImageSpec]:
    lookup: Dict[str, BaseImageSpec] = {}
    for image in plan.base_images:
        if not image.id:
            raise ProductionAgentError("Base images must include an id")
        if image.id in lookup:
            raise ProductionAgentError(f"Duplicate base image id detected: {image.id}")
        lookup[image.id] = image
    if not lookup:
        raise ProductionAgentError("MoviePlan must include at least one base image")
    return lookup


def _base_image_prompt(
    base_images: Mapping[str, BaseImageSpec],
    image_id: str,
    *,
    shot_id: str,
    role: str,
    allow_empty: bool = False,
) -> str:
    try:
        spec = base_images[image_id]
    except KeyError as exc:
        raise ProductionAgentError(f"Shot {shot_id} references missing base image '{image_id}' for {role} frame") from exc
    prompt = (spec.prompt or "").strip()
    if not prompt and not allow_empty:
        raise ProductionAgentError(f"Base image '{image_id}' referenced by shot {shot_id} has an empty prompt")
    return prompt


def _run_plan_intake(
    plan: MoviePlan,
    *,
    plan_id: str,
    run_id: str,
    output_dir: Optional[Path],
) -> _PlanIntakeResult:
    base_image_lookup = _build_base_image_lookup(plan)
    plan_dir: Optional[Path] = None
    if output_dir is not None:
        plan_dir = output_dir / "plan"
        plan_dir.mkdir(parents=True, exist_ok=True)
    base_image_assets = _build_base_image_assets(plan.base_images, plan_dir)
    plan_path = _persist_plan_payload(plan, plan_dir)
    timeline_path = _persist_dialogue_timeline(plan, plan_dir)
    schema_meta = {
        "movie_plan": _schema_metadata(schema_registry.movie_plan_schema()),
        "run_context": _schema_metadata(schema_registry.run_context_schema()),
    }

    base_image_map: Dict[str, str] = {}
    for image_id, asset in base_image_assets.items():
        entry: Optional[str] = None
        if asset.path is not None:
            entry = asset.path.as_posix()
        else:
            uri_value = asset.spec.asset_uri or asset.spec.metadata.get("asset_uri")
            if isinstance(uri_value, str) and uri_value.strip():
                entry = uri_value
        base_image_map[image_id] = entry or asset.spec.prompt
    run_context = RunContext.from_plan(
        plan,
        run_id=run_id,
        schema_uri=schema_meta.get("run_context", {}).get("uri"),
        metadata={"schemas": schema_meta},
        dialogue_timeline_uri=timeline_path.as_posix() if timeline_path else None,
        base_image_map=base_image_map,
    )
    run_context_path = _persist_run_context(run_context, plan_dir)
    stage_manifests = _build_plan_intake_manifests(
        plan=plan,
        plan_id=plan_id,
        run_id=run_id,
        plan_path=plan_path,
        dialogue_timeline_path=timeline_path,
        run_context_path=run_context_path,
        schema_meta=schema_meta,
        base_image_map=base_image_map,
    )

    return _PlanIntakeResult(
        base_image_lookup=base_image_lookup,
        base_image_assets=base_image_assets,
        run_context=run_context,
        run_context_path=run_context_path,
        plan_path=plan_path,
        dialogue_timeline_path=timeline_path,
        schema_meta=schema_meta,
        stage_manifests=stage_manifests,
    )


def _run_dialogue_stage(
    plan: MoviePlan,
    *,
    plan_id: str,
    run_id: str,
    output_dir: Optional[Path],
    voice_profiles: Mapping[str, Mapping[str, Any]],
) -> _DialogueStageResult:
    if output_dir is None:
        raise ProductionAgentError("dialogue stage requires an output directory in run mode")
    class _AgentSynthesizer(DialogueSynthesizer):
        def synthesize(
            self,
            text: str,
            *,
            voice_config: Mapping[str, Any],
            plan_id: str,
            step_id: str,
            run_id: str,
            output_dir: Path,
        ) -> Mapping[str, Any]:
            return tts_stage.synthesize(
                text,
                voice_config=voice_config,
                plan_id=plan_id,
                step_id=step_id,
                run_id=run_id,
                output_dir=output_dir,
            )

    def _resolve_voice(character_id: Optional[str]) -> Mapping[str, Any]:
        return _voice_config_for_character(character_id, voice_profiles, None)

    builder = DialogueTimelineBuilder(
        synthesizer=_AgentSynthesizer(),
        voice_resolver=_resolve_voice,
    )

    try:
        build = builder.build(
            plan,
            plan_id=plan_id,
            run_id=run_id,
            output_dir=output_dir,
        )
    except DialogueTimelineError as exc:
        raise ProductionAgentError(str(exc)) from exc

    line_entries = build.line_entries
    line_paths = build.line_paths
    summary_path = build.summary_path
    timeline_audio_path = build.timeline_audio_path
    total_duration = build.total_duration_s
    sample_rate = build.sample_rate
    sample_width = build.sample_width
    channels = build.channels
    offsets = build.timeline_offsets

    stage_manifest_schema_meta = _schema_metadata(schema_registry.stage_manifest_schema())
    manifests: List[StageManifest] = []

    def _manifest_metadata(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "plan_id": plan_id,
            "stage": "dialogue_audio",
            "entry_count": len(line_entries),
            "stage_manifest_schema": stage_manifest_schema_meta,
        }
        if extra:
            payload.update(extra)
        return _stage_event_metadata(payload)

    manifests.append(
        StageManifest(
            run_id=run_id,
            stage_id="dialogue_audio",
            artifact_type="dialogue_timeline_audio",
            name=summary_path.name,
            artifact_uri=summary_path.as_posix(),
            media_type="application/json",
            local_path=summary_path.as_posix(),
            storage_hint="local",
            mime_type="application/json",
            size_bytes=summary_path.stat().st_size,
            metadata=_manifest_metadata({"summary": True, "total_duration_s": total_duration}),
            playback_ready=False,
        )
    )

    manifests.append(
        StageManifest(
            run_id=run_id,
            stage_id="dialogue_audio",
            artifact_type="tts_timeline_audio",
            name=timeline_audio_path.name,
            artifact_uri=timeline_audio_path.as_posix(),
            media_type="audio/wav",
            local_path=timeline_audio_path.as_posix(),
            storage_hint="local",
            mime_type="audio/wav",
            size_bytes=timeline_audio_path.stat().st_size,
            duration_s=total_duration,
            metadata=_manifest_metadata(
                {
                    "entry_count": len(line_entries),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width_bytes": sample_width,
                }
            ),
            playback_ready=True,
        )
    )

    return _DialogueStageResult(
        line_entries=line_entries,
        line_paths=line_paths,
        summary_path=summary_path,
        timeline_audio_path=timeline_audio_path,
        total_duration_s=total_duration,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        timeline_offsets=offsets,
        stage_manifests=manifests,
    )


def _build_plan_intake_manifests(
    *,
    plan: MoviePlan,
    plan_id: str,
    run_id: str,
    plan_path: Optional[Path],
    dialogue_timeline_path: Optional[Path],
    run_context_path: Optional[Path],
    schema_meta: Mapping[str, Dict[str, str]],
    base_image_map: Mapping[str, str],
) -> List[StageManifest]:
    manifests: List[StageManifest] = []
    stage_id = "plan_intake"
    stage_manifest_schema_meta = _schema_metadata(schema_registry.stage_manifest_schema())

    def _make_metadata(schema_key: Optional[str], extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "plan_id": plan_id,
            "stage_manifest_schema": stage_manifest_schema_meta,
        }
        if schema_key:
            schema_info = schema_meta.get(schema_key)
            if schema_info:
                metadata["schema"] = schema_info
        if extra:
            metadata.update(extra)
        return metadata
    
    def _append_manifest(
        *,
        path: Optional[Path],
        artifact_type: str,
        name: str,
        schema_key: Optional[str],
        extra_meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if path is None:
            return
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        metadata_payload = _stage_event_metadata(_make_metadata(schema_key, extra_meta))
        manifest = StageManifest(
            run_id=run_id,
            stage_id=stage_id,
            artifact_type=artifact_type,
            name=name,
            artifact_uri=path.as_posix(),
            media_type="application/json",
            local_path=path.as_posix(),
            storage_hint="local",
            mime_type="application/json",
            size_bytes=size_bytes,
            metadata=metadata_payload,
            playback_ready=True,
        )
        manifests.append(manifest)

    _append_manifest(
        path=plan_path,
        artifact_type="movie_plan",
        name="movie_plan.json",
        schema_key="movie_plan",
        extra_meta={
            "shot_count": len(plan.shots),
            "base_image_count": len(plan.base_images),
        },
    )
    _append_manifest(
        path=dialogue_timeline_path,
        artifact_type="dialogue_timeline",
        name="dialogue_timeline.json",
        schema_key=None,
        extra_meta={
            "entry_count": len(plan.dialogue_timeline),
        },
    )
    _append_manifest(
        path=run_context_path,
        artifact_type="plan_run_context",
        name="run_context.json",
        schema_key="run_context",
        extra_meta={
            "base_image_map": dict(base_image_map),
            "dialogue_timeline_uri": dialogue_timeline_path.as_posix() if dialogue_timeline_path else None,
        },
    )

    return manifests


def _file_size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _build_shot_frames_manifest(
    *,
    run_id: str,
    plan_id: str,
    shot: ShotSpec,
    frames_path: Path,
    meta: Mapping[str, Any],
) -> StageManifest:
    metadata_payload = {
        "plan_id": plan_id,
        "shot_id": shot.id,
        "stage": "images",
        "start_frame_path": meta.get("start_frame_path"),
        "end_frame_path": meta.get("end_frame_path"),
    }
    metadata = _stage_event_metadata(metadata_payload)
    return StageManifest(
        run_id=run_id,
        stage_id=f"{shot.id}:images",
        artifact_type="shot_frames",
        name=frames_path.name,
        artifact_uri=frames_path.as_posix(),
        media_type="application/json",
        local_path=frames_path.as_posix(),
        storage_hint="local",
        mime_type="application/json",
        size_bytes=_file_size(frames_path),
        metadata=metadata,
        playback_ready=False,
    )


def _build_shot_tts_manifest(
    *,
    run_id: str,
    plan_id: str,
    shot: ShotSpec,
    summary_path: Path,
    dialogue_paths: Sequence[Path],
    meta: Mapping[str, Any],
) -> StageManifest:
    tts_meta = dict(meta.get("tts") or {})
    dialogue_path_values = [path.as_posix() for path in dialogue_paths]
    metadata_payload = {
        "plan_id": plan_id,
        "shot_id": shot.id,
        "stage": "tts",
        "dialogue_paths": dialogue_path_values,
        "lines": meta.get("lines") or tts_meta.get("lines_synthesized"),
        "tts": tts_meta,
    }
    metadata = _stage_event_metadata(metadata_payload)
    metadata.setdefault("summary_path", summary_path.as_posix())
    return StageManifest(
        run_id=run_id,
        stage_id=f"{shot.id}:tts",
        artifact_type="shot_dialogue_audio",
        name=summary_path.name,
        artifact_uri=summary_path.as_posix(),
        media_type="application/json",
        local_path=summary_path.as_posix(),
        storage_hint="local",
        mime_type="application/json",
        size_bytes=_file_size(summary_path),
        metadata=metadata,
        playback_ready=False,
    )


def _build_shot_video_manifest(
    *,
    run_id: str,
    plan_id: str,
    shot: ShotSpec,
    video_path: Path,
    shot_duration: float,
) -> StageManifest:
    metadata_payload = {
        "plan_id": plan_id,
        "shot_id": shot.id,
        "stage": "video",
        "duration_s": shot_duration,
    }
    metadata = _stage_event_metadata(metadata_payload)
    return StageManifest(
        run_id=run_id,
        stage_id=f"{shot.id}:video",
        artifact_type="shot_video",
        name=video_path.name,
        artifact_uri=video_path.as_posix(),
        media_type="video/mp4",
        local_path=video_path.as_posix(),
        storage_hint="local",
        mime_type="video/mp4",
        size_bytes=_file_size(video_path),
        duration_s=shot_duration,
        metadata=metadata,
        playback_ready=True,
    )


def _build_shot_lipsync_manifest(
    *,
    run_id: str,
    plan_id: str,
    shot: ShotSpec,
    lipsync_path: Path,
    dialogue_paths: Sequence[Path],
) -> StageManifest:
    dialogue_path_values = [path.as_posix() for path in dialogue_paths]
    metadata_payload = {
        "plan_id": plan_id,
        "shot_id": shot.id,
        "stage": "lipsync",
        "dialogue_paths": dialogue_path_values,
    }
    metadata = _stage_event_metadata(metadata_payload)
    return StageManifest(
        run_id=run_id,
        stage_id=f"{shot.id}:lipsync",
        artifact_type="shot_lipsync_video",
        name=lipsync_path.name,
        artifact_uri=lipsync_path.as_posix(),
        media_type="video/mp4",
        local_path=lipsync_path.as_posix(),
        storage_hint="local",
        mime_type="video/mp4",
        size_bytes=_file_size(lipsync_path),
        metadata=metadata,
        playback_ready=True,
    )


def _build_assemble_stage_manifest(
    *,
    run_id: str,
    plan_id: str,
    plan_title: str,
    shot_count: int,
    assemble_path: Path,
) -> StageManifest:
    metadata_payload = {
        "plan_id": plan_id,
        "stage": "assemble",
        "shot_count": shot_count,
        "plan_title": plan_title,
    }
    metadata = _stage_event_metadata(metadata_payload)
    return StageManifest(
        run_id=run_id,
        stage_id="assemble",
        artifact_type="assembly_plan",
        name=assemble_path.name,
        artifact_uri=assemble_path.as_posix(),
        media_type="application/json",
        local_path=assemble_path.as_posix(),
        storage_hint="local",
        mime_type="application/json",
        size_bytes=_file_size(assemble_path),
        metadata=metadata,
        playback_ready=False,
    )


def _build_finalize_stage_manifest(
    *,
    run_id: str,
    plan_id: str,
    shot_count: int,
    video_path: Path,
    step_result: StepResult,
) -> StageManifest:
    meta_payload: Dict[str, Any] = {
        "plan_id": plan_id,
        "stage": "finalize",
        "shot_count": shot_count,
    }
    meta_payload.update(dict(step_result.meta or {}))
    metadata = _stage_event_metadata(meta_payload)
    local_path_value = meta_payload.get("local_path")
    local_path = local_path_value if isinstance(local_path_value, str) and local_path_value else video_path.as_posix()
    download_url_value = meta_payload.get("download_url")
    download_url = download_url_value if isinstance(download_url_value, str) else None
    artifact_uri_value = meta_payload.get("artifact_uri")
    artifact_uri = step_result.artifact_uri or artifact_uri_value or local_path
    storage_hint_value = meta_payload.get("storage_hint")
    storage_hint = storage_hint_value if isinstance(storage_hint_value, str) else "local"
    size_bytes_value = meta_payload.get("size_bytes")
    size_bytes = size_bytes_value if isinstance(size_bytes_value, int) else _file_size(video_path)
    duration_value = meta_payload.get("duration_s")
    duration_s = float(duration_value) if isinstance(duration_value, (int, float)) else None
    frame_rate_value = meta_payload.get("frame_rate")
    frame_rate = float(frame_rate_value) if isinstance(frame_rate_value, (int, float)) else None
    resolution_value = meta_payload.get("resolution_px")
    resolution_px = str(resolution_value) if isinstance(resolution_value, str) else None
    checksum_value = meta_payload.get("checksum_sha256")
    checksum_sha256 = checksum_value if isinstance(checksum_value, str) else None
    playback_ready_value = meta_payload.get("playback_ready")
    playback_ready = bool(playback_ready_value) if isinstance(playback_ready_value, bool) else video_path.exists()
    media_type_value = meta_payload.get("media_type") or meta_payload.get("mime_type") or "video/mp4"
    media_type = str(media_type_value)
    mime_type = str(meta_payload.get("mime_type") or media_type)
    notes_value = meta_payload.get("notes")
    notes = str(notes_value) if isinstance(notes_value, str) else None
    name = Path(local_path).name if local_path else video_path.name
    return StageManifest(
        run_id=run_id,
        stage_id="finalize",
        artifact_type="video_final",
        name=name,
        artifact_uri=artifact_uri,
        media_type=media_type,
        local_path=local_path,
        download_url=download_url,
        storage_hint=storage_hint,
        mime_type=mime_type,
        size_bytes=size_bytes,
        duration_s=duration_s,
        frame_rate=frame_rate,
        resolution_px=resolution_px,
        checksum_sha256=checksum_sha256,
        playback_ready=playback_ready,
        notes=notes,
        metadata=metadata,
    )

def _base_image_extension(spec: BaseImageSpec, source_path: Optional[Path]) -> str:
    mime = (spec.mime_type or spec.metadata.get("mime_type") or "").strip().lower()
    if mime in {"image/png", "image/apng"}:
        return ".png"
    if mime in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if mime == "image/webp":
        return ".webp"
    if source_path and source_path.suffix:
        return source_path.suffix
    return ".bin"


def _read_base_image_payload(spec: BaseImageSpec) -> tuple[Optional[bytes], Optional[Path]]:
    for path in _base_image_candidate_paths(spec):
        try:
            return path.read_bytes(), path
        except OSError:
            continue
    return None, None


def _base_image_candidate_paths(spec: BaseImageSpec) -> List[Path]:
    candidates: List[Path] = []
    for value in (
        spec.local_path,
        spec.metadata.get("local_path"),
        spec.metadata.get("asset_path"),
    ):
        path = _maybe_path_from_string(value)
        if path is not None:
            candidates.append(path)
    for value in (spec.asset_uri, spec.metadata.get("asset_uri")):
        path = _maybe_path_from_uri(value)
        if path is not None:
            candidates.append(path)
    return candidates


def _maybe_path_from_string(value: Any) -> Optional[Path]:
    if not value or not isinstance(value, str):
        return None
    try:
        path = Path(value).expanduser()
    except (OSError, TypeError):
        return None
    return path if path.exists() else None


def _maybe_path_from_uri(value: Any) -> Optional[Path]:
    if not value or not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"", "file"}:
        return None
    target = parsed.path if parsed.scheme else value
    try:
        path = Path(unquote(target)).expanduser()
    except (OSError, TypeError):
        return None
    return path if path.exists() else None


def _build_base_image_assets(base_images: Sequence[BaseImageSpec], plan_dir: Optional[Path]) -> Dict[str, _BaseImageAsset]:
    def _write_metadata(spec: BaseImageSpec, target_dir: Path) -> Path:
        target = target_dir / f"{spec.id}.json"
        payload = _model_dump(spec)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return target

    assets: Dict[str, _BaseImageAsset] = {}
    base_image_dir = plan_dir / "base_images" if plan_dir else None
    if base_image_dir is not None:
        base_image_dir.mkdir(parents=True, exist_ok=True)

    for spec in base_images:
        metadata_path: Optional[Path] = None
        if base_image_dir is not None:
            metadata_path = _write_metadata(spec, base_image_dir)

        payload_bytes, source_path = _read_base_image_payload(spec)
        if payload_bytes is None:
            payload_bytes = json.dumps(_model_dump(spec), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")

        asset_path = source_path
        if base_image_dir is not None:
            suffix = _base_image_extension(spec, source_path)
            asset_path = base_image_dir / f"{spec.id}{suffix}"
            asset_path.write_bytes(payload_bytes)
        elif asset_path is None:
            asset_path = source_path or metadata_path

        assets[spec.id] = _BaseImageAsset(spec=spec, path=asset_path, payload_bytes=payload_bytes)
    return assets


def _persist_plan_payload(plan: MoviePlan, plan_dir: Optional[Path]) -> Optional[Path]:
    if plan_dir is None:
        return None
    path = plan_dir / "movie_plan.json"
    path.write_text(json.dumps(_model_dump(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _persist_dialogue_timeline(plan: MoviePlan, plan_dir: Optional[Path]) -> Optional[Path]:
    if plan_dir is None:
        return None
    timeline_path = plan_dir / "dialogue_timeline.json"
    timeline_payload = [_model_dump(entry) for entry in plan.dialogue_timeline]
    timeline_path.write_text(json.dumps(timeline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return timeline_path


def _persist_run_context(run_context: RunContext, plan_dir: Optional[Path]) -> Optional[Path]:
    if plan_dir is None:
        return None
    path = plan_dir / "run_context.json"
    payload = run_context.model_dump() if hasattr(run_context, "model_dump") else run_context.dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _schema_metadata(artifact: schema_registry.SchemaArtifact) -> Dict[str, str]:
    meta: Dict[str, str] = {"uri": artifact.uri}
    local_path = artifact.local_path
    if local_path and local_path.exists():
        meta["sha256"] = _hash_file_path(local_path)
    return meta


def _hash_file_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()  # type: ignore[attr-defined]
    return value


def _simulate_execution_report(plan: MoviePlan) -> SimulationReport:
    steps: List[SimulationStep] = []
    for shot in plan.shots:
        steps.append(
            SimulationStep(
                step_id=f"{shot.id}:images",
                step_type="images",
                estimated_runtime_s=max(shot.duration_sec * 0.2, 1.0),
                estimated_gpu_memory_mb=512,
                simulated_artifact_uri=f"dry-run://frames/{shot.id}",
            )
        )
        if shot.dialogue:
            steps.append(
                SimulationStep(
                    step_id=f"{shot.id}:tts",
                    step_type="tts",
                    estimated_runtime_s=0.5 * len(shot.dialogue),
                    estimated_gpu_memory_mb=64,
                    simulated_artifact_uri=f"dry-run://audio/{shot.id}",
                )
            )
        steps.append(
            SimulationStep(
                step_id=f"{shot.id}:video",
                step_type="video",
                estimated_runtime_s=max(shot.duration_sec, 2.0),
                estimated_gpu_memory_mb=2048,
                simulated_artifact_uri=f"dry-run://video/{shot.id}",
            )
        )
    total_runtime = sum(step.estimated_runtime_s for step in steps)
    summary = {
        "total_estimated_runtime_s": total_runtime,
        "total_estimated_gpu_hours": total_runtime / 3600.0,
    }
    return SimulationReport(plan_id=_plan_identifier(plan), steps=steps, resource_summary=summary)


def _plan_identifier(plan: MoviePlan) -> str:
    slug = plan.metadata.get("plan_id") if plan.metadata else None
    if not slug:
        slug = plan.title.lower().replace(" ", "-")[:32] or "movie-plan"
    return slug


def _resolve_output_dir(run_id: str, plan_id: str) -> Path:
    base = os.environ.get("SPARKLE_LOCAL_RUNS_ROOT")
    if base:
        root = Path(base)
    else:
        root = Path(__file__).resolve().parents[2] / "artifacts" / "runs"
    dest = root / run_id / plan_id
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _character_voice_map(plan: MoviePlan) -> Mapping[str, Mapping[str, Any]]:
    voices: Dict[str, Mapping[str, Any]] = {}
    for character in plan.characters:
        if character.voice_profile:
            voices[character.id] = dict(character.voice_profile)
    return voices


def _voice_config_for_shot(shot: ShotSpec, voice_profiles: Mapping[str, Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for line in shot.dialogue:
        profile = voice_profiles.get(line.character_id)
        if profile:
            return dict(profile)
    return None


def _voice_config_for_character(
    character_id: Optional[str],
    voice_profiles: Mapping[str, Mapping[str, Any]],
    fallback: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if character_id:
        profile = voice_profiles.get(character_id)
        if profile:
            return dict(profile)
    return dict(fallback) if fallback else None

def _execute_shot(
    shot: ShotSpec,
    *,
    plan_id: str,
    run_id: str,
    output_dir: Path,
    cfg: ProductionAgentConfig,
    progress_callback: Optional[Callable[[StepExecutionRecord], None]],
    pre_step_hook: Optional[Callable[[str], None]],
    records: List[StepExecutionRecord],
    voice_profiles: Mapping[str, Mapping[str, Any]],
    base_images: Mapping[str, BaseImageSpec],
    base_image_assets: MutableMapping[str, _BaseImageAsset],
    resume_snapshot: _ResumeSnapshot,
) -> _ShotArtifacts:
    artifacts = _ShotArtifacts(shot_id=shot.id)

    def _append_record(record: StepExecutionRecord) -> None:
        records.append(record)

    def _append_resume_record(record: StepExecutionRecord) -> None:
        _emit_step_record(run_id, record, progress_callback)
        _append_record(record)

    def _run_with_tracking(**kwargs: Any) -> tuple[StepExecutionRecord, StepResult]:
        try:
            record, step_result = _run_step(**kwargs)
        except StepRateLimitError as exc:
            _append_record(exc.record)
            raise
        _append_record(record)
        return record, step_result

    frames_resume = resume_snapshot.resume_shot_images(
        shot,
        base_images=base_images,
        base_image_assets=base_image_assets,
    )
    if frames_resume:
        frames_record, frames_result = frames_resume
        _append_resume_record(frames_record)
        if frames_result.path:
            manifest = _build_shot_frames_manifest(
                run_id=run_id,
                plan_id=plan_id,
                shot=shot,
                frames_path=frames_result.path,
                meta=dict(frames_result.meta or {}),
            )
            _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])
    else:
        frames_record, frames_result = _run_with_tracking(
            plan_id=plan_id,
            run_id=run_id,
            step_id=f"{shot.id}:images",
            step_type="images",
            gate_flag=cfg.adapters_flag,
            cfg=cfg,
            pre_step_hook=pre_step_hook,
            progress_callback=progress_callback,
            action=lambda: _render_frames(shot, output_dir, base_images, base_image_assets),
            meta={"shot_id": shot.id},
        )
        if frames_record.status == "succeeded" and frames_result.path:
            manifest = _build_shot_frames_manifest(
                run_id=run_id,
                plan_id=plan_id,
                shot=shot,
                frames_path=frames_result.path,
                meta=dict(frames_result.meta or {}),
            )
            _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])
    artifacts.frames_path = frames_result.path

    if shot.dialogue:
        tts_resume = resume_snapshot.resume_shot_tts(shot)
        if tts_resume:
            tts_record, tts_result, dialogue_paths = tts_resume
            artifacts.dialogue_paths = list(dialogue_paths)
            _append_resume_record(tts_record)
            if tts_result.path and artifacts.dialogue_paths:
                manifest = _build_shot_tts_manifest(
                    run_id=run_id,
                    plan_id=plan_id,
                    shot=shot,
                    summary_path=tts_result.path,
                    dialogue_paths=artifacts.dialogue_paths,
                    meta=dict(tts_result.meta or {}),
                )
                _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])
        else:
            tts_record, tts_result = _run_with_tracking(
                plan_id=plan_id,
                run_id=run_id,
                step_id=f"{shot.id}:tts",
                step_type="tts",
                gate_flag=cfg.tts_flag,
                cfg=cfg,
                pre_step_hook=pre_step_hook,
                progress_callback=progress_callback,
                action=lambda: _synthesize_dialogue(
                    shot,
                    output_dir,
                    plan_id=plan_id,
                    run_id=run_id,
                    voice_profiles=voice_profiles,
                ),
                meta={
                    "shot_id": shot.id,
                    "lines": len(shot.dialogue),
                    "characters": sorted({line.character_id for line in shot.dialogue if line.character_id}),
                },
            )
            artifacts.dialogue_paths = list(tts_result.paths or ([tts_result.path] if tts_result.path else []))
            if tts_record.status == "succeeded" and tts_result.path and artifacts.dialogue_paths:
                manifest = _build_shot_tts_manifest(
                    run_id=run_id,
                    plan_id=plan_id,
                    shot=shot,
                    summary_path=tts_result.path,
                    dialogue_paths=artifacts.dialogue_paths,
                    meta=dict(tts_result.meta or {}),
                )
                _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])

    video_resume = resume_snapshot.resume_shot_video(shot)
    if video_resume:
        video_record, video_result, video_path = video_resume
        artifacts.video_path = video_path
        _append_resume_record(video_record)
        if video_result.path:
            manifest = _build_shot_video_manifest(
                run_id=run_id,
                plan_id=plan_id,
                shot=shot,
                video_path=video_result.path,
                shot_duration=shot.duration_sec,
            )
            _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])
    else:
        video_record, video_result = _run_with_tracking(
            plan_id=plan_id,
            run_id=run_id,
            step_id=f"{shot.id}:video",
            step_type="video",
            gate_flag=cfg.adapters_flag,
            cfg=cfg,
            pre_step_hook=pre_step_hook,
            progress_callback=progress_callback,
            action=lambda: _render_video_clip(
                shot,
                output_dir,
                plan_id,
                run_id,
                base_images,
                base_image_assets,
                progress_callback,
            ),
            meta={"shot_id": shot.id, "duration_sec": shot.duration_sec},
        )
        artifacts.video_path = video_result.path
        if video_record.status == "succeeded" and video_result.path:
            manifest = _build_shot_video_manifest(
                run_id=run_id,
                plan_id=plan_id,
                shot=shot,
                video_path=video_result.path,
                shot_duration=shot.duration_sec,
            )
            _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])


    if shot.is_talking_closeup and artifacts.dialogue_paths and artifacts.video_path:
        lipsync_resume = resume_snapshot.resume_shot_lipsync(shot)
        if lipsync_resume:
            lipsync_record, lipsync_result, lipsync_path = lipsync_resume
            artifacts.lipsync_path = lipsync_path
            _append_resume_record(lipsync_record)
            if lipsync_result.path:
                manifest = _build_shot_lipsync_manifest(
                    run_id=run_id,
                    plan_id=plan_id,
                    shot=shot,
                    lipsync_path=lipsync_result.path,
                    dialogue_paths=artifacts.dialogue_paths,
                )
                _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])
        else:
            lipsync_record, lipsync_result = _run_with_tracking(
                plan_id=plan_id,
                run_id=run_id,
                step_id=f"{shot.id}:lipsync",
                step_type="lipsync",
                gate_flag=cfg.lipsync_flag,
                cfg=cfg,
                pre_step_hook=pre_step_hook,
                progress_callback=progress_callback,
                action=lambda: _lipsync_clip(shot, output_dir),
                meta={"shot_id": shot.id},
            )
            artifacts.lipsync_path = lipsync_result.path
            if lipsync_record.status == "succeeded" and lipsync_result.path:
                manifest = _build_shot_lipsync_manifest(
                    run_id=run_id,
                    plan_id=plan_id,
                    shot=shot,
                    lipsync_path=lipsync_result.path,
                    dialogue_paths=artifacts.dialogue_paths,
                )
                _record_stage_manifest_entries(run_id=run_id, manifests=[manifest])

    return artifacts


def _run_step(
    *,
    plan_id: str,
    run_id: str,
    step_id: str,
    step_type: str,
    gate_flag: Optional[str],
    cfg: ProductionAgentConfig,
    action: Callable[[], StepActionReturn],
    meta: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[StepExecutionRecord], None]] = None,
    pre_step_hook: Optional[Callable[[str], None]] = None,
    enabled: bool = True,
) -> tuple[StepExecutionRecord, StepResult]:

    def _normalize_step_result(value: StepActionReturn) -> StepResult:
        if isinstance(value, StepResult):
            return value
        if value is None:
            return StepResult()
        if isinstance(value, Path):
            return StepResult(path=value)
        if isinstance(value, str):
            return StepResult(path=Path(value))
        raise TypeError(f"Unsupported step action return type: {type(value)!r}")

    should_execute = enabled and (gate_flag is None or _flag_enabled(gate_flag))
    start = _now_iso()
    start_dt = datetime.now(timezone.utc)
    attempts = 0
    artifact_path: Optional[Path] = None
    status: Literal["queued", "succeeded", "failed", "simulated"]
    error_type: Optional[str] = None
    rate_limit_state: Optional[str] = None
    meta_payload: Dict[str, Any] = dict(meta or {})

    step_result = StepResult()
    base_event_info = {"step_id": step_id, "gate_flag": gate_flag}

    if not should_execute:
        status = "skipped" if not enabled else "simulated"
        event_meta = {**base_event_info}
        if not enabled:
            event_meta["skipped"] = True
        else:
            event_meta["simulated"] = True
        _record_stage_event(
            run_id=run_id,
            stage=step_type,
            status="success",
            attempt=1,
            metadata=_stage_event_metadata(event_meta),
        )
    else:
        status = "failed"
        for attempt in range(1, max(cfg.max_attempts, 1) + 1):
            attempts = attempt
            if pre_step_hook:
                pre_step_hook(step_id)
            _record_stage_event(
                run_id=run_id,
                stage=step_type,
                status="begin",
                attempt=attempt,
                metadata=_stage_event_metadata({**base_event_info, **meta_payload, "attempt": attempt}),
            )
            try:
                step_result = _normalize_step_result(action())
                artifact_path = step_result.path or (step_result.paths[0] if step_result.paths else None)
                status = "succeeded"
                meta_payload.update(dict(step_result.meta or {}))
                artifact_uri = step_result.artifact_uri or (str(artifact_path) if artifact_path else None)
                _record_stage_event(
                    run_id=run_id,
                    stage=step_type,
                    status="success",
                    attempt=attempt,
                    metadata=_stage_event_metadata(
                        {
                            **base_event_info,
                            **meta_payload,
                            "attempt": attempt,
                            "artifact_uri": artifact_uri,
                        }
                    ),
                )
                break
            except RateLimitQueued as exc:
                status = "queued"  # type: ignore[assignment]
                error_type = exc.__class__.__name__
                meta_payload["rate_limit"] = _rate_limit_meta(exc.decision)
                rate_limit_state = "queued"
                _record_stage_event(
                    run_id=run_id,
                    stage=step_type,
                    status="fail",
                    attempt=attempt,
                    metadata=_stage_event_metadata({**base_event_info, **meta_payload, "attempt": attempt}),
                    error=str(exc),
                )
                break
            except RateLimitExceeded as exc:
                status = "failed"
                error_type = exc.__class__.__name__
                meta_payload["rate_limit"] = _rate_limit_meta(exc.decision)
                rate_limit_state = "exceeded"
                _record_stage_event(
                    run_id=run_id,
                    stage=step_type,
                    status="fail",
                    attempt=attempt,
                    metadata=_stage_event_metadata({**base_event_info, **meta_payload, "attempt": attempt}),
                    error=str(exc),
                )
                break
            except StepTransientError as exc:
                _record_stage_event(
                    run_id=run_id,
                    stage=step_type,
                    status="fail",
                    attempt=attempt,
                    metadata=_stage_event_metadata({**base_event_info, **meta_payload, "attempt": attempt}),
                    error=str(exc),
                )
                if attempt >= cfg.max_attempts:
                    error_type = "StepTransientError"
                    break
                time.sleep(cfg.retry_delay(attempt))
                continue
            except Exception as exc:  # pragma: no cover - defensive fallback
                error_type = exc.__class__.__name__
                _record_stage_event(
                    run_id=run_id,
                    stage=step_type,
                    status="fail",
                    attempt=attempt,
                    metadata=_stage_event_metadata({**base_event_info, **meta_payload, "attempt": attempt}),
                    error=str(exc),
                )
                break
        else:  # pragma: no cover
            error_type = "UnknownError"
            _record_stage_event(
                run_id=run_id,
                stage=step_type,
                status="fail",
                attempt=attempts or 1,
                metadata=_stage_event_metadata({**base_event_info, **meta_payload, "attempt": attempts or 1}),
                error="UnknownError",
            )

    end_dt = datetime.now(timezone.utc)
    duration = (end_dt - start_dt).total_seconds()
    artifact_uri = step_result.artifact_uri or (str(artifact_path) if artifact_path else None)
    record = StepExecutionRecord(
        plan_id=plan_id,
        step_id=step_id,
        step_type=step_type,
        status=status,  # type: ignore[arg-type]
        start_time=start,
        end_time=_to_iso(end_dt),
        duration_s=duration,
        attempts=attempts,
        model_id=step_result.model_id,
        device=step_result.device,
        memory_hint_mb=step_result.memory_hint_mb,
        logs_uri=step_result.logs_uri,
        artifact_uri=artifact_uri,
        error_type=error_type,
        meta=meta_payload,
    )
    _emit_step_record(run_id, record, progress_callback)
    if should_execute:
        if status == "succeeded":
            return record, step_result
        if rate_limit_state == "queued":
            raise StepQueuedError(f"{step_id} queued by rate limiter", record=record)
        if rate_limit_state == "exceeded":
            raise StepRateLimitExceededError(f"{step_id} hit rate limit", record=record)
        raise StepExecutionError(f"{step_id} failed after {attempts} attempts")
    return record, step_result




def _emit_step_record(
    run_id: str,
    record: StepExecutionRecord,
    progress_callback: Optional[Callable[[StepExecutionRecord], None]],
) -> None:
    if progress_callback:
        progress_callback(record)
    try:
        adk_helpers.write_memory_event(
            run_id=run_id,
            event_type="production_agent.step",
            payload=record.as_dict(),
        )
    except adk_helpers.MemoryWriteError:
        pass


def _rate_limit_meta(decision: RateLimitDecision) -> Dict[str, Any]:
    return {
        "status": decision.status,
        "tokens": decision.tokens,
        "retry_after_s": decision.retry_after_s,
        "eta_epoch_s": decision.eta_epoch_s,
        "ttl_deadline_s": decision.ttl_deadline_s,
        "reason": decision.reason,
    }


def _record_stage_event(
    *,
    run_id: str,
    stage: str,
    status: Literal["begin", "success", "fail"],
    attempt: int,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    event = StageEvent(
        run_id=run_id,
        stage=stage,
        status=status,
        timestamp=time.time(),
        attempt=max(1, attempt),
        error=error,
        metadata=metadata or {},
    )
    try:
        payload = event.model_dump() if hasattr(event, "model_dump") else event.dict()
        adk_helpers.write_memory_event(run_id=run_id, event_type="production_agent.stage_event", payload=payload)
    except adk_helpers.MemoryWriteError:
        pass


def _update_stage_manifest(manifest: StageManifest, **updates: Any) -> StageManifest:
    metadata = updates.pop("metadata", None)
    if metadata is not None:
        manifest.metadata = dict(metadata)
    for field, value in updates.items():
        setattr(manifest, field, value)
    return manifest


def _string_to_path(candidate: Optional[str]) -> Optional[Path]:
    if not candidate:
        return None
    if candidate.startswith("artifact://") or candidate.startswith("artifact+fs://"):
        return None
    if candidate.startswith("file://"):
        parsed = urlparse(candidate)
        if parsed.path:
            return Path(unquote(parsed.path))
        return None
    return Path(candidate)


def _manifest_source_path(manifest: StageManifest) -> Optional[Path]:
    for value in (manifest.local_path, manifest.download_url, manifest.artifact_uri):
        path = _string_to_path(value)
        if path and path.exists() and path.is_file():
            return path
    return None


def _route_manifest_via_filesystem(manifest: StageManifest) -> StageManifest:
    if adk_helpers.is_artifact_uri(manifest.artifact_uri):
        if adk_helpers.is_filesystem_artifact_uri(manifest.artifact_uri) and manifest.storage_hint != "filesystem":
            return _update_stage_manifest(manifest, storage_hint="filesystem")
        return manifest

    source_path = _manifest_source_path(manifest)
    if source_path is None:
        return manifest

    metadata = dict(manifest.metadata or {})
    if "stage_manifest_snapshot" not in metadata:
        metadata["stage_manifest_snapshot"] = (
            manifest.model_dump() if hasattr(manifest, "model_dump") else manifest.dict()
        )
    metadata.setdefault("stage", manifest.stage_id)
    metadata.setdefault("artifact_type", manifest.artifact_type)
    metadata.setdefault("run_id", manifest.run_id)
    metadata.setdefault("source_local_path", source_path.as_posix())

    ref = adk_helpers.publish_artifact(
        local_path=source_path,
        artifact_type=manifest.artifact_type,
        metadata=metadata,
        media_type=manifest.media_type or manifest.mime_type,
        run_id=manifest.run_id,
    )

    fs_metadata = dict(ref.get("metadata") or {})
    fs_local_path = fs_metadata.get("local_path")
    fs_manifest_path = fs_metadata.get("manifest_path")
    if fs_local_path:
        metadata.setdefault("filesystem_local_path", fs_local_path)
    if fs_manifest_path:
        metadata.setdefault("filesystem_manifest_path", fs_manifest_path)
    metadata.setdefault("storage_backend", "filesystem")

    download_url: Optional[str]
    if fs_local_path:
        download_url = Path(fs_local_path).resolve().as_uri()
    else:
        download_url = manifest.download_url or ref["uri"]

    updates: Dict[str, Any] = {
        "artifact_uri": ref["uri"],
        "storage_hint": "filesystem",
        "metadata": metadata,
    }
    if fs_local_path:
        updates["local_path"] = fs_local_path
    if download_url:
        updates["download_url"] = download_url
    media_type = ref.get("media_type") or manifest.media_type or manifest.mime_type
    if media_type:
        updates["media_type"] = media_type

    return _update_stage_manifest(manifest, **updates)


def _record_stage_manifest_entries(*, run_id: str, manifests: Sequence[StageManifest]) -> None:
    manifest_list = list(manifests)
    if filesystem_backend_enabled():
        manifest_list = [_route_manifest_via_filesystem(entry) for entry in manifest_list]
    if not manifest_list:
        return
    registry = get_run_registry()
    for manifest in manifest_list:
        payload = manifest.model_dump() if hasattr(manifest, "model_dump") else manifest.dict()
        try:
            adk_helpers.write_memory_event(
                run_id=run_id,
                event_type="production_agent.stage_manifest",
                payload=payload,
            )
        except adk_helpers.MemoryWriteError:
            pass
        entry = ArtifactEntry(
            stage=manifest.stage_id,
            artifact_type=manifest.artifact_type,
            name=manifest.name,
            artifact_uri=manifest.artifact_uri,
            media_type=manifest.media_type,
            local_path=manifest.local_path,
            download_url=manifest.download_url,
            storage_hint=manifest.storage_hint,
            mime_type=manifest.mime_type,
            size_bytes=manifest.size_bytes,
            duration_s=manifest.duration_s,
            frame_rate=manifest.frame_rate,
            resolution_px=manifest.resolution_px,
            checksum_sha256=manifest.checksum_sha256,
            playback_ready=manifest.playback_ready,
            notes=manifest.notes,
            metadata=dict(manifest.metadata),
            created_at=manifest.created_at,
        )
        try:
            registry.record_artifact(run_id, entry)
        except Exception:
            continue


def _stage_event_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, Mapping):
            return {key: _convert(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return {key: _convert(val) for key, val in meta.items() if val is not None}


def _render_frames(
    shot: ShotSpec,
    output_dir: Path,
    base_images: Mapping[str, BaseImageSpec],
    base_image_assets: MutableMapping[str, _BaseImageAsset],
) -> StepResult:
    frames_dir = output_dir / "frames" / shot.id
    frames_dir.mkdir(parents=True, exist_ok=True)

    start_prompt = _base_image_prompt(
        base_images,
        shot.start_base_image_id,
        shot_id=shot.id,
        role="start",
    )
    end_prompt = _base_image_prompt(
        base_images,
        shot.end_base_image_id,
        shot_id=shot.id,
        role="end",
    )

    def _ensure_asset(image_id: str, *, role: str, prompt: str, payload: Dict[str, Any], force_new: bool) -> Path:
        spec = base_images.get(image_id)
        if spec is None:
            raise ProductionAgentError(f"Shot {shot.id} references missing base image '{image_id}'")
        if not force_new:
            asset = base_image_assets.get(image_id)
            if asset and asset.path and asset.path.exists():
                return asset.path
        target = frames_dir / f"{image_id}-{role}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        base_image_assets[image_id] = _BaseImageAsset(
            spec=spec,
            path=target,
            payload_bytes=target.read_bytes(),
        )
        return target

    start_payload = {
        "shot_id": shot.id,
        "role": "start",
        "base_image_id": shot.start_base_image_id,
        "prompt": start_prompt,
    }
    end_payload = {
        "shot_id": shot.id,
        "role": "end",
        "base_image_id": shot.end_base_image_id,
        "prompt": end_prompt,
    }
    start_frame_path = _ensure_asset(
        shot.start_base_image_id,
        role="start",
        prompt=start_prompt,
        payload=start_payload,
        force_new=False,
    )
    end_frame_path = _ensure_asset(
        shot.end_base_image_id,
        role="end",
        prompt=end_prompt,
        payload=end_payload,
        force_new=True,
    )

    summary_payload = {
        "shot_id": shot.id,
        "visual_description": shot.visual_description,
        "start_base_image_id": shot.start_base_image_id,
        "end_base_image_id": shot.end_base_image_id,
        "start_frame_prompt": start_prompt,
        "end_frame_prompt": end_prompt,
        "start_frame_path": start_frame_path.as_posix(),
        "end_frame_path": end_frame_path.as_posix(),
    }
    summary_path = frames_dir / "frames.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return StepResult(
        path=summary_path,
        meta={
            "shot_id": shot.id,
            "start_frame_path": start_frame_path.as_posix(),
            "end_frame_path": end_frame_path.as_posix(),
        },
    )


def _synthesize_dialogue(
    shot: ShotSpec,
    output_dir: Path,
    *,
    plan_id: str,
    run_id: str,
    voice_profiles: Mapping[str, Mapping[str, Any]],
) -> StepResult:
    valid_lines: List[tuple[int, DialogueLine]] = [
        (idx, line)
        for idx, line in enumerate(shot.dialogue)
        if line.text and line.text.strip()
    ]
    if not valid_lines:
        raise ProductionAgentError(f"Shot {shot.id} dialogue is empty")

    audio_dir = output_dir / "audio" / shot.id
    audio_dir.mkdir(parents=True, exist_ok=True)
    default_voice_config = _voice_config_for_shot(shot, voice_profiles)

    line_entries: List[Dict[str, Any]] = []
    line_paths: List[Path] = []
    total_duration = 0.0
    primary_metadata: Optional[Dict[str, Any]] = None
    primary_artifact_uri: Optional[str] = None
    primary_voice_metadata: Optional[Mapping[str, Any]] = None

    for idx, line in valid_lines:
        text = line.text.strip()
        voice_config = _voice_config_for_character(line.character_id, voice_profiles, default_voice_config)
        step_label = f"{shot.id}:tts:{idx:02d}"
        artifact = tts_stage.synthesize(
            text,
            voice_config=voice_config,
            plan_id=plan_id,
            step_id=step_label,
            run_id=run_id,
            output_dir=audio_dir,
        )
        metadata = dict(artifact.get("metadata") or {})
        source_path_value = metadata.get("source_path")
        local_path = Path(source_path_value) if source_path_value else audio_dir / f"{shot.id}-{idx:02d}.wav"
        line_paths.append(local_path)
        duration = metadata.get("duration_s")
        if isinstance(duration, (int, float)):
            total_duration += float(duration)

        entry: Dict[str, Any] = {
            "line_index": idx,
            "character_id": line.character_id,
            "path": str(local_path),
            "artifact_uri": artifact.get("uri"),
            "provider_id": metadata.get("provider_id"),
            "voice_id": metadata.get("voice_id"),
            "duration_s": metadata.get("duration_s"),
            "sample_rate": metadata.get("sample_rate"),
            "bit_depth": metadata.get("bit_depth"),
            "watermarked": metadata.get("watermarked"),
        }
        adapter_meta = metadata.get("adapter_metadata")
        if isinstance(adapter_meta, Mapping):
            entry["adapter_metadata"] = dict(adapter_meta)
        voice_meta = metadata.get("voice_metadata")
        if isinstance(voice_meta, Mapping):
            entry["voice_metadata"] = dict(voice_meta)
        line_entries.append(entry)

        if primary_metadata is None:
            primary_metadata = metadata
            primary_artifact_uri = artifact.get("uri")
            if isinstance(voice_meta, Mapping):
                primary_voice_metadata = dict(voice_meta)

    score_breakdown = (primary_metadata or {}).get("score_breakdown")
    score_meta = dict(score_breakdown) if isinstance(score_breakdown, Mapping) else {}
    tts_meta: Dict[str, Any] = {
        "line_artifacts": line_entries,
        "lines_synthesized": len(line_entries),
        "dialogue_paths": [str(path) for path in line_paths],
        "total_duration_s": round(total_duration, 3),
        "provider_id": (primary_metadata or {}).get("provider_id"),
        "voice_id": (primary_metadata or {}).get("voice_id"),
        "score_breakdown": score_meta,
    }
    if isinstance(primary_voice_metadata, Mapping):
        tts_meta["voice_metadata"] = dict(primary_voice_metadata)
    summary_payload = {
        "shot_id": shot.id,
        "lines": line_entries,
        "dialogue_paths": [str(path) for path in line_paths],
        "total_duration_s": total_duration,
        "provider_id": tts_meta.get("provider_id"),
        "voice_id": tts_meta.get("voice_id"),
        "voice_metadata": tts_meta.get("voice_metadata"),
    }
    summary_path = audio_dir / "tts_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return StepResult(
        path=summary_path,
        paths=tuple(line_paths),
        artifact_uri=primary_artifact_uri,
        model_id=(primary_metadata or {}).get("model_id"),
        meta={
            "tts": tts_meta,
            "dialogue_paths": [str(path) for path in line_paths],
            "lines": len(line_entries),
            "summary_path": summary_path.as_posix(),
        },
    )


def _render_video_clip(
    shot: ShotSpec,
    output_dir: Path,
    plan_id: str,
    run_id: str,
    base_images: Mapping[str, BaseImageSpec],
    base_image_assets: Mapping[str, _BaseImageAsset],
    progress_callback: Optional[Callable[[StepExecutionRecord], None]] = None,
) -> Path:
    video_dir = output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"{shot.id}.mp4"

    fps = _default_video_fps()
    num_frames = _estimate_frame_count(shot.duration_sec, fps)
    start_prompt = _base_image_prompt(base_images, shot.start_base_image_id, shot_id=shot.id, role="start")
    end_prompt = _base_image_prompt(base_images, shot.end_base_image_id, shot_id=shot.id, role="end")
    prompt_parts = [shot.visual_description or "", shot.motion_prompt or "", start_prompt, end_prompt]
    prompt = " | ".join(part for part in prompt_parts if part) or f"Shot {shot.id}"

    def _frame_payload(image_id: str, prompt_text: str) -> bytes:
        asset = base_image_assets.get(image_id)
        if asset and asset.payload_bytes:
            return asset.payload_bytes
        return _encode_prompt_bytes(prompt_text)

    start_frames = [_frame_payload(shot.start_base_image_id, start_prompt)] if start_prompt else []
    end_frames = [_frame_payload(shot.end_base_image_id, end_prompt)] if end_prompt else []

    opts = {
        "num_frames": num_frames,
        "plan_id": plan_id,
        "step_id": f"{shot.id}:video",
        "run_id": run_id,
        "output_path": str(dest),
        "output_dir": str(video_dir),
        "shot_id": shot.id,
    }

    on_progress = None
    if progress_callback:
        on_progress = _video_progress_forwarder(
            plan_id=plan_id,
            step_id=f"{shot.id}:video",
            progress_callback=progress_callback,
        )

    artifact = videos_stage.render_video(start_frames, end_frames, prompt, opts, on_progress=on_progress)
    metadata = artifact.get("metadata") or {}
    source_path = metadata.get("source_path")
    local_path = Path(source_path) if source_path else dest
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path is None:
            local_path.write_bytes(json.dumps({"shot_id": shot.id, "plan_id": plan_id}).encode("utf-8"))
    return local_path


def _lipsync_clip(shot: ShotSpec, output_dir: Path) -> Path:
    dest = output_dir / "lipsync" / f"{shot.id}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"LIPSYNC")
    return dest


def _assemble_plan(plan: MoviePlan, shots: Sequence[_ShotArtifacts], output_dir: Path) -> Path:
    dest = output_dir / "final" / f"{_plan_identifier(plan)}-assembly.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),  # type: ignore[attr-defined]
        "shots": [
            {
                "shot_id": art.shot_id,
                "frames": str(art.frames_path) if art.frames_path else None,
                "dialogue": [str(path) for path in art.dialogue_paths] if art.dialogue_paths else None,
                "video": str(art.video_path) if art.video_path else None,
                "lipsync": str(art.lipsync_path) if art.lipsync_path else None,
            }
            for art in shots
        ],
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def _run_final_delivery_stage(
    *,
    plan: MoviePlan,
    plan_id: str,
    run_id: str,
    output_dir: Path,
    shot_artifacts: Sequence[_ShotArtifacts],
    dialogue_result: Optional[_DialogueStageResult],
    assemble_path: Path,
) -> tuple[adk_helpers.ArtifactRef, StepResult]:
    final_video_path = _prepare_final_video(
        plan_id=plan_id,
        output_dir=output_dir,
        shot_artifacts=shot_artifacts,
        assemble_path=assemble_path,
    )
    timeline_audio_path = dialogue_result.timeline_audio_path if dialogue_result else None
    total_duration = dialogue_result.total_duration_s if dialogue_result else sum(shot.duration_sec for shot in plan.shots)
    frame_rate = float(plan.render_profile.video.max_fps or _default_video_fps())
    resolution = _resolve_render_resolution(plan)
    checksum = _hash_file_path(final_video_path)
    size_bytes = final_video_path.stat().st_size if final_video_path.exists() else None
    video_metadata = {
        "plan_id": plan_id,
        "run_id": run_id,
        "duration_s": total_duration,
        "frame_rate": frame_rate,
        "resolution_px": resolution,
        "checksum_sha256": checksum,
        "stage": "finalize",
    }
    video_artifact_ref = adk_helpers.publish_artifact(
        local_path=final_video_path,
        artifact_type="video_final",
        media_type="video/mp4",
        metadata=video_metadata,
        run_id=run_id,
    )
    storage_hint = "adk" if video_artifact_ref.get("storage") == "adk" else "local"
    artifact_meta = video_artifact_ref.get("metadata") or {}
    download_url = artifact_meta.get("download_url") if storage_hint == "adk" else None
    if storage_hint == "adk" and not download_url:
        download_url = video_artifact_ref.get("uri")
    step_result = StepResult(
        path=final_video_path,
        paths=(final_video_path,),
        artifact_uri=str(video_artifact_ref["uri"]),
        meta={
            "stage": "finalize",
            "artifact_type": "video_final",
            "media_type": "video/mp4",
            "local_path": final_video_path.as_posix(),
            "download_url": download_url,
            "storage_hint": storage_hint,
            "mime_type": "video/mp4",
            "size_bytes": size_bytes,
            "duration_s": total_duration,
            "frame_rate": frame_rate,
            "resolution_px": resolution,
            "checksum_sha256": checksum,
            "playback_ready": final_video_path.exists(),
            "timeline_audio": timeline_audio_path.as_posix() if timeline_audio_path else None,
            "assemble_path": assemble_path.as_posix(),
        },
    )
    return video_artifact_ref, step_result


def _prepare_final_video(
    *,
    plan_id: str,
    output_dir: Path,
    shot_artifacts: Sequence[_ShotArtifacts],
    assemble_path: Path,
) -> Path:
    dest = output_dir / "final" / f"{plan_id}-video_final.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    candidates: List[Optional[Path]] = []
    if assemble_path.suffix.lower() in {".mp4", ".mov", ".mkv"}:
        candidates.append(assemble_path)
    for art in shot_artifacts:
        if art.lipsync_path:
            candidates.append(art.lipsync_path)
        if art.video_path:
            candidates.append(art.video_path)
    for candidate in candidates:
        if candidate and candidate.exists():
            shutil.copyfile(candidate, dest)
            return dest
    placeholder = {
        "plan_id": plan_id,
        "generated_at": _now_iso(),
        "note": "finalize placeholder",
    }
    dest.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest

def _resolve_render_resolution(plan: MoviePlan) -> str:
    candidates: List[Any] = []
    if plan.render_profile.metadata:
        candidates.extend(plan.render_profile.metadata.get(key) for key in ("resolution", "resolution_px"))
    if plan.metadata:
        candidates.extend(plan.metadata.get(key) for key in ("resolution", "resolution_px"))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return f"{value[0]}x{value[1]}"
    return "1280x720"


def _record_summary_event(
    run_id: str,
    plan_id: str,
    mode: str,
    records: Sequence[StepExecutionRecord],
    *,
    simulation: Optional[SimulationReport] = None,
) -> None:
    payload: Dict[str, Any] = {
        "plan_id": plan_id,
        "mode": mode,
        "step_count": len(records),
    }
    if simulation is not None:
        payload["simulation"] = {
            "steps": len(simulation.steps),
            "resource_summary": simulation.resource_summary,
        }
    else:
        payload["artifact_uris"] = [rec.artifact_uri for rec in records if rec.artifact_uri]
    try:
        adk_helpers.write_memory_event(
            run_id=run_id,
            event_type="production_agent.summary",
            payload=payload,
        )
    except adk_helpers.MemoryWriteError:
        pass


def _flag_enabled(flag: str) -> bool:
    value = os.environ.get(flag)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false"}


def _now_iso() -> str:
    return _to_iso(datetime.now(timezone.utc))


def _to_iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _default_video_fps() -> int:
    try:
        value = int(os.environ.get("VIDEOS_STAGE_DEFAULT_FPS", "16"))
    except ValueError:
        value = 16
    return max(1, value)


def _estimate_frame_count(duration_sec: float, fps: int) -> int:
    frames = int(round(duration_sec * fps))
    if frames <= 0:
        frames = fps
    return max(1, frames)


def _encode_prompt_bytes(value: str) -> bytes:
    return value.encode("utf-8")


def _video_progress_forwarder(
    *,
    plan_id: str,
    step_id: str,
    progress_callback: Callable[[StepExecutionRecord], None],
) -> Callable[[videos_stage.CallbackEvent], None]:
    def _forward(event: videos_stage.CallbackEvent) -> None:
        record = StepExecutionRecord(
            plan_id=plan_id,
            step_id=step_id,
            step_type="video",
            status="running",
            start_time=_now_iso(),
            end_time=_now_iso(),
            duration_s=0.0,
            attempts=0,
            meta={
                "videos_stage_progress": dict(event),
            },
        )
        progress_callback(record)

    return _forward


__all__ = [
    "execute_plan",
    "ProductionAgentError",
    "PlanPolicyViolation",
    "StepExecutionError",
    "StepRateLimitError",
    "StepQueuedError",
    "StepRateLimitExceededError",
    "StepExecutionRecord",
    "SimulationReport",
    "ProductionResult",
    "ProductionAgentConfig",
]
