from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import time
import wave
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Sequence, Union

from pydantic import ValidationError

from . import adk_helpers, observability, telemetry, videos_agent, tts_agent, schema_registry
from .run_registry import ArtifactEntry, get_run_registry
from .images_agent import RateLimitExceeded, RateLimitQueued
from .ratelimit import RateLimitDecision
from .schemas import BaseImageSpec, DialogueLine, MoviePlan, ShotSpec, RunContext, StageEvent, StageManifest


class ProductionAgentError(RuntimeError):
    """Base error for production agent failures."""


class PlanPolicyViolation(ProductionAgentError):
    """Raised when a plan violates gating or safety policies."""


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
    policy_decisions: Sequence[str]


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
    policy_decisions: List[str]
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


@dataclass(frozen=True)
class _TimelineSegment:
    entry_index: int
    kind: Literal["dialogue", "silence"]
    target_duration: float
    path: Optional[Path]


def execute_plan(
    plan: MoviePlan | Mapping[str, Any],
    *,
    mode: Literal["dry", "run"] = "dry",
    progress_callback: Optional[Callable[[StepExecutionRecord], None]] = None,
    config: Optional[ProductionAgentConfig] = None,
    run_id: Optional[str] = None,
    pre_step_hook: Optional[Callable[[str], None]] = None,
) -> ProductionResult:
    """Execute or simulate a MoviePlan."""

    cfg = config or ProductionAgentConfig()
    model = _coerce_plan(plan)
    _validate_plan(model)
    plan_id = _plan_identifier(model)
    run_id = run_id or observability.get_session_id()

    if mode == "dry":
        plan_intake_result = _run_plan_intake(model, plan_id=plan_id, run_id=run_id, output_dir=None)
        report = _simulate_execution_report(model, plan_intake_result.policy_decisions)
        _record_summary_event(run_id, plan_id, "dry", [], simulation=report)
        return ProductionResult([], steps=[], simulation_report=report)

    output_dir = _resolve_output_dir(run_id, plan_id)
    records: List[StepExecutionRecord] = []
    shot_artifacts: List[_ShotArtifacts] = []

    plan_intake_holder: Dict[str, _PlanIntakeResult] = {}

    def _plan_intake_action() -> StepResult:
        result = _run_plan_intake(model, plan_id=plan_id, run_id=run_id, output_dir=output_dir)
        plan_intake_holder["result"] = result
        if result.run_context_path is None:
            raise ProductionAgentError("plan intake failed to persist run_context")
        meta: Dict[str, Any] = {
            "artifact_type": "plan_run_context",
            "media_type": "application/json",
            "plan_uri": str(result.plan_path) if result.plan_path else None,
            "dialogue_timeline_uri": result.run_context.dialogue_timeline_uri,
            "base_image_map": dict(result.run_context.base_image_map),
            "schema_uris": result.schema_meta,
            "policy_decisions": list(result.policy_decisions),
        }
        # Remove None to keep metadata JSON-friendly
        meta = {key: value for key, value in meta.items() if value is not None}
        _record_stage_manifest_entries(run_id=run_id, manifests=result.stage_manifests)
        meta.setdefault("stage_manifest_count", len(result.stage_manifests))
        return StepResult(
            path=result.run_context_path,
            artifact_uri=str(result.run_context_path),
            meta=meta,
        )

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
    records.append(plan_record)

    plan_intake_result = plan_intake_holder.get("result")
    if plan_intake_result is None:
        raise ProductionAgentError("plan intake stage did not return a result")

    base_images = plan_intake_result.base_image_lookup
    base_image_assets = plan_intake_result.base_image_assets
    voice_profiles = _character_voice_map(model)

    dialogue_stage_holder: Dict[str, _DialogueStageResult] = {}

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
            )
            shot_artifacts.append(artifacts)
    except StepRateLimitError:
        _record_summary_event(run_id, plan_id, "run", records)
        raise

    try:
        final_record, final_result = _run_step(
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
        records.append(final_record)
    except StepRateLimitError as exc:
        records.append(exc.record)
        _record_summary_event(run_id, plan_id, "run", records)
        raise

    artifact_ref = adk_helpers.publish_artifact(
        local_path=final_result.path,
        artifact_type=cfg.artifact_type,
        metadata={"plan_id": plan_id, "shot_count": len(model.shots)},
    )

    qa_publish_holder: Dict[str, adk_helpers.ArtifactRef] = {}

    def _qa_publish_action() -> StepResult:
        dialogue_result = dialogue_stage_holder.get("result")
        manifests, qa_artifact_ref, step_result = _run_qa_publish_stage(
            plan=model,
            plan_id=plan_id,
            run_id=run_id,
            output_dir=output_dir,
            shot_artifacts=shot_artifacts,
            dialogue_result=dialogue_result,
            assemble_path=final_result.path,
        )
        qa_publish_holder["artifact"] = qa_artifact_ref
        _record_stage_manifest_entries(run_id=run_id, manifests=manifests)
        return step_result

    try:
        qa_record, _ = _run_step(
            plan_id=plan_id,
            run_id=run_id,
            step_id=f"{plan_id}:qa_publish",
            step_type="qa_publish",
            gate_flag=None,
            cfg=cfg,
            progress_callback=progress_callback,
            pre_step_hook=pre_step_hook,
            action=_qa_publish_action,
            meta={"stage": "qa_publish", "shot_count": len(model.shots)},
        )
        records.append(qa_record)
    except StepRateLimitError as exc:
        records.append(exc.record)
        _record_summary_event(run_id, plan_id, "run", records)
        raise

    telemetry.emit_event(
        "production_agent.execute_plan.completed",
        {
            "plan_id": plan_id,
            "artifact_uri": artifact_ref["uri"],
            "shot_count": len(model.shots),
        },
    )
    _record_summary_event(run_id, plan_id, "run", records)
    artifacts: List[adk_helpers.ArtifactRef] = [artifact_ref]
    qa_artifact = qa_publish_holder.get("artifact")
    if qa_artifact:
        artifacts.append(qa_artifact)
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


def _run_policy_checks(plan: MoviePlan, base_images: Mapping[str, BaseImageSpec]) -> List[str]:
    banned_keywords = {"weaponized", "forbidden"}
    decisions: List[str] = []
    for shot in plan.shots:
        start_prompt = _base_image_prompt(
            base_images,
            shot.start_base_image_id,
            shot_id=shot.id,
            role="start",
            allow_empty=True,
        )
        end_prompt = _base_image_prompt(
            base_images,
            shot.end_base_image_id,
            shot_id=shot.id,
            role="end",
            allow_empty=True,
        )
        text = " ".join(
            filter(
                None,
                [shot.visual_description, start_prompt, end_prompt],
            )
        ).lower()
        if any(keyword in text for keyword in banned_keywords):
            raise PlanPolicyViolation(f"Shot {shot.id} violates content policy")
        if shot.duration_sec > 120:
            decisions.append(f"shot:{shot.id} exceeds duration target")
    return decisions


def _run_plan_intake(
    plan: MoviePlan,
    *,
    plan_id: str,
    run_id: str,
    output_dir: Optional[Path],
) -> _PlanIntakeResult:
    base_image_lookup = _build_base_image_lookup(plan)
    policy_decisions = _run_policy_checks(plan, base_image_lookup)

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
        policy_decisions=policy_decisions,
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
        policy_decisions=policy_decisions,
        base_image_map=base_image_map,
    )

    return _PlanIntakeResult(
        base_image_lookup=base_image_lookup,
        base_image_assets=base_image_assets,
        policy_decisions=list(policy_decisions),
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

    timeline_dir = output_dir / "audio" / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    timeline_audio_path = timeline_dir / "tts_timeline.wav"
    summary_path = timeline_dir / "dialogue_timeline_audio.json"

    line_entries: List[Dict[str, Any]] = []
    line_paths: List[Path] = []
    segments: List[_TimelineSegment] = []

    def _positive_float(value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        return 0.0

    for index, entry in enumerate(plan.dialogue_timeline):
        entry_type = getattr(entry, "type", "dialogue")
        start_time = _positive_float(getattr(entry, "start_time_sec", 0.0))
        duration = _positive_float(getattr(entry, "duration_sec", 0.0))
        base_payload: Dict[str, Any] = {
            "index": index,
            "type": entry_type,
            "start_time_sec": start_time,
            "duration_sec": duration,
            "character_id": getattr(entry, "character_id", None),
        }
        if entry_type == "silence":
            segments.append(
                _TimelineSegment(
                    entry_index=index,
                    kind="silence",
                    target_duration=duration,
                    path=None,
                )
            )
            line_entries.append({**base_payload, "text": None, "artifact_uri": None, "local_path": None})
            continue

        text = getattr(entry, "text", "")
        if not text or not text.strip():
            raise ProductionAgentError(f"Dialogue timeline entry {index} must include text")
        voice_config = _voice_config_for_character(getattr(entry, "character_id", None), voice_profiles, None)
        step_label = f"dialogue_timeline:{index:04d}"
        artifact = tts_agent.synthesize(
            text,
            voice_config=voice_config,
            plan_id=plan_id,
            step_id=step_label,
            run_id=run_id,
            output_dir=timeline_dir,
        )
        metadata = dict(artifact.get("metadata") or {})
        source_path = metadata.get("source_path")
        local_path = Path(source_path) if source_path else timeline_dir / f"timeline_{index:04d}.wav"
        if not local_path.exists():
            local_path.write_bytes(b"")
        line_paths.append(local_path)
        raw_duration_hint = _positive_float(metadata.get("duration_s"))
        target_duration = duration if duration > 0 else raw_duration_hint
        segments.append(
            _TimelineSegment(
                entry_index=index,
                kind="dialogue",
                target_duration=target_duration,
                path=local_path,
            )
        )
        entry_payload: Dict[str, Any] = {
            **base_payload,
            "text": text,
            "artifact_uri": artifact.get("uri"),
            "local_path": local_path.as_posix(),
            "voice_id": metadata.get("voice_id"),
            "provider_id": metadata.get("provider_id"),
            "duration_audio_s": metadata.get("duration_s"),
            "sample_rate": metadata.get("sample_rate"),
            "bit_depth": metadata.get("bit_depth"),
            "watermarked": metadata.get("watermarked"),
        }
        voice_meta = metadata.get("voice_metadata")
        if isinstance(voice_meta, Mapping):
            entry_payload["voice_metadata"] = dict(voice_meta)
        adapter_meta = metadata.get("adapter_metadata")
        if isinstance(adapter_meta, Mapping):
            entry_payload["adapter_metadata"] = dict(adapter_meta)
        line_entries.append(entry_payload)

    if not segments:
        raise ProductionAgentError("dialogue_timeline must contain at least one entry to synthesize")

    total_duration, sample_rate, sample_width, channels, offsets = _stitch_timeline_audio(segments, timeline_audio_path)

    for entry in line_entries:
        offset_meta = offsets.get(entry["index"])
        if offset_meta:
            entry["start_time_actual_s"] = offset_meta["start_time_s"]
            entry["end_time_actual_s"] = offset_meta["end_time_s"]
            entry["duration_actual_s"] = offset_meta["written_duration_s"]
            entry["duration_audio_raw_s"] = offset_meta.get("source_duration_s")
            entry["timeline_padding_s"] = offset_meta.get("padding_applied_s")
            entry["timeline_trimmed_s"] = offset_meta.get("trimmed_s")
        else:
            entry["start_time_actual_s"] = entry.get("start_time_sec", 0.0)
            planned_duration = entry.get("duration_sec") or 0.0
            entry["end_time_actual_s"] = entry["start_time_actual_s"] + planned_duration
            entry["duration_actual_s"] = planned_duration
            entry["duration_audio_raw_s"] = None
            entry["timeline_padding_s"] = 0.0
            entry["timeline_trimmed_s"] = 0.0

    summary_payload = {
        "plan_id": plan_id,
        "run_id": run_id,
        "entry_count": len(line_entries),
        "lines": line_entries,
        "timeline_audio": {
            "path": timeline_audio_path.as_posix(),
            "duration_s": total_duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
        },
        "timeline_offsets": offsets,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

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


def _stitch_timeline_audio(
    segments: Sequence[_TimelineSegment],
    timeline_path: Path,
) -> tuple[float, int, int, int, Dict[int, Dict[str, Any]]]:
    sample_rate: Optional[int] = None
    sample_width: Optional[int] = None
    channels: Optional[int] = None
    total_duration = 0.0
    offsets: Dict[int, Dict[str, float]] = {}
    timeline_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_writer_defaults(writer: wave.Wave_write) -> None:
        nonlocal sample_rate, sample_width, channels
        if sample_rate is None or sample_width is None or channels is None:
            sample_rate, sample_width, channels = 22050, 2, 1
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)

    with wave.open(str(timeline_path), "wb") as writer:
        for segment in segments:
            start_time = total_duration
            if segment.kind == "dialogue":
                if segment.path is None or not segment.path.exists():
                    raise ProductionAgentError("dialogue segment missing audio payload")
                with wave.open(str(segment.path), "rb") as reader:
                    sr = reader.getframerate()
                    sw = reader.getsampwidth()
                    ch = reader.getnchannels()
                    frame_count = reader.getnframes()
                    data = reader.readframes(frame_count)
                if sample_rate is None:
                    sample_rate, sample_width, channels = sr, sw, ch
                    writer.setnchannels(channels)
                    writer.setsampwidth(sample_width)
                    writer.setframerate(sample_rate)
                elif sr != sample_rate or sw != sample_width or ch != channels:
                    raise ProductionAgentError("dialogue audio segments must share sample parameters")
                if sample_rate is None or sample_width is None or channels is None:
                    raise ProductionAgentError("audio parameters unavailable for dialogue segment")
                bytes_per_frame = sample_width * channels
                actual_duration = frame_count / sample_rate if sample_rate else 0.0
                target_duration = segment.target_duration if segment.target_duration > 0 else actual_duration
                if target_duration <= 0:
                    target_duration = actual_duration or 1.0 / sample_rate
                desired_frames = max(1, int(round(target_duration * sample_rate)))
                if desired_frames <= frame_count:
                    writer.writeframes(data[: desired_frames * bytes_per_frame])
                else:
                    writer.writeframes(data)
                    missing_frames = desired_frames - frame_count
                    writer.writeframes(b"\x00" * bytes_per_frame * missing_frames)
                written_duration = desired_frames / sample_rate
                padding = max(0.0, written_duration - actual_duration)
                trimmed = max(0.0, actual_duration - written_duration)
                offsets[segment.entry_index] = {
                    "kind": segment.kind,
                    "start_time_s": start_time,
                    "end_time_s": start_time + written_duration,
                    "written_duration_s": written_duration,
                    "target_duration_s": target_duration,
                    "source_duration_s": actual_duration,
                    "padding_applied_s": padding,
                    "trimmed_s": trimmed,
                }
                total_duration += written_duration
            else:
                _ensure_writer_defaults(writer)
                if sample_rate is None or sample_width is None or channels is None:
                    raise ProductionAgentError("audio parameters unavailable for silence segment")
                bytes_per_frame = sample_width * channels
                target_duration = max(0.0, segment.target_duration)
                desired_frames = int(round(target_duration * sample_rate))
                if desired_frames > 0:
                    silence_frame = b"\x00" * bytes_per_frame
                    writer.writeframes(silence_frame * desired_frames)
                written_duration = desired_frames / sample_rate if sample_rate and desired_frames > 0 else 0.0
                offsets[segment.entry_index] = {
                    "kind": "silence",
                    "start_time_s": start_time,
                    "end_time_s": start_time + written_duration,
                    "written_duration_s": written_duration,
                    "target_duration_s": target_duration,
                    "source_duration_s": 0.0,
                    "padding_applied_s": 0.0,
                    "trimmed_s": 0.0,
                }
                total_duration += written_duration

    if sample_rate is None or sample_width is None or channels is None:
        raise ProductionAgentError("Failed to synthesize dialogue timeline audio")
    return total_duration, sample_rate, sample_width, channels, offsets


def _build_plan_intake_manifests(
    *,
    plan: MoviePlan,
    plan_id: str,
    run_id: str,
    plan_path: Optional[Path],
    dialogue_timeline_path: Optional[Path],
    run_context_path: Optional[Path],
    schema_meta: Mapping[str, Dict[str, str]],
    policy_decisions: Sequence[str],
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
            "policy_decisions": list(policy_decisions),
            "base_image_map": dict(base_image_map),
            "dialogue_timeline_uri": dialogue_timeline_path.as_posix() if dialogue_timeline_path else None,
        },
    )

    return manifests


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


def _simulate_execution_report(plan: MoviePlan, policy_decisions: Sequence[str]) -> SimulationReport:
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
    return SimulationReport(plan_id=_plan_identifier(plan), steps=steps, resource_summary=summary, policy_decisions=list(policy_decisions))


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
    base_image_assets: Mapping[str, _BaseImageAsset],
) -> _ShotArtifacts:
    artifacts = _ShotArtifacts(shot_id=shot.id)

    def _append_record(record: StepExecutionRecord) -> None:
        records.append(record)

    def _run_with_tracking(**kwargs: Any) -> StepResult:
        try:
            record, step_result = _run_step(**kwargs)
        except StepRateLimitError as exc:
            _append_record(exc.record)
            raise
        _append_record(record)
        return step_result

    frames_result = _run_with_tracking(
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
    artifacts.frames_path = frames_result.path

    if shot.dialogue:
        tts_result = _run_with_tracking(
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

    video_result = _run_with_tracking(
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

    if shot.is_talking_closeup and artifacts.dialogue_paths and artifacts.video_path:
        lipsync_result = _run_with_tracking(
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

    should_execute = gate_flag is None or _flag_enabled(gate_flag)
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
        status = "simulated"
        _record_stage_event(
            run_id=run_id,
            stage=step_type,
            status="success",
            attempt=1,
            metadata=_stage_event_metadata({**base_event_info, "simulated": True}),
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


def _record_stage_manifest_entries(*, run_id: str, manifests: Sequence[StageManifest]) -> None:
    if not manifests:
        return
    registry = get_run_registry()
    for manifest in manifests:
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
            qa_report_uri=manifest.qa_report_uri,
            qa_passed=manifest.qa_passed,
            qa_mode=manifest.qa_mode,
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
        artifact = tts_agent.synthesize(
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
    return StepResult(
        path=audio_dir,
        paths=tuple(line_paths),
        artifact_uri=primary_artifact_uri,
        model_id=(primary_metadata or {}).get("model_id"),
        meta={
            "tts": tts_meta,
            "dialogue_paths": [str(path) for path in line_paths],
            "lines": len(line_entries),
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

    artifact = videos_agent.render_video(start_frames, end_frames, prompt, opts, on_progress=on_progress)
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


def _run_qa_publish_stage(
    *,
    plan: MoviePlan,
    plan_id: str,
    run_id: str,
    output_dir: Path,
    shot_artifacts: Sequence[_ShotArtifacts],
    dialogue_result: Optional[_DialogueStageResult],
    assemble_path: Path,
) -> tuple[List[StageManifest], adk_helpers.ArtifactRef, StepResult]:
    stage_manifest_schema_meta = _schema_metadata(schema_registry.stage_manifest_schema())
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
    qa_mode = _resolve_qa_mode(plan)
    qa_skipped = qa_mode == "skip"
    issues = _collect_qa_publish_issues(
        plan=plan,
        shot_artifacts=shot_artifacts,
        final_video_path=final_video_path,
        timeline_audio_path=timeline_audio_path,
    )
    qa_passed = not issues and not qa_skipped
    qa_report_path, _ = _write_qa_publish_report(
        plan=plan,
        plan_id=plan_id,
        run_id=run_id,
        output_dir=output_dir,
        issues=issues,
        qa_mode=qa_mode,
        qa_passed=qa_passed,
        qa_skipped=qa_skipped,
        duration_s=total_duration,
        frame_rate=frame_rate,
        resolution=resolution,
    )
    qa_report_ref = adk_helpers.publish_artifact(
        local_path=qa_report_path,
        artifact_type="qa_publish_report",
        media_type="application/json",
        metadata={
            "plan_id": plan_id,
            "run_id": run_id,
            "qa_mode": qa_mode,
            "qa_passed": qa_passed,
            "issue_count": len(issues),
        },
        run_id=run_id,
    )
    checksum = _hash_file_path(final_video_path)
    size_bytes = final_video_path.stat().st_size if final_video_path.exists() else None
    video_metadata = {
        "plan_id": plan_id,
        "run_id": run_id,
        "qa_mode": qa_mode,
        "qa_passed": qa_passed,
        "qa_skipped": qa_skipped,
        "issue_count": len(issues),
        "issues": issues,
        "duration_s": total_duration,
        "frame_rate": frame_rate,
        "resolution_px": resolution,
        "checksum_sha256": checksum,
        "stage": "qa_publish",
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
    manifest_metadata = _stage_event_metadata(
        {
            "plan_id": plan_id,
            "stage": "qa_publish",
            "qa_mode": qa_mode,
            "qa_skipped": qa_skipped,
            "issue_count": len(issues),
            "issues": issues,
            "stage_manifest_schema": stage_manifest_schema_meta,
            "qa_report_uri": qa_report_ref.get("uri"),
            "timeline_audio": timeline_audio_path.as_posix() if timeline_audio_path else None,
            "assemble_path": assemble_path.as_posix(),
        }
    )
    manifest = StageManifest(
        run_id=run_id,
        stage_id="qa_publish",
        artifact_type="video_final",
        name=final_video_path.name,
        artifact_uri=str(video_artifact_ref["uri"]),
        media_type="video/mp4",
        local_path=final_video_path.as_posix(),
        download_url=download_url,
        storage_hint=storage_hint,
        mime_type="video/mp4",
        size_bytes=size_bytes,
        duration_s=total_duration,
        frame_rate=frame_rate,
        resolution_px=resolution,
        checksum_sha256=checksum,
        qa_report_uri=qa_report_ref.get("uri"),
        qa_passed=qa_passed,
        qa_mode=qa_mode,
        playback_ready=final_video_path.exists(),
        notes="qa_skipped" if qa_skipped else None,
        metadata=manifest_metadata,
    )
    step_result = StepResult(
        path=final_video_path,
        paths=(final_video_path,),
        artifact_uri=str(video_artifact_ref["uri"]),
        meta={
            "stage": "qa_publish",
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
            "qa_report_uri": qa_report_ref.get("uri"),
            "qa_report_local_path": qa_report_path.as_posix(),
            "qa_passed": qa_passed,
            "qa_mode": qa_mode,
            "qa_skipped": qa_skipped,
            "issue_count": len(issues),
            "issues": issues,
            "playback_ready": final_video_path.exists(),
        },
    )
    return [manifest], video_artifact_ref, step_result


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
        "note": "qa_publish placeholder",
    }
    dest.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def _collect_qa_publish_issues(
    *,
    plan: MoviePlan,
    shot_artifacts: Sequence[_ShotArtifacts],
    final_video_path: Path,
    timeline_audio_path: Optional[Path],
) -> List[str]:
    issues: List[str] = []
    artifacts_by_shot = {art.shot_id: art for art in shot_artifacts}
    if not final_video_path.exists():
        issues.append("final_video_missing")
    if timeline_audio_path is not None and not timeline_audio_path.exists():
        issues.append("timeline_audio_missing")
    for shot in plan.shots:
        art = artifacts_by_shot.get(shot.id)
        if art is None:
            issues.append(f"missing_artifacts:{shot.id}")
            continue
        if art.video_path is None or not art.video_path.exists():
            issues.append(f"video_missing:{shot.id}")
        if shot.dialogue and not art.dialogue_paths:
            issues.append(f"dialogue_missing:{shot.id}")
    return issues


def _write_qa_publish_report(
    *,
    plan: MoviePlan,
    plan_id: str,
    run_id: str,
    output_dir: Path,
    issues: Sequence[str],
    qa_mode: str,
    qa_passed: bool,
    qa_skipped: bool,
    duration_s: float,
    frame_rate: float,
    resolution: str,
) -> tuple[Path, Dict[str, Any]]:
    report_path = output_dir / "final" / f"{plan_id}-qa_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_id": plan_id,
        "run_id": run_id,
        "title": plan.title,
        "qa_mode": qa_mode,
        "qa_passed": qa_passed,
        "qa_skipped": qa_skipped,
        "issues": list(issues),
        "issue_count": len(issues),
        "duration_s": duration_s,
        "frame_rate": frame_rate,
        "resolution_px": resolution,
        "generated_at": _now_iso(),
    }
    payload.setdefault("summary", "pass" if qa_passed else "manual_review")
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path, payload


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


def _resolve_qa_mode(plan: MoviePlan) -> str:
    value: Optional[str] = None
    if plan.metadata:
        value = plan.metadata.get("qa_mode")
    if not value and plan.render_profile.metadata:
        candidate = plan.render_profile.metadata.get("qa_mode")
        if isinstance(candidate, str):
            value = candidate
    normalized = (value or "full").strip().lower()
    if normalized not in {"full", "skip"}:
        return "full"
    return normalized


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
        value = int(os.environ.get("VIDEOS_AGENT_DEFAULT_FPS", "16"))
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
) -> Callable[[videos_agent.CallbackEvent], None]:
    def _forward(event: videos_agent.CallbackEvent) -> None:
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
                "videos_agent_progress": dict(event),
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
