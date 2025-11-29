from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from pydantic import ValidationError

from . import adk_helpers, observability, telemetry, videos_agent
from .images_agent import RateLimitExceeded, RateLimitQueued
from .ratelimit import RateLimitDecision
from .schemas import MoviePlan, ShotSpec


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
    dialogue_path: Optional[Path] = None
    video_path: Optional[Path] = None
    lipsync_path: Optional[Path] = None


def execute_plan(
    plan: MoviePlan | Mapping[str, Any],
    *,
    mode: Literal["dry", "run"] = "dry",
    progress_callback: Optional[Callable[[StepExecutionRecord], None]] = None,
    config: Optional[ProductionAgentConfig] = None,
) -> ProductionResult:
    """Execute or simulate a MoviePlan."""

    cfg = config or ProductionAgentConfig()
    model = _coerce_plan(plan)
    _validate_plan(model)
    policy_decisions = _run_policy_checks(model)
    plan_id = _plan_identifier(model)
    run_id = observability.get_session_id()

    if mode == "dry":
        report = _simulate_execution_report(model, policy_decisions)
        _record_summary_event(run_id, plan_id, "dry", [], simulation=report)
        return ProductionResult([], steps=[], simulation_report=report)

    output_dir = _resolve_output_dir(run_id, plan_id)
    records: List[StepExecutionRecord] = []
    shot_artifacts: List[_ShotArtifacts] = []

    try:
        for shot in model.shots:
            artifacts = _execute_shot(
                shot,
                plan_id=plan_id,
                run_id=run_id,
                output_dir=output_dir,
                cfg=cfg,
                progress_callback=progress_callback,
                records=records,
            )
            shot_artifacts.append(artifacts)
    except StepRateLimitError:
        _record_summary_event(run_id, plan_id, "run", records)
        raise

    try:
        final_record, final_path = _run_step(
            plan_id=plan_id,
            run_id=run_id,
            step_id=f"{plan_id}:assemble",
            step_type="assemble",
            gate_flag=None,
            cfg=cfg,
            progress_callback=progress_callback,
            action=lambda: _assemble_plan(model, shot_artifacts, output_dir),
            meta={"shot_count": len(model.shots)},
        )
        records.append(final_record)
    except StepRateLimitError as exc:
        records.append(exc.record)
        _record_summary_event(run_id, plan_id, "run", records)
        raise

    artifact_ref = adk_helpers.publish_artifact(
        local_path=final_path,
        artifact_type=cfg.artifact_type,
        metadata={"plan_id": plan_id, "shot_count": len(model.shots)},
    )
    telemetry.emit_event(
        "production_agent.execute_plan.completed",
        {
            "plan_id": plan_id,
            "artifact_uri": artifact_ref["uri"],
            "shot_count": len(model.shots),
        },
    )
    _record_summary_event(run_id, plan_id, "run", records)
    return ProductionResult([artifact_ref], steps=records)


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


def _run_policy_checks(plan: MoviePlan) -> List[str]:
    banned_keywords = {"weaponized", "forbidden"}
    decisions: List[str] = []
    for shot in plan.shots:
        text = " ".join(
            filter(
                None,
                [shot.visual_description, shot.start_frame_prompt, shot.end_frame_prompt],
            )
        ).lower()
        if any(keyword in text for keyword in banned_keywords):
            raise PlanPolicyViolation(f"Shot {shot.id} violates content policy")
        if shot.duration_sec > 120:
            decisions.append(f"shot:{shot.id} exceeds duration target")
    return decisions


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


def _execute_shot(
    shot: ShotSpec,
    *,
    plan_id: str,
    run_id: str,
    output_dir: Path,
    cfg: ProductionAgentConfig,
    progress_callback: Optional[Callable[[StepExecutionRecord], None]],
    records: List[StepExecutionRecord],
) -> _ShotArtifacts:
    artifacts = _ShotArtifacts(shot_id=shot.id)

    def _append_record(record: StepExecutionRecord) -> None:
        records.append(record)

    def _run_with_tracking(**kwargs: Any) -> Path:
        try:
            record, path = _run_step(**kwargs)
        except StepRateLimitError as exc:
            _append_record(exc.record)
            raise
        _append_record(record)
        return path

    artifacts.frames_path = _run_with_tracking(
        plan_id=plan_id,
        run_id=run_id,
        step_id=f"{shot.id}:images",
        step_type="images",
        gate_flag=cfg.adapters_flag,
        cfg=cfg,
        progress_callback=progress_callback,
        action=lambda: _render_frames(shot, output_dir),
        meta={"shot_id": shot.id},
    )

    if shot.dialogue:
        artifacts.dialogue_path = _run_with_tracking(
            plan_id=plan_id,
            run_id=run_id,
            step_id=f"{shot.id}:tts",
            step_type="tts",
            gate_flag=cfg.tts_flag,
            cfg=cfg,
            progress_callback=progress_callback,
            action=lambda: _synthesize_dialogue(shot, output_dir),
            meta={"shot_id": shot.id, "lines": len(shot.dialogue)},
        )

    artifacts.video_path = _run_with_tracking(
        plan_id=plan_id,
        run_id=run_id,
        step_id=f"{shot.id}:video",
        step_type="video",
        gate_flag=cfg.adapters_flag,
        cfg=cfg,
        progress_callback=progress_callback,
        action=lambda: _render_video_clip(shot, output_dir, plan_id, run_id, progress_callback),
        meta={"shot_id": shot.id, "duration_sec": shot.duration_sec},
    )

    if shot.is_talking_closeup and artifacts.dialogue_path and artifacts.video_path:
        artifacts.lipsync_path = _run_with_tracking(
            plan_id=plan_id,
            run_id=run_id,
            step_id=f"{shot.id}:lipsync",
            step_type="lipsync",
            gate_flag=cfg.lipsync_flag,
            cfg=cfg,
            progress_callback=progress_callback,
            action=lambda: _lipsync_clip(shot, output_dir),
            meta={"shot_id": shot.id},
        )

    return artifacts


def _run_step(
    *,
    plan_id: str,
    run_id: str,
    step_id: str,
    step_type: str,
    gate_flag: Optional[str],
    cfg: ProductionAgentConfig,
    action: Callable[[], Optional[Path]],
    meta: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[StepExecutionRecord], None]] = None,
) -> tuple[StepExecutionRecord, Optional[Path]]:
    should_execute = gate_flag is None or _flag_enabled(gate_flag)
    start = _now_iso()
    start_dt = datetime.now(timezone.utc)
    attempts = 0
    artifact_path: Optional[Path] = None
    status: Literal["queued", "succeeded", "failed", "simulated"]
    error_type: Optional[str] = None
    rate_limit_state: Optional[str] = None
    meta_payload: Dict[str, Any] = dict(meta or {})

    if not should_execute:
        status = "simulated"
    else:
        status = "failed"
        for attempt in range(1, max(cfg.max_attempts, 1) + 1):
            attempts = attempt
            try:
                artifact_path = action()
                status = "succeeded"
                break
            except RateLimitQueued as exc:
                status = "queued"  # type: ignore[assignment]
                error_type = exc.__class__.__name__
                meta_payload["rate_limit"] = _rate_limit_meta(exc.decision)
                rate_limit_state = "queued"
                break
            except RateLimitExceeded as exc:
                status = "failed"
                error_type = exc.__class__.__name__
                meta_payload["rate_limit"] = _rate_limit_meta(exc.decision)
                rate_limit_state = "exceeded"
                break
            except StepTransientError:
                if attempt >= cfg.max_attempts:
                    error_type = "StepTransientError"
                    break
                time.sleep(cfg.retry_delay(attempt))
                continue
            except Exception as exc:  # pragma: no cover - defensive fallback
                error_type = exc.__class__.__name__
                break
        else:  # pragma: no cover
            error_type = "UnknownError"

    end_dt = datetime.now(timezone.utc)
    duration = (end_dt - start_dt).total_seconds()
    record = StepExecutionRecord(
        plan_id=plan_id,
        step_id=step_id,
        step_type=step_type,
        status=status,  # type: ignore[arg-type]
        start_time=start,
        end_time=_to_iso(end_dt),
        duration_s=duration,
        attempts=attempts,
        artifact_uri=str(artifact_path) if artifact_path else None,
        error_type=error_type,
        meta=meta_payload,
    )
    _emit_step_record(run_id, record, progress_callback)
    if should_execute:
        if status == "succeeded":
            return record, artifact_path
        if rate_limit_state == "queued":
            raise StepQueuedError(f"{step_id} queued by rate limiter", record=record)
        if rate_limit_state == "exceeded":
            raise StepRateLimitExceededError(f"{step_id} hit rate limit", record=record)
        raise StepExecutionError(f"{step_id} failed after {attempts} attempts")
    return record, artifact_path


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


def _render_frames(shot: ShotSpec, output_dir: Path) -> Path:
    payload = {
        "shot_id": shot.id,
        "visual_description": shot.visual_description,
        "start_frame_prompt": shot.start_frame_prompt,
        "end_frame_prompt": shot.end_frame_prompt,
    }
    dest = output_dir / "frames" / f"{shot.id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def _synthesize_dialogue(shot: ShotSpec, output_dir: Path) -> Path:
    text = "\n".join(line.text for line in shot.dialogue)
    dest = output_dir / "audio" / f"{shot.id}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(("RIFF" + text).encode("utf-8"))
    return dest


def _render_video_clip(
    shot: ShotSpec,
    output_dir: Path,
    plan_id: str,
    run_id: str,
    progress_callback: Optional[Callable[[StepExecutionRecord], None]] = None,
) -> Path:
    video_dir = output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / f"{shot.id}.mp4"

    fps = _default_video_fps()
    num_frames = _estimate_frame_count(shot.duration_sec, fps)
    prompt_parts = [shot.visual_description or "", shot.motion_prompt or ""]
    prompt = " | ".join(part for part in prompt_parts if part) or f"Shot {shot.id}"

    start_frames = [_encode_prompt_bytes(shot.start_frame_prompt)] if shot.start_frame_prompt else []
    end_frames = [_encode_prompt_bytes(shot.end_frame_prompt)] if shot.end_frame_prompt else []

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
                "dialogue": str(art.dialogue_path) if art.dialogue_path else None,
                "video": str(art.video_path) if art.video_path else None,
                "lipsync": str(art.lipsync_path) if art.lipsync_path else None,
            }
            for art in shots
        ],
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


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
