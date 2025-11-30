from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from sparkle_motion import observability, telemetry, schema_registry
from sparkle_motion.queue_runner import QueueTicket, enqueue_plan
from sparkle_motion.production_agent import (
    ProductionAgentError,
    ProductionResult,
    SimulationReport,
    StepExecutionRecord,
    StepQueuedError,
    execute_plan,
)
from sparkle_motion.run_registry import ArtifactEntry, RunHalted, get_run_registry
from sparkle_motion.schemas import MoviePlan, RunContext, StageManifest

LOG = logging.getLogger("production_agent.entrypoint")
LOG.setLevel(logging.INFO)


class RequestModel(BaseModel):
    plan: Optional[MoviePlan] = None
    plan_uri: Optional[str] = Field(default=None, description="Optional path/URI to a MoviePlan JSON artifact")
    mode: Literal["dry", "run"] = "dry"
    qa_mode: Literal["full", "skip"] = "full"

    @model_validator(mode="after")
    def _ensure_plan_source(self) -> "RequestModel":
        if not self.plan and not self.plan_uri:
            raise ValueError("Provide either 'plan' or 'plan_uri'")
        return self


class QueueInfo(BaseModel):
    ticket_id: str
    plan_id: str
    plan_title: str
    step_id: str
    eta_seconds: float
    eta_epoch_s: float
    attempt: int
    max_attempts: int
    message: str


class ResponseModel(BaseModel):
    status: Literal["success", "error", "queued", "stopped"]
    request_id: str
    run_id: str
    artifact_uris: List[str] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    simulation_report: Optional[Dict[str, Any]] = None
    queue: Optional[QueueInfo] = None
    schema_uri: Optional[str] = None


class ControlRequest(BaseModel):
    run_id: str


app = FastAPI(title="production_agent Entrypoint")
registry = get_run_registry()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, bool]:
    return {"ready": True}


@app.post("/invoke")
def invoke(req: RequestModel) -> Dict[str, Any]:
    request_id = uuid.uuid4().hex
    LOG.info("production_agent.invoke", extra={"mode": req.mode, "request_id": request_id})
    plan_model = req.plan or _load_plan_from_uri(req.plan_uri)
    if plan_model is None:
        raise HTTPException(status_code=400, detail="Unable to load MoviePlan payload")
    try:
        schema_uri = schema_registry.movie_plan_schema().uri
    except Exception:
        schema_uri = None

    run_id = _generate_run_id()
    run_context = RunContext.from_plan(plan_model, run_id=run_id, schema_uri=schema_uri)
    plan_payload = plan_model.model_dump()

    plan_id = run_context.plan_id
    plan_title = run_context.plan_title
    expected_steps = _estimate_expected_steps(plan_model)
    registry.start_run(run_id=run_id, plan_id=plan_id, plan_title=plan_title, mode=req.mode, expected_steps=expected_steps)

    def _progress(record: StepExecutionRecord) -> None:
        registry.record_step(run_id, record.as_dict())

    pre_step_hook = registry.pre_step_hook(run_id)

    try:
        result = execute_plan(
            plan_model,
            mode=req.mode,
            progress_callback=_progress,
            run_id=run_id,
            pre_step_hook=pre_step_hook,
        )
    except StepQueuedError as exc:
        LOG.warning(
            "production_agent.execute_plan queued",
            extra={"request_id": request_id, "step_id": exc.record.step_id},
        )
        try:
            ticket = enqueue_plan(plan_payload=plan_payload, mode=req.mode, queued_record=exc.record, run_id=run_id)
        except ValueError as enqueue_error:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(enqueue_error)) from enqueue_error
        registry.mark_queued(run_id, ticket.ticket_id)
        payload = _queued_response(request_id, run_id, ticket, schema_uri=schema_uri)
        _emit_events(req.mode, payload)
        return payload
    except RunHalted as exc:
        registry.stop_run(run_id, reason=str(exc))
        payload = ResponseModel(
            status="stopped",
            request_id=request_id,
            run_id=run_id,
            artifact_uris=[],
            steps=registry.get_status(run_id).get("steps", []),
            schema_uri=schema_uri,
        )
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        _emit_events(req.mode, data)
        return data
    except ProductionAgentError as exc:
        LOG.exception("production_agent.execute_plan failed", extra={"request_id": request_id})
        registry.fail_run(run_id, error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        LOG.exception("production_agent.execute_plan unexpected error", extra={"request_id": request_id})
        registry.fail_run(run_id, error=str(exc))
        raise HTTPException(status_code=500, detail="execution failed") from exc

    registry.complete_run(run_id)
    response = ResponseModel(
        status="success",
        request_id=request_id,
        run_id=run_id,
        artifact_uris=_extract_artifact_uris(result),
        steps=[_record_to_dict(record) for record in result.steps],
        simulation_report=_simulation_report_to_dict(result.simulation_report),
        schema_uri=schema_uri,
    )
    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    _emit_events(req.mode, payload)
    return payload


@app.get("/status")
def status(run_id: str) -> Dict[str, Any]:
    try:
        return registry.get_status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc


@app.get("/artifacts")
def artifacts(run_id: str, stage: Optional[str] = None) -> Dict[str, Any]:
    try:
        entries = registry.get_artifacts(run_id, stage=stage)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc
    manifests: List[Dict[str, Any]] = []
    for entry in entries:
        try:
            manifest = _entry_to_manifest(entry, run_id)
            manifests.append(manifest)
        except Exception as exc:  # pragma: no cover - defensive validation guard
            LOG.exception("Failed to validate manifest entry", extra={"run_id": run_id, "stage": entry.get("stage"), "error": str(exc)})
            raise HTTPException(status_code=500, detail="Manifest validation failed") from exc
    return {"run_id": run_id, "artifacts": manifests}


@app.post("/control/pause")
def control_pause(req: ControlRequest) -> Dict[str, Any]:
    return _control_response("pause", req.run_id)


@app.post("/control/resume")
def control_resume(req: ControlRequest) -> Dict[str, Any]:
    return _control_response("resume", req.run_id)


@app.post("/control/stop")
def control_stop(req: ControlRequest) -> Dict[str, Any]:
    return _control_response("stop", req.run_id)


def _load_plan_from_uri(uri: Optional[str]) -> Optional[MoviePlan]:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise HTTPException(status_code=400, detail=f"Unsupported plan_uri scheme: {parsed.scheme}")
    path = Path(parsed.path if parsed.scheme else uri).expanduser()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"plan_uri path not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"plan_uri JSON decode failed: {exc}") from exc
    try:
        return MoviePlan.model_validate(payload) if hasattr(MoviePlan, "model_validate") else MoviePlan.parse_obj(payload)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"plan_uri schema validation failed: {exc}") from exc


def _extract_artifact_uris(result: ProductionResult) -> List[str]:
    return [ref.get("uri", "") for ref in result if isinstance(ref, dict) and ref.get("uri")]


def _record_to_dict(record: StepExecutionRecord) -> Dict[str, Any]:
    return record.as_dict()


def _simulation_report_to_dict(report: Optional[SimulationReport]) -> Optional[Dict[str, Any]]:
    if report is None:
        return None
    return {
        "plan_id": report.plan_id,
        "resource_summary": report.resource_summary,
        "policy_decisions": list(report.policy_decisions),
        "steps": [asdict(step) for step in report.steps],
    }


def _queued_response(request_id: str, run_id: str, ticket: QueueTicket, *, schema_uri: Optional[str]) -> Dict[str, Any]:
    info = QueueInfo(
        ticket_id=ticket.ticket_id,
        plan_id=ticket.plan_id,
        plan_title=ticket.plan_title,
        step_id=ticket.step_id,
        eta_seconds=ticket.eta_seconds(),
        eta_epoch_s=ticket.eta_epoch_s,
        attempt=ticket.attempt,
        max_attempts=ticket.max_attempts,
        message=ticket.message,
    )
    response = ResponseModel(
        status="queued",
        request_id=request_id,
        run_id=run_id,
        queue=info,
        schema_uri=schema_uri,
    )
    return response.model_dump() if hasattr(response, "model_dump") else response.dict()


def _emit_events(mode: str, payload: Dict[str, Any]) -> None:
    try:
        run_id = payload.get("run_id") or observability.get_session_id()
        telemetry.emit_event(
            "production_agent.entrypoint.completed",
            {
                "run_id": run_id,
                "mode": mode,
                "artifact_uris": payload.get("artifact_uris", []),
                "step_count": len(payload.get("steps", [])),
            },
        )
    except Exception:
        pass


def _control_response(command: Literal["pause", "resume", "stop"], run_id: str) -> Dict[str, Any]:
    try:
        if command == "pause":
            state = registry.request_pause(run_id)
        elif command == "resume":
            state = registry.request_resume(run_id)
        else:
            state = registry.request_stop(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc
    return {"status": "acknowledged", "run": state}


def _entry_to_manifest(entry: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    payload = {
        "run_id": run_id,
        "stage_id": entry.get("stage", "unknown"),
        "artifact_type": entry.get("artifact_type", "artifact"),
        "name": entry.get("name", "artifact"),
        "artifact_uri": entry.get("artifact_uri", ""),
        "media_type": entry.get("media_type"),
        "local_path": entry.get("local_path"),
        "download_url": entry.get("download_url"),
        "storage_hint": entry.get("storage_hint"),
        "mime_type": entry.get("mime_type"),
        "size_bytes": entry.get("size_bytes"),
        "duration_s": entry.get("duration_s"),
        "frame_rate": entry.get("frame_rate"),
        "resolution_px": entry.get("resolution_px"),
        "checksum_sha256": entry.get("checksum_sha256"),
        "qa_report_uri": entry.get("qa_report_uri"),
        "qa_passed": entry.get("qa_passed"),
        "qa_mode": entry.get("qa_mode"),
        "playback_ready": entry.get("playback_ready"),
        "notes": entry.get("notes"),
        "metadata": entry.get("metadata", {}),
        "created_at": entry.get("created_at", datetime.now(timezone.utc).isoformat()),
    }
    manifest = StageManifest.model_validate(payload)
    return manifest.model_dump()


def _generate_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def _plan_slug_from_payload(plan: MoviePlan | Mapping[str, Any]) -> str:
    if isinstance(plan, MoviePlan):
        metadata = plan.metadata or {}
        slug = metadata.get("plan_id")
        if slug:
            return slug
        title = plan.title.strip().lower().replace(" ", "-") if plan.title else ""
        return title[:48] or "plan"

    metadata = plan.get("metadata") if isinstance(plan, Mapping) else None
    if isinstance(metadata, Mapping):
        slug = metadata.get("plan_id")
        if isinstance(slug, str) and slug.strip():
            return slug.strip()
    title = plan.get("title") if isinstance(plan, Mapping) else None
    if isinstance(title, str) and title.strip():
        slug = title.strip().lower().replace(" ", "-")
        return slug[:48] or "plan"
    return f"plan-{uuid.uuid4().hex[:8]}"


def _estimate_expected_steps(plan: MoviePlan | Mapping[str, Any]) -> Optional[int]:
    if isinstance(plan, MoviePlan):
        shots_iter = plan.shots
    else:
        shots_iter = plan.get("shots") if isinstance(plan, Mapping) else None
        if not isinstance(shots_iter, list):
            return None
    total = 1  # final assemble/qa
    for shot in shots_iter:
        shot_mapping: Mapping[str, Any]
        if isinstance(shot, dict):
            shot_mapping = shot
        elif hasattr(shot, "model_dump"):
            shot_mapping = shot.model_dump()
        else:
            continue
        total += 2  # images + video
        if shot_mapping.get("dialogue"):
            total += 1
        if shot_mapping.get("is_talking_closeup"):
            total += 1
    return total


__all__ = ["app", "invoke", "RequestModel", "ResponseModel"]
