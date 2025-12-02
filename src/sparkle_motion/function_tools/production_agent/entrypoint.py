from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Mapping, Optional, Literal, Sequence

from fastapi import FastAPI, HTTPException

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
from sparkle_motion.function_tools.production_agent.models import (
    ControlRequest,
    ProductionAgentRequest,
    ProductionAgentResponse,
    QueueInfo,
)

LOG = logging.getLogger("production_agent.entrypoint")
LOG.setLevel(logging.INFO)


RequestModel = ProductionAgentRequest
ResponseModel = ProductionAgentResponse


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
    plan_metadata = dict(plan_model.metadata or {})
    plan_metadata["qa_mode"] = req.qa_mode
    plan_metadata["qa_skipped"] = req.qa_mode == "skip"
    plan_model.metadata = plan_metadata
    try:
        schema_uri = schema_registry.movie_plan_schema().uri
    except Exception:
        schema_uri = None

    run_id = _generate_run_id()
    run_context = RunContext.from_plan(
        plan_model,
        run_id=run_id,
        schema_uri=schema_uri,
        metadata={"qa_mode": req.qa_mode, "qa_skipped": req.qa_mode == "skip"},
    )
    plan_payload = plan_model.model_dump()

    plan_id = run_context.plan_id
    plan_title = run_context.plan_title
    expected_steps = _estimate_expected_steps(plan_model)
    registry.start_run(
        run_id=run_id,
        plan_id=plan_id,
        plan_title=plan_title,
        mode=req.mode,
        expected_steps=expected_steps,
        render_profile=run_context.render_profile,
        run_metadata=run_context.metadata,
        qa_mode=req.qa_mode,
        schema_uri=schema_uri,
    )

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
            qa_mode=req.qa_mode,
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
    stage_filter = stage.strip() if stage else None
    try:
        if stage_filter:
            grouped_entries: Dict[str, List[Dict[str, Any]]] = {stage_filter: registry.get_artifacts(run_id, stage=stage_filter)}
        else:
            grouped_entries = registry.get_artifacts_by_stage(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc

    manifests_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    total_count = 0
    for stage_id, entries in grouped_entries.items():
        stage_manifests: List[Dict[str, Any]] = []
        for entry in entries:
            try:
                manifest = _entry_to_manifest(entry, run_id)
            except HTTPException:
                raise
            except Exception as exc:  # pragma: no cover - defensive validation guard
                LOG.exception(
                    "Failed to validate manifest entry",
                    extra={"run_id": run_id, "stage": entry.get("stage"), "error": str(exc)},
                )
                raise HTTPException(status_code=500, detail="Manifest validation failed") from exc
            stage_manifests.append(manifest)
        manifests_by_stage[stage_id] = stage_manifests
        total_count += len(stage_manifests)

    qa_stage_entries = manifests_by_stage.get("qa_publish")
    stage_filter_normalized = stage_filter.lower() if stage_filter else None
    if stage_filter_normalized == "qa_publish":
        _ensure_video_final_manifest_present(qa_stage_entries or [])
    elif qa_stage_entries:
        _ensure_video_final_manifest_present(qa_stage_entries)

    stage_sections: List[Dict[str, Any]] = []
    for stage_id, manifest_entries in manifests_by_stage.items():
        stage_sections.append(_summarize_stage_section(stage_id, manifest_entries))

    if stage_filter:
        artifacts_payload = stage_sections[0]["artifacts"] if stage_sections else []
    else:
        artifacts_payload = [entry for section in stage_sections for entry in section["artifacts"]]

    return {
        "run_id": run_id,
        "artifacts": artifacts_payload,
        "stages": stage_sections,
        "total_artifacts": total_count,
    }


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
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"plan_uri JSON decode failed: {exc}") from exc
    plan_payload = raw_payload
    if isinstance(raw_payload, dict):
        nested = raw_payload.get("validated_plan") or raw_payload.get("plan")
        if isinstance(nested, dict):
            plan_payload = nested
    try:
        return MoviePlan.model_validate(plan_payload) if hasattr(MoviePlan, "model_validate") else MoviePlan.parse_obj(plan_payload)
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
        "qa_skipped": entry.get("qa_skipped"),
        "playback_ready": entry.get("playback_ready"),
        "notes": entry.get("notes"),
        "metadata": entry.get("metadata", {}),
        "created_at": entry.get("created_at", datetime.now(timezone.utc).isoformat()),
    }
    manifest = StageManifest.model_validate(payload)
    manifest_dict = manifest.model_dump()
    _validate_video_final_manifest(manifest_dict)
    return manifest_dict


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
    total = 2  # assemble + qa_publish
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


def _validate_video_final_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("stage_id") != "qa_publish":
        return
    if manifest.get("artifact_type") != "video_final":
        return
    errors: List[str] = []

    def _require_str(field: str) -> None:
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} missing")

    _require_str("artifact_uri")
    _require_str("local_path")
    _require_str("mime_type")
    _require_str("resolution_px")
    _require_str("qa_report_uri")
    _require_str("qa_mode")

    checksum = manifest.get("checksum_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("checksum_sha256 invalid")

    media_type_value = manifest.get("media_type") or manifest.get("mime_type")
    if not isinstance(media_type_value, str) or not media_type_value.lower().startswith("video/"):
        errors.append("media_type invalid")

    size_bytes = manifest.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        errors.append("size_bytes invalid")

    duration_s = manifest.get("duration_s")
    if not isinstance(duration_s, (int, float)) or duration_s <= 0:
        errors.append("duration_s invalid")

    frame_rate = manifest.get("frame_rate")
    if not isinstance(frame_rate, (int, float)) or frame_rate <= 0:
        errors.append("frame_rate invalid")

    storage_hint = manifest.get("storage_hint")
    if storage_hint not in {"adk", "local", "filesystem"}:
        errors.append("storage_hint invalid")

    download_url = manifest.get("download_url")
    local_path = manifest.get("local_path")
    if storage_hint == "adk":
        if not isinstance(download_url, str) or not download_url.strip():
            errors.append("download_url missing for adk storage")
    else:
        if not isinstance(local_path, str) or not local_path.strip():
            errors.append("local_path missing for non-adk storage")
        if download_url not in (None, "") and not isinstance(download_url, str):
            errors.append("download_url invalid")

    qa_passed = manifest.get("qa_passed")
    if not isinstance(qa_passed, bool):
        errors.append("qa_passed invalid")

    playback_ready = manifest.get("playback_ready")
    if not isinstance(playback_ready, bool):
        errors.append("playback_ready invalid")

    if errors:
        raise HTTPException(status_code=500, detail="Invalid qa_publish manifest: " + ", ".join(errors))


def _ensure_video_final_manifest_present(manifests: Sequence[Mapping[str, Any]]) -> None:
    for manifest in manifests:
        if manifest.get("artifact_type") == "video_final" and manifest.get("stage_id") == "qa_publish":
            return
    raise HTTPException(status_code=409, detail="qa_publish manifest missing video_final entry")


def _summarize_stage_section(stage_id: str, manifest_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    artifacts = list(manifest_entries)
    artifact_types = sorted({entry["artifact_type"] for entry in artifacts})
    media_types = sorted({(entry.get("mime_type") or entry.get("media_type")) for entry in artifacts if entry.get("mime_type") or entry.get("media_type")})
    preview = _build_stage_previews(artifacts)
    media_summary = _aggregate_media_summary(artifacts)
    qa_summary = _aggregate_qa_summary(artifacts)
    created_first, created_last = _created_at_range(artifacts)
    total_size = sum(entry.get("size_bytes") or 0 for entry in artifacts if isinstance(entry.get("size_bytes"), int))
    total_duration = sum(entry.get("duration_s") or 0.0 for entry in artifacts if isinstance(entry.get("duration_s"), (int, float)))
    return {
        "stage_id": stage_id,
        "count": len(artifacts),
        "artifact_types": artifact_types,
        "media_types": media_types,
        "media_summary": media_summary,
        "qa_summary": qa_summary,
        "preview": preview,
        "size_bytes_total": total_size,
        "duration_s_total": total_duration,
        "created_at": {
            "first": created_first,
            "last": created_last,
        },
        "artifacts": artifacts,
    }


def _build_stage_previews(artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    preview: Dict[str, Optional[Dict[str, Any]]] = {"image": None, "audio": None, "video": None, "other": None}
    for entry in artifacts:
        category = _classify_media_category(entry)
        if category in preview and preview[category] is None:
            preview[category] = _build_preview_entry(entry)
        if category == "other" and preview["other"] is None:
            preview["other"] = _build_preview_entry(entry)
        if all(preview.values()):
            break
    return preview


def _classify_media_category(entry: Mapping[str, Any]) -> str:
    media_type_value = (entry.get("mime_type") or entry.get("media_type") or "").lower()
    if media_type_value.startswith("video/"):
        return "video"
    if media_type_value.startswith("audio/"):
        return "audio"
    if media_type_value.startswith("image/"):
        return "image"
    return "other"


def _build_preview_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(entry.get("metadata") or {})
    thumbnail = metadata.get("thumbnail") or metadata.get("thumbnail_uri") or metadata.get("preview_image")
    if not thumbnail and _classify_media_category(entry) == "image":
        thumbnail = entry.get("artifact_uri") or entry.get("local_path")
    return {
        "artifact_type": entry.get("artifact_type"),
        "stage_id": entry.get("stage_id"),
        "artifact_uri": entry.get("artifact_uri"),
        "local_path": entry.get("local_path"),
        "download_url": entry.get("download_url"),
        "media_type": entry.get("media_type") or entry.get("mime_type"),
        "thumbnail_uri": thumbnail,
        "duration_s": entry.get("duration_s"),
        "size_bytes": entry.get("size_bytes"),
        "qa_passed": entry.get("qa_passed"),
        "qa_mode": entry.get("qa_mode"),
        "qa_skipped": entry.get("qa_skipped"),
        "playback_ready": entry.get("playback_ready"),
    }


def _aggregate_media_summary(artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for entry in artifacts:
        category = _classify_media_category(entry)
        bucket = summary.setdefault(category, {"count": 0, "total_duration_s": 0.0, "playback_ready": False})
        bucket["count"] += 1
        if isinstance(entry.get("duration_s"), (int, float)):
            bucket["total_duration_s"] += float(entry["duration_s"])
        if entry.get("playback_ready"):
            bucket["playback_ready"] = True
    return summary


def _aggregate_qa_summary(artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    values = [entry.get("qa_passed") for entry in artifacts if entry.get("qa_passed") is not None]
    if not values:
        return {"total": 0, "passed": 0, "failed": 0}
    passed = sum(1 for value in values if value is True)
    failed = sum(1 for value in values if value is False)
    return {"total": len(values), "passed": passed, "failed": failed}


def _created_at_range(artifacts: Sequence[Mapping[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    timestamps = [entry.get("created_at") for entry in artifacts if isinstance(entry.get("created_at"), str)]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


__all__ = ["app", "invoke", "RequestModel", "ResponseModel"]
