from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from sparkle_motion import observability, telemetry
from sparkle_motion.production_agent import (
    ProductionAgentError,
    ProductionResult,
    SimulationReport,
    StepExecutionRecord,
    execute_plan,
)

LOG = logging.getLogger("production_agent.entrypoint")
LOG.setLevel(logging.INFO)


class RequestModel(BaseModel):
    plan: Optional[Dict[str, Any]] = None
    plan_uri: Optional[str] = Field(default=None, description="Optional path/URI to a MoviePlan JSON artifact")
    mode: Literal["dry", "run"] = "dry"

    @model_validator(mode="after")
    def _ensure_plan_source(self) -> "RequestModel":
        if not self.plan and not self.plan_uri:
            raise ValueError("Provide either 'plan' or 'plan_uri'")
        return self


class ResponseModel(BaseModel):
    status: Literal["success", "error"]
    request_id: str
    artifact_uris: List[str] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    simulation_report: Optional[Dict[str, Any]] = None


app = FastAPI(title="production_agent Entrypoint")


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
    plan_payload = req.plan or _load_plan_from_uri(req.plan_uri)
    if plan_payload is None:
        raise HTTPException(status_code=400, detail="Unable to load MoviePlan payload")

    try:
        result = execute_plan(plan_payload, mode=req.mode)
    except ProductionAgentError as exc:
        LOG.exception("production_agent.execute_plan failed", extra={"request_id": request_id})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        LOG.exception("production_agent.execute_plan unexpected error", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="execution failed") from exc

    response = ResponseModel(
        status="success",
        request_id=request_id,
        artifact_uris=_extract_artifact_uris(result),
        steps=[_record_to_dict(record) for record in result.steps],
        simulation_report=_simulation_report_to_dict(result.simulation_report),
    )
    payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
    _emit_events(req.mode, payload)
    return payload


def _load_plan_from_uri(uri: Optional[str]) -> Optional[Dict[str, Any]]:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise HTTPException(status_code=400, detail=f"Unsupported plan_uri scheme: {parsed.scheme}")
    path = Path(parsed.path if parsed.scheme else uri).expanduser()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"plan_uri path not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"plan_uri JSON decode failed: {exc}") from exc


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


def _emit_events(mode: str, payload: Dict[str, Any]) -> None:
    try:
        run_id = observability.get_session_id()
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


__all__ = ["app", "invoke", "RequestModel", "ResponseModel"]
