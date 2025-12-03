from __future__ import annotations

"""Typed request/response envelopes for the production_agent FunctionTool."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from sparkle_motion.schemas import MoviePlan

__all__ = [
    "ProductionAgentRequest",
    "QueueInfo",
    "ProductionAgentResponse",
    "ControlRequest",
]


class ProductionAgentRequest(BaseModel):
    """Canonical request payload for production_agent."""

    plan: Optional[MoviePlan] = None
    plan_uri: Optional[str] = Field(default=None, description="Optional URI pointing to a MoviePlan artifact")
    mode: Literal["dry", "run"] = "dry"

    @model_validator(mode="after")
    def _ensure_plan_source(self) -> "ProductionAgentRequest":
        if not self.plan and not self.plan_uri:
            raise ValueError("Provide either 'plan' or 'plan_uri'")
        return self


class QueueInfo(BaseModel):
    """Metadata describing a queued production run."""

    ticket_id: str
    plan_id: str
    plan_title: str
    step_id: str
    eta_seconds: float
    eta_epoch_s: float
    attempt: int
    max_attempts: int
    message: str


class ProductionAgentResponse(BaseModel):
    """Canonical response envelope for production_agent."""

    status: Literal["success", "error", "queued", "stopped"]
    request_id: str
    run_id: str
    artifact_uris: List[str] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    simulation_report: Optional[Dict[str, Any]] = None
    queue: Optional[QueueInfo] = None
    schema_uri: Optional[str] = None


class ControlRequest(BaseModel):
    """Simple payload for pause/resume/stop control endpoints."""

    run_id: str
