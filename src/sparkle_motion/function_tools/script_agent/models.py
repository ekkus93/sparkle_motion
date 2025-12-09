"""Typed request/response envelopes for the script_agent FunctionTool."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = ["ScriptAgentRequest", "ScriptAgentResponse"]


class ScriptAgentRequest(BaseModel):
    """Canonical request payload for the script_agent FunctionTool."""

    title: Optional[str] = Field(default=None, description="Optional working title for the MoviePlan")
    shots: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional set of pre-seeded shots (dict form) for refinement",
    )
    prompt: Optional[str] = Field(default=None, description="High-level narrative prompt for plan generation")

    @model_validator(mode="after")
    def _at_least_one(self) -> "ScriptAgentRequest":
        if not (self.prompt or self.title or (self.shots and len(self.shots) > 0)):
            raise ValueError("Provide prompt, title, or shots")
        return self


class ScriptAgentResponse(BaseModel):
    """Canonical response envelope for the script_agent FunctionTool."""

    status: Literal["success", "error"]
    artifact_uri: Optional[str] = None
    request_id: str
    schema_uri: Optional[str] = None
