from __future__ import annotations

"""Typed request/response envelopes for the qa_qwen2vl FunctionTool."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "QaFramePayload",
    "QaQwen2VlOptions",
    "QaQwen2VlRequest",
    "QaQwen2VlResponse",
]


class QaFramePayload(BaseModel):
    """Represents a single frame inspected by the QA adapter."""

    id: Optional[str] = None
    uri: Optional[str] = None
    data_base64: Optional[str] = Field(default=None, alias="data_b64")
    prompt: Optional[str] = None

    @model_validator(mode="after")
    def _ensure_source(self) -> "QaFramePayload":
        if not self.uri and not self.data_base64:
            raise ValueError("each frame must provide uri or data_base64")
        return self


class QaQwen2VlOptions(BaseModel):
    """Adapter-level overrides supported by qa_qwen2vl."""

    fixture_only: Optional[bool] = None
    max_new_tokens: Optional[int] = Field(default=None, ge=16, le=1024)
    policy_path: Optional[str] = None
    fixture_seed: Optional[int] = None
    model_id: Optional[str] = None
    force_real_engine: Optional[bool] = None
    dtype: Optional[str] = None
    attention: Optional[str] = None
    min_pixels: Optional[int] = Field(default=None, ge=64, le=4096)
    max_pixels: Optional[int] = Field(default=None, ge=64, le=6144)
    cache_ttl_s: Optional[float] = Field(default=None, ge=60.0, le=7200.0)
    max_download_bytes: Optional[int] = Field(default=None, ge=1024, le=50 * 1024 * 1024)
    download_timeout_s: Optional[float] = Field(default=None, ge=1.0, le=120.0)
    metadata: Optional[Dict[str, Any]] = None


class QaQwen2VlRequest(BaseModel):
    """Canonical request payload for qa_qwen2vl."""

    frames: List[QaFramePayload]
    prompt: Optional[str] = None
    plan_id: Optional[str] = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    movie_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    options: Optional[QaQwen2VlOptions] = None

    @model_validator(mode="after")
    def _validate_frames(self) -> "QaQwen2VlRequest":
        if not self.frames:
            raise ValueError("frames must be provided")
        has_prompt = bool((self.prompt or "").strip()) or any((frame.prompt or "").strip() for frame in self.frames)
        if not has_prompt:
            raise ValueError("provide prompt either globally or per frame")
        return self


class QaQwen2VlResponse(BaseModel):
    """Canonical response envelope for qa_qwen2vl."""

    status: Literal["success"]
    request_id: str
    decision: Literal["approve", "regenerate", "escalate", "pending"]
    artifact_uri: str
    metadata: Dict[str, Any]
    report: Dict[str, Any]
    human_task_id: Optional[str] = None
