from __future__ import annotations

"""Typed request/response envelopes for the lipsync_wav2lip FunctionTool."""

from typing import Any, Dict, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "LipsyncMediaPayload",
    "LipsyncWav2LipOptions",
    "LipsyncWav2LipRequest",
    "LipsyncWav2LipResponse",
]


class LipsyncMediaPayload(BaseModel):
    """Represents an input media source for the lipsync tool."""

    model_config = ConfigDict(populate_by_name=True)

    uri: Optional[str] = None
    path: Optional[str] = None
    data_base64: Optional[str] = Field(default=None, alias="data_b64")

    @model_validator(mode="after")
    def _require_source(self) -> "LipsyncMediaPayload":
        if not (self.uri or self.path or self.data_base64):
            raise ValueError("uri/path or data_b64 required")
        return self


class LipsyncWav2LipOptions(BaseModel):
    """Additional wav2lip execution options passed through to the adapter."""

    model_config = ConfigDict(extra="forbid")

    fixture_only: Optional[bool] = None
    checkpoint_path: Optional[str] = None
    face_det_checkpoint: Optional[str] = None
    pads: Optional[Sequence[int]] = None
    resize_factor: Optional[int] = Field(default=None, ge=1, le=8)
    nosmooth: Optional[bool] = None
    crop: Optional[Sequence[int]] = None
    fps: Optional[float] = Field(default=None, gt=0)
    timeout_s: Optional[int] = Field(default=None, ge=30, le=3600)
    retries: Optional[int] = Field(default=None, ge=0, le=3)
    repo_path: Optional[str] = None
    script_path: Optional[str] = None
    python_bin: Optional[str] = None
    fixture_seed: Optional[int] = None
    allow_fixture_fallback: Optional[bool] = True


class LipsyncWav2LipRequest(BaseModel):
    """Canonical request payload for the lipsync_wav2lip FunctionTool."""

    model_config = ConfigDict(populate_by_name=True)

    face: LipsyncMediaPayload
    audio: LipsyncMediaPayload
    plan_id: Optional[str] = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    movie_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    out_basename: Optional[str] = None
    options: Optional[LipsyncWav2LipOptions] = None


class LipsyncWav2LipResponse(BaseModel):
    """Canonical response envelope for the lipsync_wav2lip FunctionTool."""

    status: Literal["success", "error"]
    artifact_uri: Optional[str]
    request_id: str
    metadata: Dict[str, Any]
    logs: Dict[str, Any]
