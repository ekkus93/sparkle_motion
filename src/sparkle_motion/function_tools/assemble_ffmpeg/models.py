"""Typed request/response envelopes for the assemble_ffmpeg FunctionTool."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "AssembleClip",
    "AssembleAudio",
    "AssembleOptions",
    "AssembleRequest",
    "AssembleResponse",
]


class AssembleClip(BaseModel):
    """Represents a single video clip that will be concatenated."""

    uri: str
    start_s: float = 0.0
    end_s: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    transition: Optional[Dict[str, Any]] = None


class AssembleAudio(BaseModel):
    """Optional audio bed applied over the assembled clips."""

    uri: str
    start_s: float = 0.0
    end_s: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    gain_db: Optional[float] = None


class AssembleOptions(BaseModel):
    """Adapter-level overrides for assemble_ffmpeg."""

    video_codec: Optional[str] = Field(default="libx264")
    audio_codec: Optional[str] = Field(default="aac")
    pix_fmt: Optional[str] = Field(default="yuv420p")
    crf: Optional[int] = Field(default=18, ge=0)
    preset: Optional[str] = Field(default="veryslow")
    audio_bitrate: Optional[str] = Field(default="192k")
    timeout_s: Optional[float] = Field(default=120.0, gt=0)
    retries: Optional[int] = Field(default=0, ge=0)
    fixture_only: Optional[bool] = None


class AssembleRequest(BaseModel):
    """Canonical request payload for assemble_ffmpeg."""

    plan_id: Optional[str] = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    seed: Optional[int] = None
    clips: List[AssembleClip]
    audio: Optional[AssembleAudio] = None
    options: Optional[AssembleOptions] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_clips(self) -> "AssembleRequest":
        if not self.clips:
            raise ValueError("At least one clip is required")
        return self


class AssembleResponse(BaseModel):
    """Canonical response envelope for assemble_ffmpeg."""

    status: Literal["success", "error"]
    artifact_uri: Optional[str]
    request_id: str
    metadata: Optional[Dict[str, Any]] = None
