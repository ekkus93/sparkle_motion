"""Typed request/response envelopes for the videos_wan FunctionTool."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "VideosWanOptions",
    "VideosWanRequest",
    "VideosWanResponse",
]


class VideosWanOptions(BaseModel):
    """Optional Wan inference parameters forwarded to the adapter."""

    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=128)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0)
    negative_prompt: Optional[str] = None
    motion_bucket_id: Optional[int] = Field(default=None, ge=0)
    megapixels: Optional[float] = Field(default=None, ge=0.0)
    fixture_only: Optional[bool] = None


class VideosWanRequest(BaseModel):
    """Canonical request payload for the videos_wan FunctionTool."""

    prompt: str = Field(min_length=1, description="Primary Wan prompt text")
    plan_id: Optional[str] = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    seed: Optional[int] = None
    chunk_index: Optional[int] = Field(default=None, ge=0)
    chunk_count: Optional[int] = Field(default=None, ge=1)
    num_frames: int = Field(default=64, ge=1, le=2048)
    fps: int = Field(default=24, ge=1, le=120)
    width: int = Field(default=1280, ge=64, le=4096)
    height: int = Field(default=720, ge=64, le=4096)
    metadata: Optional[Dict[str, Any]] = None
    options: Optional[VideosWanOptions] = None
    start_frame_uri: Optional[str] = None
    end_frame_uri: Optional[str] = None

    @model_validator(mode="after")
    def _validate_prompt(self) -> "VideosWanRequest":
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")
        self.prompt = prompt
        return self


class VideosWanResponse(BaseModel):
    """Canonical response envelope for the videos_wan FunctionTool."""

    status: Literal["success", "error"]
    artifact_uri: Optional[str]
    request_id: str
    metadata: Optional[Dict[str, Any]] = None
