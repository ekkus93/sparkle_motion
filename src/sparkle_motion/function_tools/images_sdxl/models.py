from __future__ import annotations

"""Typed request/response envelopes for the images_sdxl FunctionTool."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "ImagesSDXLArtifact",
    "ImagesSDXLRequest",
    "ImagesSDXLResponse",
]


class ImagesSDXLArtifact(BaseModel):
    """Metadata for each rendered image artifact."""

    artifact_uri: str
    metadata: Dict[str, Any]


class ImagesSDXLRequest(BaseModel):
    """Canonical request payload for the images_sdxl FunctionTool."""

    prompt: str = Field(min_length=1, description="Primary SDXL prompt text")
    negative_prompt: Optional[str] = Field(
        default=None, description="Primary negative prompt to steer undesired concepts"
    )
    prompt_2: Optional[str] = Field(
        default=None,
        description="Optional secondary prompt for refiner/micro-conditioning",
    )
    negative_prompt_2: Optional[str] = Field(
        default=None, description="Optional secondary negative prompt",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata propagated into artifact records",
    )
    plan_id: Optional[str] = Field(
        default=None,
        description="Plan identifier for bookkeeping/observability",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Run identifier for artifact publication + observability",
    )
    base_image_id: Optional[str] = Field(
        default=None,
        description="Identifier of the base-image slot being rendered (len(shots)+1).",
    )
    batch_start: int = Field(default=0, ge=0, description="Offset into the base-images list")
    count: int = Field(default=1, ge=1, le=8, description="Number of images to render")
    seed: Optional[int] = Field(default=None, description="Deterministic seed override")
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=30, ge=1, le=200)
    cfg_scale: float = Field(default=7.5, ge=0.0, le=30.0)
    sampler: str = Field(default="ddim", min_length=1)
    denoising_start: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    denoising_end: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_prompt_and_dims(self) -> "ImagesSDXLRequest":
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        self.prompt = prompt
        if self.width % 8 or self.height % 8:
            raise ValueError("width and height must be divisible by 8")
        if self.denoising_start is not None and self.denoising_end is not None:
            if self.denoising_end <= self.denoising_start:
                raise ValueError("denoising_end must be greater than denoising_start")
        return self


class ImagesSDXLResponse(BaseModel):
    """Canonical response envelope for the images_sdxl FunctionTool."""

    status: Literal["success", "error"]
    request_id: str
    artifact_uri: Optional[str]
    artifacts: List[ImagesSDXLArtifact]
