"""Typed request/response envelopes for the tts_chatterbox FunctionTool."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "TTSChatterboxRequest",
    "TTSChatterboxResponse",
]


class TTSChatterboxRequest(BaseModel):
    """Canonical request payload for the tts_chatterbox FunctionTool."""

    prompt: Optional[str] = Field(default=None, description="Prompt text that should be synthesized")
    text: Optional[str] = Field(default=None, description="Optional explicit text to synthesize")
    voice_id: str = Field(default="emma", min_length=1)
    language: Optional[str] = Field(default=None, description="BCP47 language tag or None for auto")
    sample_rate: int = Field(default=24000, ge=8000, le=96000)
    bit_depth: int = Field(default=16, ge=8, le=32)
    seed: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    plan_id: Optional[str] = None
    step_id: Optional[str] = None
    run_id: Optional[str] = None

    @model_validator(mode="after")
    def _ensure_prompt_or_text(self) -> "TTSChatterboxRequest":
        text = (self.text or "").strip()
        prompt = (self.prompt or "").strip()
        if not text and not prompt:
            raise ValueError("Provide either 'text' or 'prompt'")
        self.text = text or None
        self.prompt = prompt or None
        return self


class TTSChatterboxResponse(BaseModel):
    """Canonical response envelope for the tts_chatterbox FunctionTool."""

    status: Literal["success", "error"]
    artifact_uri: Optional[str]
    request_id: str
    metadata: Optional[Dict[str, Any]] = None
