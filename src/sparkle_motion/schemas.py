from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CharacterSpec(BaseModel):
    """Description of a character used in the movie."""

    id: str
    name: str
    description: Optional[str] = None
    voice_profile: Dict[str, str] = Field(default_factory=dict)


class DialogueLine(BaseModel):
    character_id: str
    text: str
    start_time_sec: Optional[float] = None


class ShotSpec(BaseModel):
    """A single shot specification used by the pipeline."""

    id: str
    duration_sec: float = Field(..., gt=0)
    setting: Optional[str] = None
    visual_description: str
    start_frame_prompt: str
    end_frame_prompt: str
    motion_prompt: Optional[str] = None
    is_talking_closeup: bool = False
    dialogue: List[DialogueLine] = Field(default_factory=list)


class AssetRefsShot(BaseModel):
    """Paths to generated assets for a shot (file paths or Drive URIs)."""

    start_frame: Optional[str] = None
    end_frame: Optional[str] = None
    raw_clip: Optional[str] = None
    dialogue_audio: List[str] = Field(default_factory=list)
    final_video_clip: Optional[str] = None


class AssetRefs(BaseModel):
    """Top-level asset registry keyed by shot id."""

    shots: Dict[str, AssetRefsShot] = Field(default_factory=dict)
    extras: Dict[str, str] = Field(default_factory=dict)


class QAReportPerShot(BaseModel):
    shot_id: str
    prompt_match: str
    finger_issues: bool = False
    artifact_notes: List[str] = Field(default_factory=list)


class QAReport(BaseModel):
    movie_title: Optional[str] = None
    per_shot: List[QAReportPerShot] = Field(default_factory=list)
    summary: Optional[str] = None


class MoviePlan(BaseModel):
    """Top-level production plan the ScriptAgent emits.

    Required fields for an orchestrator run:
      - title
      - shots: each shot must have id, duration_sec, visual_description,
        start_frame_prompt and end_frame_prompt.
    """

    title: str
    characters: List[CharacterSpec] = Field(default_factory=list)
    shots: List[ShotSpec] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "examples": [
                {
                    "title": "Night Rooftop Confession",
                    "characters": [{"id": "c1", "name": "Ava"}],
                    "shots": [
                        {
                            "id": "shot_001",
                            "duration_sec": 8,
                            "visual_description": "Wide shot with neon signs",
                            "start_frame_prompt": "A neon-lit rooftop, rain...",
                            "end_frame_prompt": "Same rooftop, Ava closeup...",
                            "motion_prompt": "A slow dolly from wide to close",
                            "is_talking_closeup": True,
                            "dialogue": [{"character_id": "c1", "text": "I always loved you."}]
                        }
                    ],
                    "metadata": {"seed": "12345"}
                }
            ]
        }
