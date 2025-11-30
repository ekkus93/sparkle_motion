from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


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
    prompt_match: float = Field(..., ge=0.0, le=1.0)
    finger_issues: bool = False
    finger_issue_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    artifact_notes: List[str] = Field(default_factory=list)
    missing_audio_detected: bool = False
    safety_violation: bool = False
    audio_peak_db: Optional[float] = None


class QAReport(BaseModel):
    movie_title: Optional[str] = None
    per_shot: List[QAReportPerShot] = Field(default_factory=list)
    summary: Optional[str] = None
    decision: Optional[Literal["approve", "regenerate", "escalate", "pending"]] = None
    issues_found: Optional[int] = None
    aggregate_prompt_match: Optional[float] = Field(None, ge=0.0, le=1.0)


class StageEvent(BaseModel):
    """Single manifest entry emitted for each stage attempt."""

    run_id: str
    stage: str
    status: Literal["begin", "success", "fail"]
    timestamp: float
    attempt: int = Field(ge=1)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    """Per-stage checkpoint persisted to support resume and retries."""

    stage: str
    status: Literal["begin", "success", "failed"]
    timestamp: float
    attempt: int = Field(ge=1)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


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

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class RunContext(BaseModel):
    """Materialized production context derived from a validated MoviePlan."""

    run_id: str
    plan_id: str
    plan_title: str
    plan: MoviePlan
    schema_uri: Optional[str] = None
    shot_order: List[str] = Field(default_factory=list)
    dialogue_timeline_uri: Optional[str] = None
    base_image_map: Dict[str, str] = Field(default_factory=dict)
    render_profile: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_plan(
        cls,
        plan: MoviePlan,
        *,
        run_id: str,
        schema_uri: Optional[str] = None,
        render_profile: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RunContext":
        """Construct a RunContext using canonical identifiers from the plan."""

        plan_id = (plan.metadata or {}).get("plan_id") or plan.title.strip().lower().replace(" ", "-")[:64] or "movie-plan"
        shot_order = [shot.id for shot in plan.shots]
        base_meta = dict(plan.metadata or {})
        ctx_meta = dict(metadata or {})
        if base_meta:
            ctx_meta.setdefault("plan_metadata", base_meta)
        return cls(
            run_id=run_id,
            plan_id=plan_id,
            plan_title=plan.title,
            plan=plan,
            schema_uri=schema_uri,
            shot_order=shot_order,
            render_profile=dict(render_profile or {}),
            metadata=ctx_meta,
        )


class StageManifest(BaseModel):
    """Structured artifact manifest entry served via production_agent /artifacts."""

    run_id: str
    stage_id: str
    artifact_type: str
    name: str
    artifact_uri: str
    media_type: Optional[str] = None
    local_path: Optional[str] = None
    download_url: Optional[str] = None
    storage_hint: Optional[Literal["adk", "local"]] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)
    duration_s: Optional[float] = Field(default=None, ge=0)
    frame_rate: Optional[float] = Field(default=None, ge=0)
    resolution_px: Optional[str] = None
    checksum_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    qa_report_uri: Optional[str] = None
    qa_passed: Optional[bool] = None
    qa_mode: Optional[str] = None
    playback_ready: Optional[bool] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

