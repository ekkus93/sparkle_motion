from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal, Annotated, Union, Mapping
from pydantic import BaseModel, Field, ConfigDict, model_validator


_TIMELINE_TOLERANCE = 1e-3
_ORDERING_TOLERANCE = 1e-6


class CharacterSpec(BaseModel):
    """Description of a character used in the movie."""

    id: str
    name: str
    description: Optional[str] = None
    voice_profile: Dict[str, str] = Field(default_factory=dict)


class DialogueLine(BaseModel):
    """Single dialogue entry attached to a shot's script."""

    character_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShotSpec(BaseModel):
    """A single shot specification used by the pipeline."""

    id: str
    duration_sec: float = Field(..., gt=0)
    setting: Optional[str] = None
    visual_description: str
    start_base_image_id: str = Field(..., min_length=1)
    end_base_image_id: str = Field(..., min_length=1)
    motion_prompt: Optional[str] = None
    is_talking_closeup: bool = False
    dialogue: List[DialogueLine] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_base_image_refs(self) -> "ShotSpec":
        start = self.start_base_image_id.strip()
        end = self.end_base_image_id.strip()
        if not start or not end:
            raise ValueError("start_base_image_id and end_base_image_id must be non-empty")
        if start == end:
            raise ValueError("start_base_image_id and end_base_image_id must differ")
        self.start_base_image_id = start
        self.end_base_image_id = end
        return self


class BaseImageSpec(BaseModel):
    """Keyframe specification shared across all shots."""

    id: str
    prompt: str
    asset_uri: Optional[str] = None
    local_path: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class _DialogueTimelineBase(BaseModel):
    start_time_sec: float = Field(..., ge=0.0)
    duration_sec: float = Field(..., gt=0.0)


class DialogueTimelineDialogue(_DialogueTimelineBase):
    type: Literal["dialogue"] = "dialogue"
    character_id: str
    text: str


class DialogueTimelineSilence(_DialogueTimelineBase):
    type: Literal["silence"] = "silence"


DialogueTimelineEntry = Annotated[
    Union[DialogueTimelineDialogue, DialogueTimelineSilence],
    Field(discriminator="type"),
]


class RenderProfileVideo(BaseModel):
    model_id: str
    max_fps: Optional[float] = Field(default=None, gt=0.0)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_model_id(self) -> "RenderProfileVideo":
        if not self.model_id or not self.model_id.strip():
            raise ValueError("render_profile.video.model_id must be a non-empty string")
        self.model_id = self.model_id.strip()
        return self


class RenderProfile(BaseModel):
    video: RenderProfileVideo
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
      - base_images: ordered inventory of keyframes with len == len(shots) + 1
      - shots: each shot must reference consecutive base image ids via
        `start_base_image_id` / `end_base_image_id`.
    """
    title: str
    characters: List[CharacterSpec] = Field(default_factory=list)
    base_images: List[BaseImageSpec] = Field(..., min_length=1)
    shots: List[ShotSpec] = Field(..., min_length=1)
    dialogue_timeline: List[DialogueTimelineEntry] = Field(..., min_length=1)
    render_profile: RenderProfile
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "Night Rooftop Confession",
                    "characters": [{"id": "c1", "name": "Ava"}],
                    "base_images": [
                        {"id": "frame_000", "prompt": "Rooftop establishing"},
                        {"id": "frame_001", "prompt": "Close on Ava"},
                    ],
                    "shots": [
                        {
                            "id": "shot_001",
                            "duration_sec": 8,
                            "visual_description": "Wide shot with neon signs",
                            "start_base_image_id": "frame_000",
                            "end_base_image_id": "frame_001",
                            "motion_prompt": "A slow dolly from wide to close",
                            "is_talking_closeup": True,
                        }
                    ],
                    "dialogue_timeline": [
                        {
                            "type": "dialogue",
                            "character_id": "c1",
                            "text": "I always loved you.",
                            "start_time_sec": 0.0,
                            "duration_sec": 4.0,
                        }
                    ],
                    "render_profile": {
                        "video": {"model_id": "wan-2.1"},
                        "metadata": {},
                    },
                    "metadata": {"seed": "12345"}
                }
            ]
        }
    )

    @model_validator(mode="after")
    def _validate_relationships(self) -> "MoviePlan":
        if not self.shots:
            raise ValueError("MoviePlan must contain at least one shot")
        expected_base_images = len(self.shots) + 1
        if len(self.base_images) != expected_base_images:
            raise ValueError(
                f"base_images count mismatch: expected len(shots)+1 == {expected_base_images} "
                f"but got {len(self.base_images)}"
            )

        seen_base_ids: set[str] = set()
        for base_image in self.base_images:
            base_id = base_image.id.strip() if base_image.id else ""
            if not base_id:
                raise ValueError("Base image ids must be non-empty")
            if base_id in seen_base_ids:
                raise ValueError(f"Duplicate base image id detected: {base_id}")
            seen_base_ids.add(base_id)
            prompt = base_image.prompt.strip() if base_image.prompt else ""
            if not prompt:
                raise ValueError(f"Base image {base_id} must include a non-empty prompt")

        id_order = [base_image.id for base_image in self.base_images]
        id_to_index = {base_image.id: idx for idx, base_image in enumerate(self.base_images)}
        for index, shot in enumerate(self.shots):
            start_idx = id_to_index.get(shot.start_base_image_id)
            end_idx = id_to_index.get(shot.end_base_image_id)
            if start_idx is None:
                raise ValueError(f"Shot {shot.id} references missing start_base_image_id {shot.start_base_image_id}")
            if end_idx is None:
                raise ValueError(f"Shot {shot.id} references missing end_base_image_id {shot.end_base_image_id}")
            if start_idx != index:
                raise ValueError(
                    f"Shot {shot.id} must start at base_images[{index}] (got {id_order[start_idx]})"
                )
            if end_idx != index + 1:
                raise ValueError(
                    f"Shot {shot.id} must end at base_images[{index + 1}] (got {id_order[end_idx]})"
                )

        total_runtime = sum(shot.duration_sec for shot in self.shots)
        if total_runtime <= 0:
            raise ValueError("MoviePlan total duration must be positive")

        if not self.dialogue_timeline:
            raise ValueError("dialogue_timeline must describe the full runtime, even if silent")

        timeline = list(self.dialogue_timeline)
        if timeline[0].start_time_sec > _TIMELINE_TOLERANCE:
            raise ValueError("dialogue_timeline must start at time 0; prepend a silence entry if needed")

        last_end = 0.0
        for entry in timeline:
            if entry.start_time_sec - last_end > _TIMELINE_TOLERANCE:
                raise ValueError("dialogue_timeline contains gaps; insert silence entries to cover idle spans")
            if entry.start_time_sec + _ORDERING_TOLERANCE < last_end:
                raise ValueError("dialogue_timeline entries must be ordered by start_time_sec")
            entry_end = entry.start_time_sec + entry.duration_sec
            if entry_end > total_runtime + _TIMELINE_TOLERANCE:
                raise ValueError("dialogue_timeline exceeds total shot duration")
            last_end = max(last_end, entry_end)

        if total_runtime - last_end > _TIMELINE_TOLERANCE:
            raise ValueError(
                f"dialogue_timeline ends at {last_end:.3f}s but shots run for {total_runtime:.3f}s"
            )

        if last_end - total_runtime > _TIMELINE_TOLERANCE:
            raise ValueError(
                f"dialogue_timeline exceeds total shot duration by {(last_end - total_runtime):.3f}s"
            )

        return self


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
        dialogue_timeline_uri: Optional[str] = None,
        base_image_map: Optional[Mapping[str, str]] = None,
    ) -> "RunContext":
        """Construct a RunContext using canonical identifiers from the plan."""

        plan_id = (plan.metadata or {}).get("plan_id") or plan.title.strip().lower().replace(" ", "-")[:64] or "movie-plan"
        shot_order = [shot.id for shot in plan.shots]
        base_meta = dict(plan.metadata or {})
        ctx_meta = dict(metadata or {})
        if base_meta:
            ctx_meta.setdefault("plan_metadata", base_meta)
        if render_profile is not None:
            render_profile_payload = dict(render_profile)
        else:
            if hasattr(plan.render_profile, "model_dump"):
                render_profile_payload = plan.render_profile.model_dump()
            else:  # pragma: no cover - legacy pydantic v1 fallback
                render_profile_payload = plan.render_profile.dict()  # type: ignore[attr-defined]

        return cls(
            run_id=run_id,
            plan_id=plan_id,
            plan_title=plan.title,
            plan=plan,
            schema_uri=schema_uri,
            shot_order=shot_order,
            dialogue_timeline_uri=dialogue_timeline_uri,
            base_image_map=dict(base_image_map or {}),
            render_profile=render_profile_payload,
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
    storage_hint: Optional[Literal["adk", "local", "filesystem"]] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)
    duration_s: Optional[float] = Field(default=None, ge=0)
    frame_rate: Optional[float] = Field(default=None, ge=0)
    resolution_px: Optional[str] = None
    checksum_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    playback_ready: Optional[bool] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

