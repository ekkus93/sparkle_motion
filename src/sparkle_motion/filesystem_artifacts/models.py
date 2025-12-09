"""Pydantic models shared by the filesystem ArtifactService shim."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class ArtifactManifest(BaseModel):
    """Caller-provided manifest fields for artifact uploads."""

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(pattern=_RUN_ID_PATTERN.pattern)
    stage: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: Optional[dict[str, Any]] = None
    local_path_hint: Optional[str] = None


class ArtifactJsonUpload(BaseModel):
    """JSON upload payload helper for application/json requests."""

    manifest: ArtifactManifest
    payload_b64: Optional[str] = None
    filename_hint: Optional[str] = None


class StoragePaths(BaseModel):
    """Relative and absolute filesystem locations for an artifact."""

    backend: Literal["filesystem"] = "filesystem"
    relative_path: str
    absolute_path: str
    manifest_path: str


class ArtifactRecord(BaseModel):
    """Canonical response payload for artifact metadata APIs."""

    artifact_id: str
    artifact_uri: str
    run_id: str
    stage: str
    artifact_type: str
    mime_type: str
    metadata: dict[str, Any]
    tags: Optional[dict[str, Any]] = None
    created_at: int
    manifest: dict[str, Any]
    storage: StoragePaths


class ArtifactListResponse(BaseModel):
    """Listing response for /artifacts."""

    items: list[ArtifactRecord]
    next_page_token: Optional[str] = None
