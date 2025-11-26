"""Artifact service shim implementations for local testing.

Exports `FileArtifactService` and `GcsArtifactService` that implement a
minimal async `save_artifact` method returning a numeric revision id.
"""

from .file_artifact_service import FileArtifactService
from .gcs_artifact_service import GcsArtifactService

__all__ = ["FileArtifactService", "GcsArtifactService"]
