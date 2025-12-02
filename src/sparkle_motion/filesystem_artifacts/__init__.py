from __future__ import annotations

"""Filesystem ArtifactService shim public interface."""

from .app import create_app
from .config import FilesystemArtifactsConfig

__all__ = ["create_app", "FilesystemArtifactsConfig"]
