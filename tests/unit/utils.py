"""Test helpers for backend-aware artifact assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from sparkle_motion import adk_helpers
from sparkle_motion.utils.env import resolve_artifacts_backend

_MANAGED_URI_PREFIXES = ("file://", "artifact://", "artifact+fs://")


def expected_artifact_scheme() -> str:
    return resolve_artifacts_backend()


def assert_managed_artifact_uri(uri: str) -> None:
    assert isinstance(uri, str) and uri, "artifact_uri must be a non-empty string"
    assert any(uri.startswith(prefix) for prefix in _MANAGED_URI_PREFIXES), uri


def assert_backend_artifact_uri(uri: str) -> None:
    assert_managed_artifact_uri(uri)
    backend = expected_artifact_scheme()
    if backend == "filesystem":
        assert adk_helpers.is_filesystem_artifact_uri(uri), uri
    elif adk_helpers.is_artifact_uri(uri):
        return
    else:
        assert uri.startswith("file://"), uri


def artifact_local_path(uri: str, metadata: Mapping[str, Any] | None = None) -> Path | None:
    """Best-effort resolver for filesystem paths referenced by artifact URIs."""

    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    if adk_helpers.is_filesystem_artifact_uri(uri):
        meta = metadata or {}
        for key in ("local_path", "source_path"):
            candidate = meta.get(key)
            if isinstance(candidate, str) and candidate:
                return Path(candidate)
    return None
