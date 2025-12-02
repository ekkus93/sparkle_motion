from __future__ import annotations

"""Test helpers for backend-aware artifact assertions."""

import os


def expected_artifact_scheme() -> str:
    return (os.getenv("ARTIFACTS_BACKEND") or "adk").strip().lower()


def assert_backend_artifact_uri(uri: str) -> None:
    assert uri, "artifact_uri must be non-empty"
    backend = expected_artifact_scheme()
    if backend == "filesystem":
        assert uri.startswith("artifact+fs://"), uri
    else:
        assert uri.startswith("file://"), uri
