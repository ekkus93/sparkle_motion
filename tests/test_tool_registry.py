from __future__ import annotations

import pytest

from sparkle_motion import schema_registry
from sparkle_motion.tool_registry import resolve_schema_references, SchemaResolutionError


def test_resolve_schema_references_uses_registry(monkeypatch):
    monkeypatch.delenv("ADK_USE_FIXTURE", raising=False)
    resolved = resolve_schema_references({"output": {"registry_name": "movie_plan"}})
    assert resolved["output"] == schema_registry.movie_plan_schema().uri


def test_resolve_schema_references_prefers_local_when_fixture(monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    with pytest.warns(RuntimeWarning):
        resolved = resolve_schema_references({"output": {"registry_name": "movie_plan"}})
    assert resolved["output"].startswith("file://")


def test_resolve_schema_references_rejects_invalid_entries():
    with pytest.raises(SchemaResolutionError):
        resolve_schema_references({"output": {}})
