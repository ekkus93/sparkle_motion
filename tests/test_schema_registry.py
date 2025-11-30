from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from sparkle_motion import schema_registry


def test_list_schema_names_contains_expected_entries():
    names = list(schema_registry.list_schema_names())
    assert {
        "movie_plan",
        "asset_refs",
        "qa_report",
        "stage_event",
        "checkpoint",
        "run_context",
        "stage_manifest",
    }.issubset(names)


def test_get_schema_uri_and_path_for_movie_plan():
    uri = schema_registry.get_schema_uri("movie_plan")
    path = schema_registry.get_schema_path("movie_plan")
    assert uri == "artifact://sparkle-motion/schemas/movie_plan/v1"
    assert path.exists()
    assert path.name == "MoviePlan.schema.json"


def test_get_qa_policy_bundle_fields():
    bundle = schema_registry.get_qa_policy_bundle()
    assert bundle.uri == "artifact://sparkle-motion/qa_policy/v1"
    assert bundle.bundle_path.exists()
    assert bundle.bundle_path.name == "qa_policy_v1.tar.gz"
    assert bundle.manifest_path.exists()
    assert bundle.manifest_path.name == "manifest.json"


def test_typed_getters_return_expected_names():
    assert schema_registry.movie_plan_schema().name == "movie_plan"
    assert schema_registry.asset_refs_schema().name == "asset_refs"
    assert schema_registry.qa_report_schema().name == "qa_report"
    assert schema_registry.stage_event_schema().name == "stage_event"
    assert schema_registry.checkpoint_schema().name == "checkpoint"
    assert schema_registry.run_context_schema().name == "run_context"
    assert schema_registry.stage_manifest_schema().name == "stage_manifest"


def test_resolve_schema_uri_prefers_artifact_outside_fixture(monkeypatch):
    monkeypatch.delenv("ADK_USE_FIXTURE", raising=False)
    uri = schema_registry.resolve_schema_uri("movie_plan")
    assert uri == schema_registry.get_schema_uri("movie_plan")


def test_resolve_schema_uri_uses_local_in_fixture_mode(monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    with pytest.warns(RuntimeWarning) as recorded:
        uri = schema_registry.resolve_schema_uri("movie_plan")
    assert uri.startswith("file://")
    assert any("local schema fallback" in str(w.message) for w in recorded)


def test_resolve_schema_uri_missing_local_warns_and_returns_artifact(monkeypatch):
    catalog = schema_registry.load_catalog()
    broken = replace(catalog, schemas=dict(catalog.schemas))
    broken.schemas["movie_plan"] = replace(
        broken.schemas["movie_plan"], local_path=Path("/nonexistent/schema.json")
    )
    monkeypatch.setattr(schema_registry, "load_catalog", lambda config_path=None: broken)
    with pytest.warns(RuntimeWarning, match="does not exist"):
        uri = schema_registry.resolve_schema_uri("movie_plan", prefer_local=True)
    assert uri == broken.schemas["movie_plan"].uri


def test_resolve_qa_policy_bundle_prefers_local_in_fixture_mode(monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    with pytest.warns(RuntimeWarning) as recorded:
        bundle_uri, manifest_uri = schema_registry.resolve_qa_policy_bundle()
    assert bundle_uri.startswith("file://")
    assert manifest_uri.startswith("file://")
    assert any("qa_policy.bundle" in str(w.message) for w in recorded)