from __future__ import annotations

from pathlib import Path

from sparkle_motion import schema_registry


def test_list_schema_names_contains_expected_entries():
    names = list(schema_registry.list_schema_names())
    assert {"movie_plan", "asset_refs", "qa_report", "stage_event", "checkpoint"}.issubset(names)


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