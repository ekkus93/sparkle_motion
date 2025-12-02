from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from sparkle_motion import adk_helpers


@pytest.fixture(autouse=True)
def _fixture_mode(monkeypatch):
	monkeypatch.setenv("ADK_USE_FIXTURE", "1")
	monkeypatch.delenv("SPARKLE_LOCAL_RUNS_ROOT", raising=False)
	service = adk_helpers.get_memory_service()
	if hasattr(service, "clear_memory_events"):
		service.clear_memory_events()
	yield
	if hasattr(service, "clear_memory_events"):
		service.clear_memory_events()


def _uri_to_path(uri: str) -> Path:
	parsed = urlparse(uri)
	return Path(parsed.path)


def _filesystem_backend_enabled() -> bool:
	return (os.getenv("ARTIFACTS_BACKEND") or "adk").strip().lower() == "filesystem"


def test_publish_artifact_records_fixture_event(monkeypatch, tmp_path: Path):
	monkeypatch.setenv("SPARKLE_RUN_ID", "run-publish-test")
	src = tmp_path / "plan.json"
	src.write_text("{}", encoding="utf-8")

	ref = adk_helpers.publish_artifact(local_path=src, artifact_type="movie_plan", metadata={"priority": "high"})

	expected_storage = "filesystem" if _filesystem_backend_enabled() else "local"
	assert ref["storage"] == expected_storage
	assert ref["metadata"]["artifact_type"] == "movie_plan"
	service = adk_helpers.get_memory_service()
	events = service.list_memory_events("run-publish-test")
	assert events
	assert events[-1]["event_type"] == "adk_helpers.publish_artifact"
	assert events[-1]["payload"]["storage"] == expected_storage


def test_publish_local_persists_payload(monkeypatch, tmp_path: Path):
	monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path))
	ref = adk_helpers.publish_local(payload=b"hello", artifact_type="movie_plan")

	if _filesystem_backend_enabled():
		path = Path(ref["metadata"]["local_path"])
	else:
		path = _uri_to_path(ref["uri"])
	assert path.exists()
	assert path.read_bytes() == b"hello"


def test_publish_local_filesystem_backend(monkeypatch, tmp_path: Path):
	monkeypatch.setenv("ARTIFACTS_BACKEND", "filesystem")
	monkeypatch.setenv("ARTIFACTS_FS_ROOT", str(tmp_path / "fs"))
	monkeypatch.setenv("ARTIFACTS_FS_INDEX", str(tmp_path / "index.db"))
	monkeypatch.setenv("ARTIFACTS_FS_TOKEN", "secret-token")
	adk_helpers._reset_filesystem_store_for_tests()
	ref = adk_helpers.publish_local(payload=b"hello", artifact_type="movie_plan", suffix=".txt")

	assert ref["storage"] == "filesystem"
	assert ref["metadata"]["storage_backend"] == "filesystem"
	assert ref["metadata"]["artifact_type"] == "movie_plan"
	path = Path(ref["metadata"]["local_path"])
	assert path.exists()
	assert path.read_text(encoding="utf-8") == "hello"
	manifest = Path(ref["metadata"]["manifest_path"])
	assert manifest.exists()
	assert "artifact_uri" in json.loads(manifest.read_text(encoding="utf-8"))


def test_request_human_input_fixture_records_event(monkeypatch):
	monkeypatch.setenv("SPARKLE_RUN_ID", "run-human")
	task_id = adk_helpers.request_human_input(run_id=None, reason="need review", artifact_uri="file://artifact")

	assert task_id.startswith("fixture-review-")
	events = adk_helpers.get_memory_service().list_memory_events("run-human")
	assert any(evt["event_type"] == "adk_helpers.request_human_input" for evt in events)


def test_set_backend_overrides_publish(tmp_path: Path):
	src = tmp_path / "artifact.json"
	src.write_text("{}", encoding="utf-8")

	override_ref = {
		"uri": "override://artifact/1",
		"storage": "adk",
		"artifact_type": "movie_plan",
		"media_type": "application/json",
		"metadata": {"artifact_type": "movie_plan"},
		"run_id": "run-override",
	}

	backend = adk_helpers.HelperBackend(publish=lambda **_: override_ref)

	with adk_helpers.set_backend(backend):
		ref = adk_helpers.publish_artifact(local_path=src, artifact_type="movie_plan")

	assert ref["uri"] == override_ref["uri"]
	assert ref["storage"] == override_ref["storage"]
	assert ref["artifact_type"] == override_ref["artifact_type"]


def test_ensure_schema_artifacts_loads_catalog():
	catalog = adk_helpers.ensure_schema_artifacts()
	assert catalog.schemas