from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkle_motion import adk_helpers
from sparkle_motion.filesystem_artifacts.config import FilesystemArtifactsConfig
from sparkle_motion.filesystem_artifacts.models import ArtifactManifest
from sparkle_motion.filesystem_artifacts.storage import FilesystemArtifactStore
from sparkle_motion.run_registry import (
    get_run_registry,
    _reset_filesystem_store_for_tests,
)


def _setup_filesystem_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FilesystemArtifactStore:
    root = tmp_path / "fs"
    index = root / "index.db"
    monkeypatch.setenv("ARTIFACTS_BACKEND", "filesystem")
    monkeypatch.setenv("ARTIFACTS_FS_ROOT", str(root))
    monkeypatch.setenv("ARTIFACTS_FS_INDEX", str(index))
    monkeypatch.setenv("ARTIFACTS_FS_ALLOW_INSECURE", "1")
    adk_helpers._reset_filesystem_store_for_tests()
    _reset_filesystem_store_for_tests()
    cfg = FilesystemArtifactsConfig.from_env()
    return FilesystemArtifactStore(cfg)


def _save_manifest(
    store: FilesystemArtifactStore,
    *,
    run_id: str,
    stage: str,
    artifact_type: str,
    payload: dict,
    metadata: dict,
) -> None:
    manifest = ArtifactManifest(
        run_id=run_id,
        stage=stage,
        artifact_type=artifact_type,
        mime_type="application/json",
        metadata=metadata,
    )
    store.save_artifact(
        manifest=manifest,
        payload=json.dumps(payload).encode("utf-8"),
        filename_hint=f"{artifact_type}.json",
    )


def test_get_status_filesystem_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _setup_filesystem_env(monkeypatch, tmp_path)
    run_id = "run_fs_status"
    plan_id = "plan-filesystem"
    run_context_payload = {
        "plan_id": plan_id,
        "plan_title": "Filesystem Plan",
        "render_profile": {"video": {"model_id": "wan-2.1"}},
        "metadata": {"foo": "bar"},
        "schema_uri": "artifact://sparkle-motion/schemas/run_context/v1",
    }
    _save_manifest(
        store,
        run_id=run_id,
        stage="plan_intake",
        artifact_type="plan_run_context",
        payload=run_context_payload,
        metadata={
            "plan_id": plan_id,
            "qa_mode": "skip",
            "qa_skipped": True,
        },
    )
    _save_manifest(
        store,
        run_id=run_id,
        stage="qa_publish",
        artifact_type="video_final",
        payload={"ok": True},
        metadata={"plan_id": plan_id},
    )
    registry = get_run_registry()
    registry.discard_run(run_id)

    status = registry.get_status(run_id)

    assert status["run_id"] == run_id
    assert status["plan_id"] == plan_id
    assert status["plan_title"] == "Filesystem Plan"
    assert status["status"] == "succeeded"
    assert status["qa_mode"] == "skip"
    assert status["qa_skipped"] is True
    assert status["artifact_counts"]["qa_publish"] == 1
    assert status["render_profile"]["video"]["model_id"] == "wan-2.1"
    assert status["metadata"]["foo"] == "bar"
    assert status["timeline"], "expected synthesized timeline entries"


def test_get_status_filesystem_missing_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _setup_filesystem_env(monkeypatch, tmp_path)
    registry = get_run_registry()
    registry.discard_run("missing-run")

    with pytest.raises(KeyError):
        registry.get_status("missing-run")
