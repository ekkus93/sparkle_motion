from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkle_motion import adk_helpers
from sparkle_motion.filesystem_artifacts.config import FilesystemArtifactsConfig
from sparkle_motion.filesystem_artifacts.models import ArtifactManifest
from sparkle_motion.filesystem_artifacts.storage import FilesystemArtifactStore
from sparkle_motion.run_registry import (
    ArtifactEntry,
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
        stage="finalize",
        artifact_type="video_final",
        payload={"ok": True},
        metadata={"plan_id": plan_id, "qa_mode": "skip", "qa_skipped": True},
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
    assert status["artifact_counts"]["finalize"] == 1
    assert status["render_profile"]["video"]["model_id"] == "wan-2.1"
    assert status["metadata"]["foo"] == "bar"
    assert status["timeline"], "expected synthesized timeline entries"


def test_get_status_filesystem_missing_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _setup_filesystem_env(monkeypatch, tmp_path)
    registry = get_run_registry()
    registry.discard_run("missing-run")

    with pytest.raises(KeyError):
        registry.get_status("missing-run")


def test_list_artifacts_filesystem_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _setup_filesystem_env(monkeypatch, tmp_path)
    run_id = "run_fs_artifacts"
    stage = "finalize"
    artifact_type = "video_final"
    artifact_uri = f"artifact+fs://{run_id}/{stage}/{artifact_type}/demo"
    plan_id = "plan-filesystem-artifacts"
    manifest_snapshot = {
        "run_id": run_id,
        "stage_id": stage,
        "artifact_type": artifact_type,
        "name": "final.mp4",
        "artifact_uri": artifact_uri,
        "media_type": "video/mp4",
        "local_path": "/tmp/final.mp4",
        "download_url": None,
        "storage_hint": "filesystem",
        "mime_type": "video/mp4",
        "size_bytes": 1024,
        "duration_s": 9.5,
        "frame_rate": 24.0,
        "resolution_px": "1280x720",
        "checksum_sha256": "f" * 64,
        "qa_report_uri": None,
        "qa_mode": "full",
        "qa_skipped": False,
        "playback_ready": True,
        "notes": "filesystem fixture",
        "metadata": {"plan_id": plan_id},
        "created_at": "2025-12-02T00:00:00Z",
    }
    _save_manifest(
        store,
        run_id=run_id,
        stage=stage,
        artifact_type=artifact_type,
        payload={"ok": True},
        metadata={
            "plan_id": plan_id,
            "qa_skipped": False,
            "stage_manifest_snapshot": manifest_snapshot,
        },
    )
    registry = get_run_registry()
    registry.discard_run(run_id)

    manifests = registry.list_artifacts(run_id)
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["run_id"] == run_id
    assert manifest["stage_id"] == stage
    assert manifest["artifact_type"] == artifact_type
    assert manifest["storage_hint"] == "filesystem"
    assert manifest["metadata"]["plan_id"] == plan_id
    assert manifest["artifact_uri"].startswith(f"artifact+fs://{run_id}/{stage}/{artifact_type}/")
    assert manifest["mime_type"] == "video/mp4"

    filtered_manifests = registry.list_artifacts(run_id, stage=stage)
    assert filtered_manifests == manifests

    assert registry.list_artifacts(run_id, stage="unknown-stage") == []


def test_list_artifacts_preserves_adk_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = get_run_registry()
    run_id = "run_adk_registry"
    plan_id = "plan-adk"
    registry.discard_run(run_id)
    registry.start_run(run_id=run_id, plan_id=plan_id, plan_title="ADK Plan", mode="run")

    adk_entry = ArtifactEntry(
        stage="plan_intake",
        artifact_type="movie_plan",
        name="plan.json",
        artifact_uri="artifact://sparkle-motion/plans/plan-adk",
        storage_hint="adk",
        metadata={"plan_id": plan_id},
    )
    registry.record_artifact(run_id, adk_entry)

    manifests = registry.list_artifacts(run_id)
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["artifact_uri"].startswith("artifact://sparkle-motion/plans/plan-adk")
    assert manifest["stage_id"] == "plan_intake"
    assert manifest["artifact_type"] == "movie_plan"
    assert manifest["metadata"]["plan_id"] == plan_id
    stage_filtered = registry.list_artifacts(run_id, stage="plan_intake")
    assert stage_filtered == manifests
    registry.discard_run(run_id)


def test_list_artifacts_merges_adk_and_filesystem(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _setup_filesystem_env(monkeypatch, tmp_path)
    registry = get_run_registry()
    run_id = "run_mixed_backends"
    plan_id = "plan-mixed"
    registry.discard_run(run_id)
    registry.start_run(run_id=run_id, plan_id=plan_id, plan_title="Mixed Plan", mode="run")

    adk_entry = ArtifactEntry(
        stage="plan_intake",
        artifact_type="movie_plan",
        name="plan.json",
        artifact_uri="artifact://sparkle-motion/plans/plan-mixed",
        storage_hint="adk",
        metadata={"plan_id": plan_id},
    )
    registry.record_artifact(run_id, adk_entry)

    _save_manifest(
        store,
        run_id=run_id,
        stage="finalize",
        artifact_type="video_final",
        payload={"ok": True},
        metadata={
            "plan_id": plan_id,
            "stage_manifest_snapshot": {
                "run_id": run_id,
                "stage_id": "finalize",
                "artifact_type": "video_final",
                "name": "final.mp4",
                "artifact_uri": f"artifact+fs://{run_id}/finalize/video_final/fixture",
                "storage_hint": "filesystem",
            },
        },
    )

    manifests = registry.list_artifacts(run_id)
    assert len(manifests) >= 2
    assert any(item["artifact_uri"].startswith("artifact://") for item in manifests)
    assert any(item["artifact_uri"].startswith("artifact+fs://") for item in manifests)
    assert {item["stage_id"] for item in manifests} == {"plan_intake", "finalize"}
    registry.discard_run(run_id)
