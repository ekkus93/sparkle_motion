from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkle_motion.filesystem_artifacts.config import FilesystemArtifactsConfig
from sparkle_motion.filesystem_artifacts.models import ArtifactManifest
from sparkle_motion.filesystem_artifacts.storage import FilesystemArtifactStore


@pytest.fixture
def fixture_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "filesystem"
    return {
        "reference_manifest": root / "adk_manifest_reference.json",
    }


def test_filesystem_manifest_matches_adk_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_paths: dict[str, Path],
) -> None:
    payload = b"fixture-manifest-payload"
    checksum = "3afaac217bfd89eff3f4623dbce7d6434b4a11eb93a6c5550e0bc9eb64f67475"
    fake_uuid = "11111111111111111111111111111111"

    class _FakeUUID:
        hex = fake_uuid

    monkeypatch.setattr("sparkle_motion.filesystem_artifacts.storage.uuid.uuid4", lambda: _FakeUUID())
    monkeypatch.setattr("sparkle_motion.filesystem_artifacts.storage.time.time", lambda: 1_700_000_000)

    config = FilesystemArtifactsConfig(
        root=tmp_path / "fs_root",
        index_path=tmp_path / "fs_index.db",
        base_url="http://127.0.0.1:7077",
        token=None,
        allow_insecure=True,
        max_payload_bytes=1024,
    )
    store = FilesystemArtifactStore(config)

    manifest = ArtifactManifest(
        run_id="run_demo",
        stage="dialogue_audio",
        artifact_type="tts_timeline",
        mime_type="audio/wav",
        metadata={
            "schema_uri": "artifact://sparkle-motion/schemas/tts_timeline/v1",
            "size_bytes": len(payload),
            "checksum": {"sha256": checksum},
            "qa_report_uri": "artifact://sparkle-motion/qa_reports/run_demo/dialogue_audio",
        },
        qa={
            "decision": "approve",
            "report_uri": "artifact://sparkle-motion/qa_reports/run_demo/dialogue_audio",
            "issues": [],
        },
        tags={"qa_mode": "full"},
        local_path_hint="/tmp/original_tts_timeline.wav",
    )

    record = store.save_artifact(manifest=manifest, payload=payload, filename_hint="timeline.wav")

    manifest_path = Path(record.storage.manifest_path)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(fixture_paths["reference_manifest"].read_text(encoding="utf-8"))

    # Local paths depend on tmp directories; align reference placeholders with the generated values.
    reference["local_path"] = actual["local_path"]
    reference["download_url"] = actual["download_url"]

    assert actual == reference
    assert record.manifest == actual
