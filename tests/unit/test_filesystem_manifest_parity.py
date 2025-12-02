from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

from sparkle_motion.filesystem_artifacts.config import FilesystemArtifactsConfig
from sparkle_motion.filesystem_artifacts.models import ArtifactManifest
from sparkle_motion.filesystem_artifacts.storage import FilesystemArtifactStore


@dataclass(frozen=True)
class ManifestScenario:
    id: str
    reference_key: str
    artifact_slug: str
    run_id: str
    stage: str
    artifact_type: str
    mime_type: str
    payload: bytes
    filename_hint: str
    metadata: dict[str, Any]
    qa: dict[str, Any] | None
    tags: dict[str, Any] | None
    local_path_hint: str


_SCENARIOS = (
    ManifestScenario(
        id="tts_timeline",
        reference_key="tts_reference",
        artifact_slug="11111111111111111111111111111111",
        run_id="run_demo",
        stage="dialogue_audio",
        artifact_type="tts_timeline",
        mime_type="audio/wav",
        payload=b"fixture-manifest-payload",
        filename_hint="timeline.wav",
        metadata={
            "schema_uri": "artifact://sparkle-motion/schemas/tts_timeline/v1",
            "size_bytes": 24,
            "checksum": {
                "sha256": "3afaac217bfd89eff3f4623dbce7d6434b4a11eb93a6c5550e0bc9eb64f67475",
            },
            "qa_report_uri": "artifact://sparkle-motion/qa_reports/run_demo/dialogue_audio",
        },
        qa={
            "decision": "approve",
            "report_uri": "artifact://sparkle-motion/qa_reports/run_demo/dialogue_audio",
            "issues": [],
        },
        tags={"qa_mode": "full"},
        local_path_hint="/tmp/original_tts_timeline.wav",
    ),
    ManifestScenario(
        id="qa_report",
        reference_key="qa_reference",
        artifact_slug="22222222222222222222222222222222",
        run_id="run_demo_qa",
        stage="qa_publish",
        artifact_type="qa_report",
        mime_type="application/json",
        payload=b"qa-report-payload",
        filename_hint="qa_report.json",
        metadata={
            "schema_uri": "artifact://sparkle-motion/schemas/qa_report/v1",
            "size_bytes": 17,
            "checksum": {
                "sha256": "a81cdbeb446ca71026f84735a0e834107721392786350bfa7996d3993d0ef855",
            },
            "qa_report_uri": "artifact://sparkle-motion/qa_reports/run_demo_qa/qa_publish",
            "qa_summary": {
                "issues": 0,
                "warnings": 0,
                "notes": ["fixture"],
            },
        },
        qa={
            "decision": "approve",
            "report_uri": "artifact://sparkle-motion/qa_reports/run_demo_qa/qa_publish",
            "issues": [],
        },
        tags={"qa_mode": "skip"},
        local_path_hint="/tmp/qa_report.json",
    ),
)


@pytest.fixture
def fixture_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "filesystem"
    return {
        "tts_reference": root / "adk_manifest_reference.json",
        "qa_reference": root / "qa_report_manifest_reference.json",
    }


@pytest.fixture(params=_SCENARIOS, ids=lambda scenario: scenario.id)
def manifest_pair(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_paths: dict[str, Path],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    scenario: ManifestScenario = request.param

    class _FixedUUID:
        def __init__(self, hex_value: str) -> None:
            self.hex = hex_value

    monkeypatch.setattr(
        "sparkle_motion.filesystem_artifacts.storage.uuid.uuid4",
        lambda: _FixedUUID(scenario.artifact_slug),
    )
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
        run_id=scenario.run_id,
        stage=scenario.stage,
        artifact_type=scenario.artifact_type,
        mime_type=scenario.mime_type,
        metadata=json.loads(json.dumps(scenario.metadata)),
        qa=json.loads(json.dumps(scenario.qa)) if scenario.qa is not None else None,
        tags=json.loads(json.dumps(scenario.tags)) if scenario.tags is not None else None,
        local_path_hint=scenario.local_path_hint,
    )

    record = store.save_artifact(
        manifest=manifest,
        payload=scenario.payload,
        filename_hint=scenario.filename_hint,
    )

    manifest_path = Path(record.storage.manifest_path)
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(fixture_paths[scenario.reference_key].read_text(encoding="utf-8"))

    reference["local_path"] = actual["local_path"]
    reference["download_url"] = actual["download_url"]

    return scenario.id, actual, reference


def test_filesystem_manifest_matches_adk_reference(
    manifest_pair: tuple[str, dict[str, Any], dict[str, Any]]
) -> None:
    _, actual, reference = manifest_pair
    assert actual == reference


def test_filesystem_manifest_reports_no_diffs_for_critical_fields(
    manifest_pair: tuple[str, dict[str, Any], dict[str, Any]],
) -> None:
    scenario_id, actual, reference = manifest_pair
    critical_paths = (
        "size_bytes",
        "checksum.sha256",
        "metadata.schema_uri",
        "metadata.size_bytes",
        "metadata.checksum.sha256",
        "metadata.qa_report_uri",
        "qa.decision",
        "qa.report_uri",
        "qa.issues",
    )
    diffs = _diff_manifest_fields(actual, reference, critical_paths)
    assert diffs == [], f"{scenario_id}: Unexpected manifest differences: {diffs}"


def _diff_manifest_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    paths: Iterable[str],
) -> list[str]:
    differences: list[str] = []
    for path in paths:
        actual_value, actual_error = _lookup_path(actual, path)
        expected_value, expected_error = _lookup_path(expected, path)
        if actual_error or expected_error:
            differences.append(
                f"{path}: missing value (actual_error={actual_error!r}, expected_error={expected_error!r})"
            )
            continue
        if actual_value != expected_value:
            differences.append(f"{path}: expected {expected_value!r}, got {actual_value!r}")
    return differences


def _lookup_path(data: Mapping[str, Any], path: str) -> tuple[Any | None, str | None]:
    node: Any = data
    for segment in path.split("."):
        if isinstance(node, Mapping) and segment in node:
            node = node[segment]
            continue
        return None, f"missing segment '{segment}'"
    return node, None
