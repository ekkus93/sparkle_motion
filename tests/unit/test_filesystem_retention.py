from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from sparkle_motion.filesystem_artifacts.cli import main as filesystem_cli_main
from sparkle_motion.filesystem_artifacts.config import FilesystemArtifactsConfig
from sparkle_motion.filesystem_artifacts.models import ArtifactManifest
from sparkle_motion.filesystem_artifacts.retention import (
    RetentionOptions,
    execute_plan,
    load_artifacts,
    plan_retention,
)
from sparkle_motion.filesystem_artifacts.storage import FilesystemArtifactStore


@pytest.fixture()
def fs_config(tmp_path: Path) -> FilesystemArtifactsConfig:
    root = tmp_path / "fs_root"
    index = tmp_path / "index.db"
    return FilesystemArtifactsConfig(
        root=root,
        index_path=index,
        base_url="http://127.0.0.1:7077",
        token="local-token",
        allow_insecure=True,
        max_payload_bytes=1024 * 1024,
    )


@pytest.fixture()
def store(fs_config: FilesystemArtifactsConfig) -> FilesystemArtifactStore:
    return FilesystemArtifactStore(fs_config)


def _save_artifact(
    store: FilesystemArtifactStore,
    *,
    run_id: str,
    stage: str,
    artifact_type: str,
    size: int,
    timestamp: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sparkle_motion.filesystem_artifacts.storage.time.time", lambda: timestamp)
    manifest = ArtifactManifest(
        run_id=run_id,
        stage=stage,
        artifact_type=artifact_type,
        mime_type="application/octet-stream",
        metadata={"schema_uri": "schemas/sample.json"},
    )
    store.save_artifact(manifest=manifest, payload=b"x" * size, filename_hint="artifact.bin")


def test_plan_retention_applies_age_size_and_free_constraints(
    store: FilesystemArtifactStore,
    fs_config: FilesystemArtifactsConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_700_000_000
    _save_artifact(store, run_id="run-old", stage="plan", artifact_type="movie_plan", size=10, timestamp=now - 20 * 86400, monkeypatch=monkeypatch)
    _save_artifact(store, run_id="run-mid", stage="dialogue", artifact_type="tts", size=40, timestamp=now - 2 * 86400, monkeypatch=monkeypatch)
    _save_artifact(store, run_id="run-new", stage="finalize", artifact_type="video_final", size=50, timestamp=now - 1 * 86400, monkeypatch=monkeypatch)

    artifacts = load_artifacts(fs_config)
    options = RetentionOptions(
        max_bytes=60,
        max_age_seconds=7 * 86400,
        min_free_bytes=70,
    )
    plan = plan_retention(artifacts, options, disk_free_bytes=0, now_ts=now)

    reasons = [candidate.reason for candidate in plan.candidates]
    assert reasons == ["max_age", "max_bytes", "min_free"]
    assert plan.freed_bytes == 100
    assert plan.remaining_bytes == 0


def test_execute_plan_removes_files_and_rows(
    store: FilesystemArtifactStore,
    fs_config: FilesystemArtifactsConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_700_100_000
    _save_artifact(store, run_id="run-clean", stage="dialogue", artifact_type="tts", size=12, timestamp=now - 5, monkeypatch=monkeypatch)
    artifacts = load_artifacts(fs_config)
    plan = plan_retention(artifacts, RetentionOptions(max_bytes=0), disk_free_bytes=0, now_ts=now)
    assert plan.candidates, "Expected at least one candidate"
    candidate = plan.candidates[0]
    payload_path = (fs_config.root / candidate.artifact.relative_path)
    assert payload_path.exists()

    execute_plan(plan, fs_config, dry_run=False)

    assert not payload_path.exists()
    with sqlite3.connect(fs_config.index_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM artifacts WHERE artifact_id = ?",
            (candidate.artifact.artifact_id,),
        ).fetchone()
        assert row is None


def test_cli_prune_dry_run_outputs_summary(
    store: FilesystemArtifactStore,
    fs_config: FilesystemArtifactsConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _save_artifact(
        store,
        run_id="run-cli",
        stage="plan",
        artifact_type="movie_plan",
        size=8,
        timestamp=1_700_200_000,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setenv("ARTIFACTS_BACKEND", "filesystem")
    monkeypatch.setenv("ARTIFACTS_FS_ALLOW_INSECURE", "1")
    monkeypatch.setenv("ARTIFACTS_FS_TOKEN", "local-token")
    args = [
        "prune",
        "--root",
        str(fs_config.root),
        "--index",
        str(fs_config.index_path),
        "--max-bytes",
        "0",
    ]
    code = filesystem_cli_main(args)
    captured = capsys.readouterr()
    assert code == 0
    assert "Dry run complete" in captured.out
    assert "Would prune" in captured.out


def test_cli_env_prints_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "fs_root"
    index = tmp_path / "index.db"
    args = [
        "env",
        "--root",
        str(root),
        "--index",
        str(index),
        "--token",
        "shim-token",
        "--emit-token",
    ]
    monkeypatch.delenv("ARTIFACTS_FS_TOKEN", raising=False)
    code = filesystem_cli_main(args)
    captured = capsys.readouterr()
    assert code == 0
    assert "export ARTIFACTS_BACKEND=filesystem" in captured.out
    assert str(root.resolve()) in captured.out
    assert "shim-token" in captured.out


def test_cli_health_reports_status(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    class DummyResponse:
        status_code = 200
        text = '{"status": "ok"}'

        def json(self) -> dict[str, str]:
            return {"status": "ok"}

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> DummyResponse:
        assert url == "http://127.0.0.1:7077/healthz"
        assert headers.get("Authorization") == "Bearer shim-token"
        assert timeout == 5.0
        return DummyResponse()

    monkeypatch.setattr("sparkle_motion.filesystem_artifacts.cli.httpx.get", fake_get)
    args = [
        "health",
        "--root",
        str(tmp_path / "root"),
        "--index",
        str(tmp_path / "index.db"),
        "--token",
        "shim-token",
        "--timeout",
        "5.0",
    ]
    code = filesystem_cli_main(args)
    captured = capsys.readouterr()
    assert code == 0
    assert "Filesystem backend healthy" in captured.out


def test_cli_serve_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {}

    def fake_run(app, host: str, port: int, log_level: str, reload: bool) -> None:
        called.update({"host": host, "port": port, "log_level": log_level, "reload": reload, "app": app})

    monkeypatch.setattr("sparkle_motion.filesystem_artifacts.cli.uvicorn.run", fake_run)
    args = [
        "serve",
        "--root",
        str(tmp_path / "root"),
        "--index",
        str(tmp_path / "index.db"),
        "--allow-insecure",
        "--host",
        "0.0.0.0",
        "--port",
        "7788",
        "--log-level",
        "debug",
    ]
    code = filesystem_cli_main(args)
    assert code == 0
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 7788
    assert called["log_level"] == "debug"
