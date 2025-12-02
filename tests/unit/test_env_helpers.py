from __future__ import annotations

import pytest

from sparkle_motion.filesystem_artifacts.config import FilesystemArtifactsConfig
from sparkle_motion.utils import env as env_utils


def test_resolve_artifacts_backend_defaults(monkeypatch):
    monkeypatch.delenv("ARTIFACTS_BACKEND", raising=False)
    assert env_utils.resolve_artifacts_backend() == "adk"
    assert not env_utils.filesystem_backend_enabled()


def test_resolve_artifacts_backend_filesystem(monkeypatch):
    monkeypatch.setenv("ARTIFACTS_BACKEND", "filesystem")
    assert env_utils.resolve_artifacts_backend() == "filesystem"
    assert env_utils.filesystem_backend_enabled()


def test_resolve_artifacts_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv("ARTIFACTS_BACKEND", "bogus")
    with pytest.raises(ValueError):
        env_utils.resolve_artifacts_backend()


def test_filesystem_config_reads_process_env(monkeypatch, tmp_path):
    root = tmp_path / "fs"
    index = root / "index.db"
    monkeypatch.setenv("ARTIFACTS_FS_ROOT", str(root))
    monkeypatch.setenv("ARTIFACTS_FS_INDEX", str(index))
    monkeypatch.setenv("ARTIFACTS_FS_TOKEN", "test-token")
    cfg = FilesystemArtifactsConfig.from_env()
    assert cfg.root == root.resolve()
    assert cfg.index_path == index.resolve()
    assert cfg.token == "test-token"
    assert root.exists()
    assert index.parent.exists()

