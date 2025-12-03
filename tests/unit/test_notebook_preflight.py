from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sparkle_motion import notebook_preflight as np


class DummyCompleted:
    def __init__(self) -> None:
        self.stdout = "ok"
        self.stderr = ""


def test_check_env_vars_reports_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in np.DEFAULT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    result = np._check_env_vars(np.DEFAULT_ENV_VARS)
    assert result.status == "error"
    assert "Missing env vars" in result.detail


def test_check_drive_mount_skips_when_not_colab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setattr(np, "_running_in_colab", lambda: False)
    result = np._check_drive_mount(tmp_path, tmp_path / "workspace", True)
    assert result.status == "warning"
    assert "skipping" in result.detail.lower()


def test_probe_ready_endpoints_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    class FailingClient:
        def __init__(self, **_: Any) -> None:
            pass

        def get(self, *_: Any, **__: Any) -> Any:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

        def close(self) -> None:
            pass

    monkeypatch.setattr(np.httpx, "Client", FailingClient)
    result = np._probe_ready_endpoints(["http://localhost:1/ready"])
    assert result.status == "error"
    assert "probe failed" in result.detail.lower()


def test_run_preflight_checks_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    req_path = tmp_path / "requirements-ml.txt"
    req_path.write_text("httpx\n")
    workspace = tmp_path / "workspace"

    def fake_run(*_: Any, **__: Any) -> DummyCompleted:
        return DummyCompleted()

    monkeypatch.setenv("SPARKLE_DB_PATH", "db")
    monkeypatch.setenv("ARTIFACTS_DIR", "artifacts")
    monkeypatch.setenv("GOOGLE_ADK_PROFILE", "local-colab")
    monkeypatch.setattr(np.subprocess, "run", fake_run)
    monkeypatch.setattr(np.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(np, "_running_in_colab", lambda: False)

    results = np.run_preflight_checks(
        requirements_path=req_path,
        mount_point=tmp_path,
        workspace_dir=workspace,
        ready_endpoints=(),
        pip_mode="install",
        require_drive=False,
        skip_gpu_checks=True,
    )
    status_map = {result.name: result.status for result in results}
    assert status_map["env"] == "ok"
    assert status_map["pip"] == "ok"
    assert status_map["drive"] == "warning"
    assert status_map["ready"] == "warning"
    assert any(result.detail for result in results)
