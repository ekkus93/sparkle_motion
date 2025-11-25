from __future__ import annotations

from pathlib import Path

from sparkle_motion.colab_helper import WorkspaceLayout, download_model, ensure_workspace, run_smoke_check


def test_ensure_workspace_creates_directories(tmp_path: Path) -> None:
    layout = ensure_workspace(tmp_path / "workspace")
    assert layout.models.exists()
    assert layout.assets.exists()
    assert layout.outputs.exists()
    assert layout.runs.exists()
    assert layout.logs.exists()


def test_run_smoke_check_writes_json(tmp_path: Path) -> None:
    layout = WorkspaceLayout(
        root=tmp_path,
        models=tmp_path / "models",
        assets=tmp_path / "assets",
        outputs=tmp_path / "outputs",
        runs=tmp_path / "runs",
        logs=tmp_path / "logs",
    )
    path = run_smoke_check(layout, message="test")
    data = path.read_text(encoding="utf-8")
    assert "test" in data
    assert path.exists()


def test_download_model_uses_injected_downloader(tmp_path: Path) -> None:
    calls = {}

    def fake_downloader(**kwargs):
        calls.update(kwargs)
        target = Path(kwargs["cache_dir"]) / "model"
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    out = download_model(
        repo_id="demo/repo",
        target_dir=tmp_path / "models",
        revision="main",
        allow_patterns=["*.bin"],
        downloader=fake_downloader,
    )
    assert out.exists()
    assert calls["repo_id"] == "demo/repo"
    assert calls["revision"] == "main"
    assert calls["allow_patterns"] == ["*.bin"]
