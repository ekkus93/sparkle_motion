from __future__ import annotations

import sys
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sparkle_motion.function_tools.assemble_ffmpeg import adapter

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def test_fixture_assembly_creates_file(tmp_path, monkeypatch, deterministic_media_assets: MediaAssets):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    clip_path = tmp_path / "clip.mp4"
    shutil.copyfile(deterministic_media_assets.video, clip_path)

    result = adapter.assemble_movie(
        clips=[adapter.ClipSpec(uri=clip_path)],
        plan_id="plan-fixture",
        output_dir=tmp_path / "artifacts",
    )

    assert result.engine == "fixture"
    assert result.path.exists()
    assert result.metadata["clip_count"] == 1


def test_run_command_timeout(monkeypatch):
    with pytest.raises(adapter.CommandTimeoutError):
        adapter.run_command(
            [sys.executable, "-c", "import time; time.sleep(0.3)"],
            timeout_s=0.05,
            retries=0,
        )


def test_should_use_real_engine_gate(monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    assert adapter.should_use_real_engine() is False
    monkeypatch.delenv("ADK_USE_FIXTURE")
    monkeypatch.setenv("SMOKE_ASSEMBLE", "1")
    assert adapter.should_use_real_engine() is True
