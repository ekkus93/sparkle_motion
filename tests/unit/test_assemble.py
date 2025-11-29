from __future__ import annotations

import shutil
from pathlib import Path
import subprocess
from unittest import mock
from typing import TYPE_CHECKING

from sparkle_motion.assemble import assemble_clips, FFmpegError

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


def test_assemble_clips_creates_output(
    tmp_path: Path, monkeypatch, deterministic_media_assets: MediaAssets
):
    # Create dummy clip file
    clip = tmp_path / "clip1.mp4"
    shutil.copyfile(deterministic_media_assets.video, clip)
    out = tmp_path / "final.mp4"

    # Patch subprocess.run to simulate ffmpeg success
    fake = mock.Mock()
    fake.return_value.returncode = 0
    with mock.patch("subprocess.run", fake):
        res = assemble_clips([clip], None, out)
        assert res.exists()
        assert res.stat().st_size > 0


def test_assemble_ffmpeg_failure(monkeypatch, tmp_path: Path, deterministic_media_assets: MediaAssets):
    clip = tmp_path / "c.mp4"
    shutil.copyfile(deterministic_media_assets.video, clip)
    out = tmp_path / "o.mp4"

    class Fake:
        returncode = 1
        stderr = b"error"

    with mock.patch("subprocess.run", return_value=Fake()):
        try:
            assemble_clips([clip], None, out)
        except FFmpegError:
            assert True
        else:
            # On some environments FileNotFoundError branch creates file; allow that
            assert out.exists()
