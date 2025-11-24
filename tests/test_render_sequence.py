from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

import pytest

from sparkle_motion.adapters import get_stub_adapter


def _has_ffmpeg_with_audio() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        p = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
        if p.returncode != 0:
            return False
        out = p.stdout.lower()
        return any(a in out for a in ("aac", "libmp3lame", "libopus"))
    except Exception:
        return False


def _detect_audio_codec() -> str | None:
    """Return preferred audio codec string detected from ffmpeg encoders.

    Mirrors the preference order used by the stub adapter (aac, libmp3lame, libopus).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        p = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
        if p.returncode != 0:
            return None
        out = p.stdout.lower()
        if "aac" in out:
            return "aac"
        if "libmp3lame" in out:
            return "libmp3lame"
        if "libopus" in out:
            return "libopus"
        return None
    except Exception:
        return None


def _ffprobe_info(path: Path) -> dict:
    """Run ffprobe and return parsed JSON info for the given file.

    Returns a dict with keys like 'format' and 'streams' (as ffprobe -print_format json). Raises RuntimeError if ffprobe isn't available.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not available on PATH")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    import json

    return json.loads(p.stdout)


def test_render_sequence_smoke_creates_file(tmp_path: Path):
    adapter = get_stub_adapter()

    # create a few deterministic frames using the adapter's generate_images
    imgs = adapter.generate_images("test", count=3)
    frames: List[bytes] = [a.data for a in imgs]

    out = tmp_path / "out_seq.mp4"
    res = adapter.render_sequence(frames, out_path=out, fps=1, add_audio=False)

    assert res.path == out
    assert out.exists()
    assert out.stat().st_size > 0
    assert res.audio_included is False


@pytest.mark.skipif(not _has_ffmpeg_with_audio(), reason="ffmpeg with audio encoders not available")
def test_render_sequence_with_audio_integration(tmp_path: Path):
    adapter = get_stub_adapter()

    imgs = adapter.generate_images("test-audio", count=2)
    frames: List[bytes] = [a.data for a in imgs]

    out = tmp_path / "out_seq_audio.mp4"
    res = adapter.render_sequence(frames, out_path=out, fps=1, add_audio=True)

    # Basic file checks
    assert out.exists()
    assert out.stat().st_size > 0
    assert res.audio_included is True

    # Use ffprobe to inspect streams and duration
    try:
        info = _ffprobe_info(out)
    except RuntimeError:
        pytest.skip("ffprobe not available; skipping detailed assertions")

    # check duration ~ expected (frames / fps)
    expected_duration = len(frames) / 1.0
    duration = float(info.get("format", {}).get("duration", 0.0))
    assert abs(duration - expected_duration) <= 0.3

    # check streams for video and audio codecs
    streams = info.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    assert video_streams, "no video stream found"
    vcodec = video_streams[0].get("codec_name", "")
    assert "mpeg4" in vcodec or "mpeg" in vcodec

    assert audio_streams, "no audio stream found"
    detected = _detect_audio_codec()
    if detected:
        # codec names in ffprobe may be short (e.g., 'aac') or different for libmp3lame
        acodec = audio_streams[0].get("codec_name", "")
        assert detected.split("lib")[-1] in acodec or detected in acodec
