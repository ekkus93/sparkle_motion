from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sparkle_motion.adapters import get_stub_adapter


ffmpeg_bin = shutil.which("ffmpeg")
ffprobe_bin = shutil.which("ffprobe")


def _ffprobe_info(path: Path) -> dict:
    """Run ffprobe and return parsed JSON info for the given file.

    Raises RuntimeError if ffprobe isn't available or fails.
    """
    if not ffprobe_bin:
        raise RuntimeError("ffprobe not available on PATH")
    cmd = [
        ffprobe_bin,
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


def _detect_audio_codec() -> str | None:
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


@pytest.mark.skipif(ffmpeg_bin is None, reason="ffmpeg not available on PATH")
def test_assembled_file_playable_with_ffmpeg(tmp_path: Path):
    """Integration test: synthesize a real MP4 (via assemble) and assert
    ffmpeg can decode it. The test is skipped if ffmpeg is not installed.
    """
    adapter = get_stub_adapter()
    res = adapter.assemble([], out_path=tmp_path / "out.mp4")
    out_path = res.path
    assert out_path.exists()

    # Ask ffmpeg to decode the file; return code 0 indicates decode succeeded
    cmd = [ffmpeg_bin, "-v", "error", "-i", str(out_path), "-f", "null", "-"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # If ffprobe is available, assert stream codecs and duration
    if ffprobe_bin:
        info = _ffprobe_info(out_path)
        # duration should be approximately the encode time used in assemble (0.6s)
        duration = float(info.get("format", {}).get("duration", 0.0))
        assert abs(duration - 0.6) <= 0.3
        streams = info.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        assert video_streams, "no video stream found"
        vcodec = video_streams[0].get("codec_name", "")
        assert "mpeg4" in vcodec or "mpeg" in vcodec


@pytest.mark.skipif(ffmpeg_bin is None, reason="ffmpeg not available on PATH")
def test_assemble_add_audio_true_produces_audio(tmp_path: Path):
    """When add_audio=True and ffmpeg is present, assembled file should have an audio stream."""
    adapter = get_stub_adapter()
    res = adapter.assemble([], out_path=tmp_path / "with_audio.mp4", add_audio=True)
    out_path = res.path
    assert out_path.exists()
    # Use ffprobe when available for robust stream inspection
    if ffprobe_bin:
        info = _ffprobe_info(out_path)
        streams = info.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        assert audio_streams, "no audio stream found"
        detected = _detect_audio_codec()
        if detected:
            acodec = audio_streams[0].get("codec_name", "")
            assert detected.split("lib")[-1] in acodec or detected in acodec
    else:
        # Fallback: check ffmpeg stderr contains Audio: line
        proc = subprocess.run([ffmpeg_bin, "-hide_banner", "-i", str(out_path)], capture_output=True, text=True)
        assert "Audio:" in proc.stderr

    assert res.audio_included is True
