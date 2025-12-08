from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from sparkle_motion.function_tools.assemble_ffmpeg import adapter

from . import helpers


def _generate_color_clip(dest: Path, duration_s: float = 1.5, *, color: str = "navy") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=640x360:d={duration_s}",
        "-r",
        "24",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(f"ffmpeg color clip generation failed (code {proc.returncode}): {proc.stderr.strip()}")


def _generate_sine_audio(dest: Path, duration_s: float, *, frequency: int = 320) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={frequency}:duration={duration_s}",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "3",
        str(dest),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(f"ffmpeg sine audio generation failed (code {proc.returncode}): {proc.stderr.strip()}")


def _video_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"ffprobe duration probe failed for {path}: {proc.stderr.strip()}")
    return float(proc.stdout.strip())


def _video_has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"ffprobe audio stream probe failed for {path}: {proc.stderr.strip()}")
    return any(line.strip().lower() == "audio" for line in proc.stdout.splitlines())


@pytest.mark.gpu
def test_assemble_single_clip(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_ASSEMBLE", "SMOKE_ADAPTERS"],
        disable_keys=["ADK_USE_FIXTURE", "ASSEMBLE_FFMPEG_FIXTURE_ONLY"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "ASSEMBLE_FFMPEG_FIXTURE_ONLY": "0",
        },
    )

    clip_path = tmp_path / "inputs" / "clip.mp4"
    _generate_color_clip(clip_path, duration_s=1.5)

    output_dir = tmp_path / "outputs"
    result = adapter.assemble_movie(
        clips=[adapter.ClipSpec(uri=clip_path)],
        audio=None,
        plan_id="gpu-single-clip",
        output_dir=output_dir,
        options={"fixture_only": False},
    )

    assert result.engine == "ffmpeg"
    assert result.path.exists()
    assert result.path.suffix == ".mp4"
    assert result.path.stat().st_size > 0

    probe = helpers.probe_video(result.path)
    assert probe.returncode == 0, f"ffprobe failed for {result.path}: {probe.stderr}"

    input_duration = _video_duration(clip_path)
    output_duration = _video_duration(result.path)
    assert pytest.approx(input_duration, abs=0.05) == output_duration

    assert _video_has_audio_stream(result.path) is False, "Assembled clip should not contain an audio stream"

    metadata = result.metadata
    assert metadata.get("engine") == "ffmpeg"
    assert metadata.get("clip_count") == 1
    assert metadata.get("audio_attached") is False


@pytest.mark.gpu
def test_assemble_multiple_clips_with_audio(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_ASSEMBLE", "SMOKE_ADAPTERS"],
        disable_keys=["ADK_USE_FIXTURE", "ASSEMBLE_FFMPEG_FIXTURE_ONLY"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "ASSEMBLE_FFMPEG_FIXTURE_ONLY": "0",
        },
    )

    clip_durations: Sequence[float] = (1.1, 1.3, 0.9)
    clip_colors: Sequence[str] = ("navy", "teal", "maroon")
    clips: list[adapter.ClipSpec] = []
    total_duration = 0.0
    for idx, duration in enumerate(clip_durations):
        clip_path = tmp_path / "inputs" / f"clip_{idx}.mp4"
        _generate_color_clip(clip_path, duration_s=duration, color=clip_colors[idx % len(clip_colors)])
        clips.append(adapter.ClipSpec(uri=clip_path))
        total_duration += duration

    audio_path = tmp_path / "inputs" / "bgm.mp3"
    _generate_sine_audio(audio_path, duration_s=total_duration + 0.25)

    output_dir = tmp_path / "outputs"
    result = adapter.assemble_movie(
        clips=clips,
        audio=adapter.AudioSpec(uri=audio_path),
        plan_id="gpu-multi-clip",
        output_dir=output_dir,
        options={"fixture_only": False},
    )

    assert result.engine == "ffmpeg"
    assert result.path.exists()
    assert result.path.stat().st_size > 0
    assert result.metadata.get("clip_count") == len(clips)
    assert result.metadata.get("audio_attached") is True

    probe = helpers.probe_video(result.path)
    assert probe.returncode == 0, f"ffprobe failed for {result.path}: {probe.stderr}"

    final_duration = _video_duration(result.path)
    assert pytest.approx(total_duration, abs=0.2) == final_duration

    assert _video_has_audio_stream(result.path) is True

    metadata = result.metadata
    assert metadata.get("engine") == "ffmpeg"
    assert metadata.get("plan_id") == "gpu-multi-clip"
