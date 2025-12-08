from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Sequence

import pytest

from sparkle_motion.function_tools.lipsync_wav2lip import adapter as lipsync_adapter

from . import helpers


def _resolve_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not configured; see docs/MODEL_INSTALL_NOTES.md for Wav2Lip setup")
    path = Path(value).expanduser()
    if not path.exists():
        pytest.skip(f"{name} path does not exist: {path}")
    return path


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    face = tmp_path / "inputs" / "face.mp4"
    audio = tmp_path / "inputs" / "dialogue.wav"
    face.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(helpers.asset_path("sample_video.mp4"), face)
    shutil.copy2(helpers.asset_path("sample_audio.wav"), audio)
    return face, audio


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
        raise AssertionError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise AssertionError(f"Unable to parse duration for {path}: {proc.stdout}") from exc


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
        raise AssertionError(f"ffprobe stream probe failed for {path}: {proc.stderr.strip()}")
    return any(line.strip().lower() == "audio" for line in proc.stdout.splitlines())


def _audio_duration(path: Path) -> float:
    with wave.open(path.as_posix(), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    if rate == 0:
        raise AssertionError(f"Audio file has zero sample rate: {path}")
    return frames / float(rate)


def _concat_audio_segments(segments: Sequence[Path], dest: Path) -> float:
    if not segments:
        raise AssertionError("Audio segment list must not be empty")
    dest.parent.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    sample_rate = None
    sample_width = None
    channels = None

    with wave.open(dest.as_posix(), "wb") as target:
        for idx, segment in enumerate(segments):
            with wave.open(segment.as_posix(), "rb") as src:
                if idx == 0:
                    channels = src.getnchannels()
                    sample_width = src.getsampwidth()
                    sample_rate = src.getframerate()
                    target.setnchannels(channels)
                    target.setsampwidth(sample_width)
                    target.setframerate(sample_rate)
                else:
                    if src.getnchannels() != channels or src.getsampwidth() != sample_width or src.getframerate() != sample_rate:
                        raise AssertionError("All audio segments must share the same format")
                frames = src.readframes(src.getnframes())
                total_frames += src.getnframes()
                target.writeframes(frames)

    if sample_rate is None:
        raise AssertionError("Failed to determine sample rate while concatenating audio")
    return total_frames / float(sample_rate)


def _extend_video(face_src: Path, repeats: int, dest: Path) -> None:
    if repeats <= 1:
        shutil.copy2(face_src, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        str(repeats - 1),
        "-i",
        str(face_src),
        "-c",
        "copy",
        str(dest),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(f"ffmpeg loop failed (code {proc.returncode}): {proc.stderr.strip()}")


@pytest.mark.gpu
def test_lipsync_single_clip(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    repo_path = _resolve_env_path("WAV2LIP_REPO")
    checkpoint_path = _resolve_env_path("LIPSYNC_WAV2LIP_MODEL")

    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_LIPSYNC", "SMOKE_ADAPTERS"],
        disable_keys=["LIPSYNC_WAV2LIP_FIXTURE_ONLY", "ADK_USE_FIXTURE"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "LIPSYNC_WAV2LIP_FIXTURE_ONLY": "0",
            "WAV2LIP_REPO": str(repo_path),
            "LIPSYNC_WAV2LIP_MODEL": str(checkpoint_path),
        },
    )

    face_video, audio_track = _prepare_inputs(tmp_path)
    output_path = tmp_path / "outputs" / "synced.mp4"

    result = lipsync_adapter.run_wav2lip(
        face_video,
        audio_track,
        output_path,
        opts={
            "repo_path": repo_path,
            "checkpoint_path": checkpoint_path,
            "allow_fixture_fallback": False,
        },
    )

    assert result.path == output_path
    assert output_path.exists() and output_path.stat().st_size > 0
    assert output_path.suffix == ".mp4"

    probe = helpers.probe_video(output_path)
    assert probe.returncode == 0, f"ffprobe failed: {probe.stderr}"

    input_duration = _video_duration(face_video)
    output_duration = _video_duration(output_path)
    assert pytest.approx(input_duration, abs=0.2) == output_duration

    assert _video_has_audio_stream(output_path), "Synced clip must contain an audio stream"

    metadata = result.metadata
    engine = metadata.get("engine")
    assert engine in {"wav2lip_subprocess", "wav2lip_fixture"}
    assert metadata.get("face_video")
    assert metadata.get("audio")

    logs = result.logs or {}
    assert "stdout" in logs


@pytest.mark.gpu
def test_lipsync_multiple_audio_tracks(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    repo_path = _resolve_env_path("WAV2LIP_REPO")
    checkpoint_path = _resolve_env_path("LIPSYNC_WAV2LIP_MODEL")

    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_LIPSYNC", "SMOKE_ADAPTERS"],
        disable_keys=["LIPSYNC_WAV2LIP_FIXTURE_ONLY", "ADK_USE_FIXTURE"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "LIPSYNC_WAV2LIP_FIXTURE_ONLY": "0",
            "WAV2LIP_REPO": str(repo_path),
            "LIPSYNC_WAV2LIP_MODEL": str(checkpoint_path),
        },
    )

    sample_audio = helpers.asset_path("sample_audio.wav")
    segments: list[Path] = []
    for idx in range(3):
        seg = tmp_path / "segments" / f"line_{idx}.wav"
        seg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample_audio, seg)
        segments.append(seg)

    combined_audio = tmp_path / "combined" / "dialogue.wav"
    combined_duration = _concat_audio_segments(segments, combined_audio)

    base_video = helpers.asset_path("sample_video.mp4")
    face_video = tmp_path / "inputs" / "face_loop.mp4"
    _extend_video(base_video, repeats=len(segments), dest=face_video)
    face_duration = _video_duration(face_video)

    output_path = tmp_path / "outputs" / "synced_multi.mp4"
    result = lipsync_adapter.run_wav2lip(
        face_video,
        combined_audio,
        output_path,
        opts={
            "repo_path": repo_path,
            "checkpoint_path": checkpoint_path,
            "allow_fixture_fallback": False,
        },
    )

    assert result.path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    probe = helpers.probe_video(output_path)
    assert probe.returncode == 0

    final_duration = _video_duration(output_path)
    assert pytest.approx(face_duration, abs=0.3) == final_duration
    assert pytest.approx(combined_duration, abs=0.3) == final_duration

    assert _video_has_audio_stream(output_path)

    metadata = result.metadata
    assert metadata.get("engine") in {"wav2lip_subprocess", "wav2lip_fixture"}
    assert metadata.get("audio") == str(combined_audio)
    assert metadata.get("face_video") == str(face_video)

    logs = result.logs or {}
    assert "stdout" in logs