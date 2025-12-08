from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import pytest

from sparkle_motion.function_tools.videos_wan import adapter as wan_adapter

from . import helpers


def _render_real_wan(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: Path,
    *,
    prompt: str,
    render_kwargs: Mapping[str, Any],
    subdir: str,
) -> wan_adapter.VideoRenderResult:
    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_ADAPTERS", "SMOKE_VIDEOS"],
        disable_keys=["VIDEOS_WAN_FIXTURE_ONLY", "ADK_USE_FIXTURE"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "VIDEOS_WAN_FIXTURE_ONLY": "0",
        },
    )

    output_dir = tmp_path / subdir
    with helpers.temp_output_dir(monkeypatch, "VIDEOS_WAN_OUTPUT_DIR", output_dir):
        return wan_adapter.render_clip(prompt=prompt, **dict(render_kwargs))


def _sample_frame_bytes() -> bytes:
    return helpers.asset_path("sample_image.png").read_bytes()


def _mutate_frame_bytes(data: bytes, *, offset: int, xor: int = 0xAA) -> bytes:
    mutated = bytearray(data)
    idx = offset % len(mutated)
    mutated[idx] ^= xor
    return bytes(mutated)


@pytest.mark.gpu
def test_wan_render_short_clip(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    start_frame_bytes = _sample_frame_bytes()
    end_frame_bytes = _mutate_frame_bytes(start_frame_bytes, offset=0, xor=0xFF)

    result = _render_real_wan(
        monkeypatch,
        tmp_path,
        prompt="Gentle camera pan",
        render_kwargs={
            "num_frames": 16,
            "fps": 8,
            "width": 720,
            "height": 720,
            "seed": 200,
            "start_frame": start_frame_bytes,
            "end_frame": end_frame_bytes,
        },
        subdir="wan_short_clip",
    )

    assert result.path.exists(), "Rendered MP4 should exist on disk"
    assert result.path.suffix == ".mp4", "Wan adapter must emit MP4 clips"
    assert result.path.stat().st_size > 10 * 1024, "Real Wan clip should exceed 10KB"

    probe = helpers.probe_video(result.path)
    assert probe.returncode == 0, f"ffprobe failed: {probe.stderr}"

    assert result.engine == "wan2.1", "Real engine should be reported"
    assert result.frame_count == 16
    assert result.fps == 8
    assert pytest.approx(result.duration_s, abs=0.1) == 2.0

    metadata = result.metadata
    assert metadata.get("engine") == "wan2.1"
    assert metadata.get("num_frames") == 16
    assert metadata.get("fps") == 8
    assert metadata.get("duration_s", result.duration_s) == pytest.approx(2.0, abs=0.1)
    assert metadata.get("start_frame_hash") == hashlib.sha1(start_frame_bytes).hexdigest()
    assert metadata.get("end_frame_hash") == hashlib.sha1(end_frame_bytes).hexdigest()


@pytest.mark.gpu
def test_wan_keyframe_interpolation(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    start_frame = _sample_frame_bytes()
    end_frame = _mutate_frame_bytes(start_frame, offset=37, xor=0x33)

    result = _render_real_wan(
        monkeypatch,
        tmp_path,
        prompt="Keyframe desert-to-ocean mix",
        render_kwargs={
            "num_frames": 32,
            "fps": 16,
            "width": 720,
            "height": 720,
            "seed": 210,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
        subdir="wan_keyframe",
    )

    assert result.path.exists()
    assert helpers.probe_video(result.path).returncode == 0
    assert result.frame_count == 32
    assert result.fps == 16
    assert pytest.approx(result.duration_s, abs=0.1) == 2.0

    metadata = result.metadata
    assert metadata.get("engine") == "wan2.1"
    assert metadata.get("start_frame_hash") == hashlib.sha1(start_frame).hexdigest()
    assert metadata.get("end_frame_hash") == hashlib.sha1(end_frame).hexdigest()


@pytest.mark.gpu
def test_wan_seed_reproducibility(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    render_kwargs = {
        "num_frames": 24,
        "fps": 12,
        "width": 512,
        "height": 512,
        "seed": 150,
        "start_frame": _sample_frame_bytes(),
    }

    first = _render_real_wan(
        monkeypatch,
        tmp_path,
        prompt="Slow zoom",
        render_kwargs=render_kwargs,
        subdir="wan_seed_a",
    )
    second = _render_real_wan(
        monkeypatch,
        tmp_path,
        prompt="Slow zoom",
        render_kwargs=render_kwargs,
        subdir="wan_seed_b",
    )

    assert first.metadata.get("seed") == second.metadata.get("seed") == 150
    assert first.frame_count == second.frame_count == 24
    assert first.path.read_bytes() == second.path.read_bytes(), "Wan renders must be deterministic for identical seeds"


@pytest.mark.gpu
def test_wan_adaptive_chunking(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    result = _render_real_wan(
        monkeypatch,
        tmp_path,
        prompt="Long sequence",
        render_kwargs={
            "num_frames": 128,
            "fps": 16,
            "width": 720,
            "height": 720,
            "seed": 300,
            "chunk_index": 0,
            "chunk_count": 8,
        },
        subdir="wan_chunking",
    )

    assert result.frame_count == 128
    assert result.fps == 16
    assert pytest.approx(result.duration_s, abs=0.2) == 8.0
    metadata = result.metadata
    assert metadata.get("chunk_count") == 8
    assert metadata.get("chunk_index") == 0

