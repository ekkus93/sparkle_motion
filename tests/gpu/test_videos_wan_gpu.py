from __future__ import annotations

from pathlib import Path

import pytest

from sparkle_motion.function_tools.videos_wan import adapter as wan_adapter

from . import helpers


@pytest.mark.gpu
def test_wan_render_short_clip(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
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

    start_frame_path = helpers.asset_path("sample_image.png")
    start_frame_bytes = start_frame_path.read_bytes()
    # Flip a byte so the end frame differs without creating a new asset on disk.
    mutated_end = bytearray(start_frame_bytes)
    mutated_end[0] ^= 0xFF
    end_frame_bytes = bytes(mutated_end)

    output_dir = tmp_path / "wan_videos"
    with helpers.temp_output_dir(monkeypatch, "VIDEOS_WAN_OUTPUT_DIR", output_dir):
        result = wan_adapter.render_clip(
            prompt="Gentle camera pan",
            num_frames=16,
            fps=8,
            width=720,
            height=720,
            seed=200,
            start_frame=start_frame_bytes,
            end_frame=end_frame_bytes,
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
