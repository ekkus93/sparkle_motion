#!/usr/bin/env python3
"""Generate deterministic media fixtures for tests.

This script keeps `tests/fixtures/assets/` populated with tiny (<50 KB)
PNG, WAV, MP4, and JSON plan files so tests can reference consistent
artifacts without synthesizing bytes on the fly.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

from PIL import Image

from sparkle_motion.function_tools.videos_wan import adapter as videos_adapter

ASSET_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "assets"


def _ensure_dir() -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    return ASSET_DIR


def _write_png(path: Path) -> None:
    width, height = 32, 32
    image = Image.new("RGB", (width, height))
    pixels = []
    for y in range(height):
        for x in range(width):
            r = (x * 11 + y * 7) % 256
            g = (x * 3 + y * 13) % 256
            b = (x * 5 + y * 17) % 256
            pixels.append((r, g, b))
    image.putdata(pixels)
    image.save(path, format="PNG", optimize=True)


def _write_wav(path: Path) -> None:
    sample_rate = 16_000
    duration_s = 0.35
    frequency = 660.0
    frames = int(sample_rate * duration_s)
    amplitude = 32767
    buffer = bytearray()
    for n in range(frames):
        value = int(amplitude * math.sin(2 * math.pi * frequency * n / sample_rate))
        buffer += struct.pack("<h", value)
    with wave.open(path.as_posix(), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(buffer)


def _write_mp4(path: Path) -> None:
    payload = videos_adapter._fixture_payload(  # type: ignore[attr-defined]
        seed=123456,
        width=128,
        height=72,
        num_frames=24,
    )
    path.write_bytes(payload)


def _write_plan(path: Path) -> None:
    plan = {
        "plan_id": "fixture-plan-001",
        "title": "Fixture Demo Plan",
        "shots": [
            {
                "shot_id": "shot-1",
                "prompt": "A calm river at sunrise",
                "num_frames": 16,
                "audio_ref": "tests/fixtures/assets/sample_audio.wav",
                "video_ref": "tests/fixtures/assets/sample_video.mp4",
                "image_ref": "tests/fixtures/assets/sample_image.png",
            }
        ],
    }
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main() -> None:
    base = _ensure_dir()
    _write_png(base / "sample_image.png")
    _write_wav(base / "sample_audio.wav")
    _write_mp4(base / "sample_video.mp4")
    _write_plan(base / "sample_plan.json")
    print(f"Wrote fixture assets to {base}")


if __name__ == "__main__":
    main()
