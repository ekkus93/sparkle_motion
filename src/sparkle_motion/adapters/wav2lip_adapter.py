from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Wav2LipAdapter:
    """Scaffold for Wav2Lip-based lipsyncing.

    Real usage: prefer Python API when available; otherwise a subprocess wrapper
    to the Wav2Lip repo (pinned commit) is acceptable. This scaffold writes a
    small fake MP4 for unit tests.
    """
    checkpoint: Optional[str] = None

    def run(self, video_path: Path, audio_path: Path, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # In unit tests we avoid running real Wav2Lip: create a fake file.
        out_path.write_bytes(b"FAKE_LIPSYNC_MP4")
        return out_path
