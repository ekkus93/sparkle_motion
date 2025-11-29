from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from sparkle_motion.function_tools.lipsync_wav2lip import adapter


@dataclass
class Wav2LipAdapter:
    """Adapter wrapper for invoking the FunctionTool implementation."""

    options: Optional[Mapping[str, Any]] = None

    def run(self, video_path: Path, audio_path: Path, out_path: Path) -> Path:
        result = adapter.run_wav2lip(video_path, audio_path, out_path, opts=self.options or {})
        return result.path
