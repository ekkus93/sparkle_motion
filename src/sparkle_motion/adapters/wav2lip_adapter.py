"""Wav2Lip adapter stub.

Expose `lipsync(movie_plan, asset_refs, run_dir)` which performs lip-syncing
on shots flagged as talking closeups. This stub raises MissingDependencyError.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from .common import MissingDependencyError


def lipsync(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Apply Wav2Lip to talking shots.

    Replace with real Wav2Lip invocation. Typical requirements include a
    Wav2Lip repo, a face-detection dependency, and ffmpeg for muxing.
    """
    raise MissingDependencyError(
        "Wav2Lip adapter not implemented.\n"
        "Implement src.sparkle_motion.adapters.wav2lip_adapter.lipsync to call Wav2Lip.\n"
        "For now the orchestrator will fall back to a lightweight simulation if this raises."
    )
