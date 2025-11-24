"""Wan FLF2V adapter stub.

Real implementation should provide `generate_video(movie_plan, asset_refs, run_dir)`
which produces raw clips for each shot. This stub raises MissingDependencyError
with guidance on implementation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from .common import MissingDependencyError


def generate_video(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Generate raw video clips from start/end frames using Wan FLF2V.

    Required work: wrap Wan's repo/run script or use its Python API (if available)
    to accept start/end images and produce an mp4 per shot.

    Example required packages / repos:
      - a local clone of the Wan FLF2V code (not always on PyPI)
      - ffmpeg (system binary) for packaging clips

    This stub intentionally raises MissingDependencyError and documents where
    to implement the real logic.
    """
    raise MissingDependencyError(
        "Wan adapter not implemented.\n"
        "Implement src.sparkle_motion.adapters.wan_adapter.generate_video to call Wan FLF2V.\n"
        "Wan may require cloning an external repo and providing model checkpoints.\n"
        "For now the orchestrator will fall back to a lightweight simulation if this raises."
    )
