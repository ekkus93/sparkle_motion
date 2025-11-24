"""Assembly adapter stub (ffmpeg helpers).

Expose `assemble(movie_plan, asset_refs, run_dir)` which muxes audio/video per-shot
and concatenates final movie. This stub raises MissingDependencyError with guidance.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from .common import MissingDependencyError


def assemble(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Assemble final per-shot outputs and write movie_final.mp4.

    Real implementation should use ffmpeg to mux audio and video, then
    concatenate shots. Ensure ffmpeg system binary is available.
    """
    raise MissingDependencyError(
        "Assemble adapter not implemented.\n"
        "Implement src.sparkle_motion.adapters.assemble_adapter.assemble using ffmpeg or moviepy.\n"
        "System dependency: ffmpeg (installable in Colab via apt).\n"
        "For now the orchestrator will fall back to a lightweight simulation if this raises."
    )
