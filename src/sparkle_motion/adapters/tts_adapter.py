"""Text-to-speech adapter stub (Chatterbox / other TTS).

Expose `generate_audio(movie_plan, asset_refs, run_dir)` to produce per-line or
per-shot WAV files and update asset_refs accordingly. This stub raises
MissingDependencyError with instructions.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from .common import MissingDependencyError


def generate_audio(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Generate WAV audio for each dialogue line.

    Real implementations may call Chatterbox TTS, Coqui TTS, or another
    local TTS package. If you want a thin fallback, implement a basic
    gTTS-based writer here.
    """
    raise MissingDependencyError(
        "TTS adapter not implemented.\n"
        "Implement src.sparkle_motion.adapters.tts_adapter.generate_audio using your chosen TTS backend.\n"
        "Example packages: TTS (Coqui), pyttsx3, or a local Chatterbox server.\n"
        "For now the orchestrator will fall back to a lightweight simulation if this raises."
    )
