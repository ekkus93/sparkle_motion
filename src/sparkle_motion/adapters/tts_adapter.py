"""Text-to-speech adapter with deterministic fallback output."""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import MissingDependencyError

_SAMPLE_RATE = 16000
_AMP = 16000


def _synthesize_wave(text: str, *, duration_sec: float = 0.6) -> bytes:
    """Return a mono WAV byte stream encoding a simple sine wave voice."""

    duration = max(0.2, min(duration_sec, 2.0))
    frames = int(_SAMPLE_RATE * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_SAMPLE_RATE)
        for idx in range(frames):
            # encode frequency based on text length to keep outputs unique-ish
            freq = 220 + (hash(text) % 200)
            sample = int(_AMP * math.sin(2 * math.pi * freq * (idx / _SAMPLE_RATE)))
            wav_file.writeframes(sample.to_bytes(2, "little", signed=True))
    return buffer.getvalue()


def _ensure_shots(asset_refs: Dict[str, Any]) -> Dict[str, Any]:
    shots = asset_refs.setdefault("shots", {})
    if not isinstance(shots, dict):
        raise ValueError("asset_refs['shots'] must be a dict")
    return shots


def _dialogue_entries(shot: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    dialogue = shot.get("dialogue") or []
    return [d for d in dialogue if isinstance(d, dict)]


def _write_audio_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _synthesize_wave(text, duration_sec=max(0.4, len(text) * 0.05))
    path.write_bytes(data)


def generate_audio(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Generate stub WAV files per dialogue line.

    A production adapter would swap this logic with a real TTS model. We keep the
    same file layout so downstream stages can function identically in tests.
    """

    shots_refs = _ensure_shots(asset_refs)
    shots_plan = movie_plan.get("shots", []) or []
    if not isinstance(shots_plan, Iterable):
        raise ValueError("movie_plan['shots'] must be iterable")

    for shot in shots_plan:
        if not isinstance(shot, dict):
            continue
        shot_id = shot.get("id")
        if not shot_id:
            continue
        entries = list(_dialogue_entries(shot))
        if not entries:
            continue

        shot_refs = shots_refs.setdefault(shot_id, {})
        audio_paths: List[str] = []
        for line_idx, line in enumerate(entries):
            text = line.get("text") or "(silence)"
            wav_path = run_dir / f"{shot_id}_line_{line_idx:03d}.wav"
            try:
                _write_audio_file(wav_path, text)
            except Exception as exc:  # pragma: no cover - defensive
                raise MissingDependencyError(f"Failed to write stub audio: {exc}") from exc
            audio_paths.append(str(wav_path))

        if audio_paths:
            shot_refs["dialogue_audio"] = audio_paths
            shot_refs.setdefault("extras", {})["tts"] = {
                "lines": len(audio_paths),
                "sample_rate": _SAMPLE_RATE,
            }

    return asset_refs
