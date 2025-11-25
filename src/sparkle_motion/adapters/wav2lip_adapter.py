"""Wav2Lip adapter with a deterministic stub fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import MissingDependencyError
from .stub_adapter import get_stub_adapter


def _ensure_shots(asset_refs: Dict[str, Any]) -> Dict[str, Any]:
    shots = asset_refs.setdefault("shots", {})
    if not isinstance(shots, dict):
        raise ValueError("asset_refs['shots'] must be a dict")
    return shots


def _shot_entries(movie_plan: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    shots = movie_plan.get("shots", []) or []
    return [shot for shot in shots if isinstance(shot, dict)]


def lipsync(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Apply stubbed lip-sync to each shot.

    We attach audio when dialogue clips exist; otherwise the stub renders a
    silent MP4. Production deployments can replace this with a real Wav2Lip
    pipeline without changing the downstream interface.
    """

    adapter = get_stub_adapter()
    shots_refs = _ensure_shots(asset_refs)

    for shot in _shot_entries(movie_plan):
        shot_id = shot.get("id")
        if not shot_id:
            continue
        shot_refs = shots_refs.setdefault(shot_id, {})
        audio_paths: List[str] = shot_refs.get("dialogue_audio", []) or []
        add_audio = bool(audio_paths)

        out_path = run_dir / f"{shot_id}_final.mp4"
        try:
            result = adapter.assemble(
                shots=[{"id": shot_id, "raw_clip": shot_refs.get("raw_clip")}],
                out_path=out_path,
                add_audio=add_audio,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise MissingDependencyError(f"Wav2Lip stub assembly failed: {exc}") from exc

        shot_refs["final_video_clip"] = str(result.path)
        shot_refs.setdefault("extras", {})["wav2lip"] = {
            "audio_attached": bool(result.audio_included),
            "duration_sec": result.duration,
            "video_codec": result.video_codec,
            "audio_codec": result.audio_codec,
        }

    return asset_refs
