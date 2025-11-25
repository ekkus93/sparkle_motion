"""Assembly adapter using the shared stub implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import MissingDependencyError
from .stub_adapter import get_stub_adapter


def _ordered_shots(movie_plan: Dict[str, Any]) -> List[str]:
    shots = movie_plan.get("shots", []) or []
    if not isinstance(shots, Iterable):
        return []
    ordered: List[str] = []
    for shot in shots:
        if isinstance(shot, dict) and shot.get("id"):
            ordered.append(shot["id"])
    return ordered


def assemble(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Concatenate per-shot clips into a final movie placeholder."""

    adapter = get_stub_adapter()
    shots_refs = asset_refs.setdefault("shots", {})
    out_path = run_dir / "movie_final.mp4"

    ordered_ids = _ordered_shots(movie_plan)
    clip_entries = []
    for shot_id in ordered_ids:
        shot_entry = shots_refs.get(shot_id) or {}
        clip_path = shot_entry.get("final_video_clip") or shot_entry.get("raw_clip")
        if clip_path:
            clip_entries.append({"id": shot_id, "clip": clip_path})

    if not clip_entries:
        raise MissingDependencyError("No clips available to assemble; upstream stages must run first.")

    try:
        result = adapter.assemble(shots=clip_entries, out_path=out_path, add_audio=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise MissingDependencyError(f"Stub assembly failed: {exc}") from exc

    extras = asset_refs.setdefault("extras", {})
    extras["final_movie"] = str(result.path)
    extras["final_movie_metadata"] = {
        "duration_sec": result.duration,
        "video_codec": result.video_codec,
        "audio_codec": result.audio_codec,
    }
    return asset_refs
