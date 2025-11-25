"""Wan adapter with deterministic stub fallback.

The real implementation should call Wan's Flow Matching video generator, but for
local development we synthesize short placeholder clips using the shared
``StubAdapter``. This keeps the orchestrator functional without heavyweight
dependencies while still providing realistic file outputs and metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import MissingDependencyError
from .stub_adapter import StubAdapter, get_stub_adapter


def _load_frame_bytes(path: str | Path | None) -> bytes | None:
  if not path:
    return None
  try:
    return Path(path).read_bytes()
  except OSError:
    return None


def _prompt_frames(adapter: StubAdapter, prompt: str, count: int = 2) -> List[bytes]:
  assets = adapter.generate_images(prompt=prompt, count=count)
  return [asset.data for asset in assets]


def _collect_frames(
  *,
  adapter: StubAdapter,
  shot_refs: Dict[str, Any],
  prompt: str,
) -> List[bytes]:
  frame_candidates: List[bytes] = []
  for key in ("start_frame", "end_frame"):
    data = _load_frame_bytes(shot_refs.get(key))
    if data:
      frame_candidates.append(data)
  if frame_candidates:
    return frame_candidates
  return _prompt_frames(adapter, prompt or "wan_stub_frame", count=2)


def _ensure_shots(asset_refs: Dict[str, Any]) -> Dict[str, Any]:
  shots = asset_refs.setdefault("shots", {})
  if not isinstance(shots, dict):
    raise ValueError("asset_refs['shots'] must be a dict")
  return shots


def generate_video(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
  """Generate raw video clips for each shot.

  When Wan is unavailable, we fall back to :class:`StubAdapter` which renders a
  short MP4 per shot. The returned ``asset_refs`` is mutated in-place and also
  contains lightweight metadata for observability.
  """

  adapter = get_stub_adapter()
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
    description = shot.get("visual_description") or f"Shot {shot_id}"
    shot_refs = shots_refs.setdefault(shot_id, {})
    frames = _collect_frames(adapter=adapter, shot_refs=shot_refs, prompt=description)

    out_path = run_dir / f"{shot_id}_raw.mp4"
    try:
      result = adapter.render_sequence(
        frames,
        out_path=out_path,
        fps=max(1, int(shot.get("fps", 12))),
        add_audio=False,
      )
    except Exception as exc:  # pragma: no cover - defensive
      raise MissingDependencyError(f"Wan stub rendering failed: {exc}") from exc

    shot_refs["raw_clip"] = str(result.path)
    extras = shot_refs.setdefault("extras", {})
    extras["wan"] = {
      "duration_sec": result.duration,
      "video_codec": result.video_codec,
      "audio_codec": result.audio_codec,
      "frames_used": len(frames),
    }

  return asset_refs
