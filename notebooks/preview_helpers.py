"""Shared helpers for rendering production-agent artifacts inside notebooks.

These functions keep the Colab control panel, artifacts viewer, and
final-deliverable helper cells in sync so media previews look identical
throughout the control surface.
"""
from __future__ import annotations

from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, Iterable

import ipywidgets as widgets
import requests
from IPython.display import Audio, Video, display

DEFAULT_TIMEOUT_S = 15.0

__all__ = [
    "fetch_stage_manifest",
    "render_stage_summary",
    "create_image_widget",
    "create_audio_widget",
    "create_video_widget",
    "display_artifact_previews",
]


def fetch_stage_manifest(
    *,
    base_url: str,
    run_id: str,
    stage: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """Retrieve the structured manifest for a single stage via /artifacts.

    Returns the dictionary stored under payload["stages"][0] so callers can
    inspect counts, preview metadata, and the flattened artifact list.
    """

    response = requests.get(
        f"{base_url.rstrip('/')}/artifacts",
        params={"run_id": run_id, "stage": stage},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    stages = payload.get("stages", [])
    if not stages:
        raise RuntimeError(f"Stage '{stage}' returned no manifest entries for run {run_id}.")
    return stages[0]


def render_stage_summary(stage_manifest: Dict[str, Any]) -> str:
    """Return a one-line summary describing how many artifacts are available."""

    count = stage_manifest.get("count", 0)
    media_types = ", ".join(stage_manifest.get("media_types", [])) or "unknown media"
    artifact_types = ", ".join(stage_manifest.get("artifact_types", [])) or "unknown artifacts"
    stage_status = stage_manifest.get("status") or stage_manifest.get("state")
    status_note = f" | status: {stage_status}" if stage_status else ""
    return f"{count} artifact(s) [{artifact_types}] ({media_types}){status_note}"


def _require_local_path(entry: Dict[str, Any]) -> Path:
    path = Path(entry.get("local_path", "")).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact {entry.get('label') or entry.get('artifact_type', '<unknown>')} "
            f"missing local_path {path!s}. Download it before previewing."
        )
    return path


def create_image_widget(entry: Dict[str, Any], *, width: int = 320) -> widgets.Image:
    """Build an ipywidgets.Image for an artifact manifest entry."""

    local_path = _require_local_path(entry)
    data = local_path.read_bytes()
    fmt = (entry.get("metadata", {}) or {}).get("mime_type")
    if not fmt:
        fmt = local_path.suffix.lstrip(".") or "png"
    return widgets.Image(value=b64encode(data), format=fmt, width=width)


def create_audio_widget(entry: Dict[str, Any], *, autoplay: bool = False) -> Audio:
    """Build an IPython.display.Audio widget for a manifest entry."""

    local_path = _require_local_path(entry)
    return Audio(filename=str(local_path), autoplay=autoplay)


def create_video_widget(entry: Dict[str, Any], *, width: int = 640) -> Video:
    """Build an IPython.display.Video widget for a manifest entry."""

    local_path = _require_local_path(entry)
    return Video(filename=str(local_path), embed=True, width=width)


def display_artifact_previews(
    stage_manifest: Dict[str, Any],
    *,
    max_items: int | None = None,
    image_width: int = 320,
    video_width: int = 640,
    autoplay_audio: bool = False,
) -> None:
    """Display inline previews for artifacts based on their media_type field."""

    artifacts: Iterable[Dict[str, Any]] = stage_manifest.get("artifacts", [])
    if max_items is not None:
        artifacts = list(artifacts)[:max_items]

    for entry in artifacts:
        media_type = entry.get("media_type")
        label = entry.get("label") or entry.get("artifact_type") or "artifact"
        print(f"\n{label}")
        try:
            if media_type == "image":
                display(create_image_widget(entry, width=image_width))
            elif media_type == "audio":
                display(create_audio_widget(entry, autoplay=autoplay_audio))
            elif media_type == "video":
                display(create_video_widget(entry, width=video_width))
            else:
                local_path = _require_local_path(entry)
                print(f"Preview unavailable for media_type='{media_type}'. File: {local_path}")
        except FileNotFoundError as exc:
            print(exc)