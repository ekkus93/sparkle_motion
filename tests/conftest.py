"""Test configuration helpers."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    for entry in (src, root):
        entry_str = str(entry)
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)


def _ensure_pythonpath_env() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    entries = [str(src), str(root)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        if entry and entry not in seen:
            ordered.append(entry)
            seen.add(entry)
    os.environ["PYTHONPATH"] = os.pathsep.join(ordered)


@dataclass(frozen=True)
class MediaAssets:
    """Container for deterministic media fixture paths."""

    image: Path
    audio: Path
    video: Path
    plan: Path


@pytest.fixture(scope="session")
def deterministic_media_assets() -> MediaAssets:
    """Expose shared deterministic fixture files for any test that needs media."""

    root = Path(__file__).resolve().parents[1]
    assets_dir = root / "tests" / "fixtures" / "assets"
    missing = [entry for entry in ("sample_image.png", "sample_audio.wav", "sample_video.mp4", "sample_plan.json") if not (assets_dir / entry).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Deterministic media assets missing: {joined}. Run PYTHONPATH=src python scripts/generate_fixture_assets.py")
    return MediaAssets(
        image=assets_dir / "sample_image.png",
        audio=assets_dir / "sample_audio.wav",
        video=assets_dir / "sample_video.mp4",
        plan=assets_dir / "sample_plan.json",
    )


_ensure_src_on_path()
_ensure_pythonpath_env()
