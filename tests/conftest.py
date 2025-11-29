"""Test configuration helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path


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


_ensure_src_on_path()
_ensure_pythonpath_env()
