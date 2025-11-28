from __future__ import annotations
from typing import Any, Dict, List

def inspect_frames(frames: List[bytes], prompts: List[str]) -> Dict[str, Any]:
    """Lightweight QA stub that accepts frames/prompts and returns a QA report.

    Tests may monkeypatch this function to simulate rejections.
    """
    return {"ok": True}


__all__ = ["inspect_frames"]
