from __future__ import annotations
from typing import Any, Dict, List

from sparkle_motion.function_tools.images_sdxl import adapter


def render_images(prompt: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compatibility wrapper around the shared adapter implementation."""

    results = adapter.render_images(prompt, opts)
    payload: List[Dict[str, Any]] = []
    for item in results:
        payload.append(
            {
                "data": item.data,
                "metadata": dict(item.metadata),
                "local_path": str(item.path),
            }
        )
    return payload


__all__ = ["render_images"]
