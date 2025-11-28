from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class DiffusersAdapter:
    """Scaffold adapter for image generation (SDXL / Diffusers).

    Real implementation should use `diffusers` pipelines and respect the
    `model_context` from `gpu_utils` for load/unload semantics.
    """
    weights: Optional[str] = None
    pipeline: Optional[Any] = None

    def load(self) -> None:
        try:
            # Real code example (commented):
            # from diffusers import StableDiffusionPipeline
            # self.pipeline = StableDiffusionPipeline.from_pretrained(self.weights)
            self.pipeline = object()
        except Exception:
            self.pipeline = None

    def render_images(self, prompt: str, opts: Optional[dict] = None) -> List[Path]:
        out = []
        # Create small PNG placeholders for unit tests
        for i in range(1):
            p = Path("artifacts") / "images" / f"{prompt[:10]}_{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PNG" + bytes([i]))
            out.append(p)
        return out

    def unload(self) -> None:
        self.pipeline = None
