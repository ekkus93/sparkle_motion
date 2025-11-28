from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class WanAdapter:
    """Scaffold adapter for Wan2.1 I2V/T2V models.

    Notes:
    - Do not import heavy libraries at module import time. Perform lazy
      imports in `load()` / `run()`.
    - The real implementation should use Diffusers' `WanPipeline` or the
      Wan repo's `generate.py` example.
    """
    weights: Optional[str] = None
    model: Optional[object] = None

    def load(self) -> None:
        """Lazy-load model. Keep this lightweight for unit tests.

        Real implementation: from diffusers import WanPipeline / WanImageToVideoPipeline
        and call `.from_pretrained(weights, subfolder=...)` then `.to('cuda')`.
        """
        # Lazy import placeholder
        try:
            # Example (real code commented):
            # from diffusers import WanImageToVideoPipeline
            # self.model = WanImageToVideoPipeline.from_pretrained(self.weights)
            self.model = object()
        except Exception:
            self.model = None

    def run(self, start_frames, end_frames, prompt: str, out_path: Path) -> Path:
        """Run inference and write an MP4 to `out_path`.

        In this scaffold, we create a small fake file so unit tests can assert
        outputs without real model downloads.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # If not loaded, create a fake artifact for tests
        if self.model is None:
            out_path.write_bytes(b"FAKE_MP4")
            return out_path

        # TODO: real pipeline inference here
        out_path.write_bytes(b"WAN_REALISTIC_MP4_BYTES")
        return out_path

    def unload(self) -> None:
        """Release resources (CUDA memory, close pipelines)."""
        self.model = None
