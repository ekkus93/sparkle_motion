from __future__ import annotations
from typing import Any, Dict, List
import hashlib

def _deterministic_bytes(prompt: str, seed: Any, index: int) -> bytes:
    s = f"{prompt}|{seed}|{index}"
    return hashlib.sha256(s.encode("utf-8")).digest()


def render_images(prompt: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic stub adapter for images. Returns list of dicts with 'data' bytes.

    opts: expects 'count' and optional 'seed' and 'batch_start'
    """
    count = int(opts.get("count", 1))
    seed = opts.get("seed", 0)
    batch_start = int(opts.get("batch_start", 0))

    out = []
    for i in range(count):
        idx = batch_start + i
        # special-case to allow tests to create duplicates when prompt contains 'duplicate'
        if "duplicate" in str(prompt) and idx % 2 == 0:
            data = _deterministic_bytes(prompt, seed, 0)
        else:
            data = _deterministic_bytes(prompt, seed, idx)

        meta = {"prompt": prompt, "seed": seed, "index": idx}
        out.append({"data": data, "metadata": meta})
    return out


__all__ = ["render_images"]
