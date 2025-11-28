from __future__ import annotations
from typing import Any, Dict, List, Optional

from sparkle_motion.ratelimit import RateLimiter
from sparkle_motion.utils.dedupe import RecentIndex, compute_hash

try:
    from function_tools.images_sdxl.entrypoint import render_images
except Exception:  # adapter may be stubbed in tests
    def render_images(prompt: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore
        raise RuntimeError("images_sdxl adapter not available")

try:
    from function_tools.qa_qwen2vl.entrypoint import inspect_frames
except Exception:
    def inspect_frames(frames: List[bytes], prompts: List[str]) -> Dict[str, Any]:  # type: ignore
        return {"ok": True}


class PlanPolicyViolation(RuntimeError):
    pass


def render(prompt: str, opts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Orchestrate image rendering with batching, optional QA and dedupe.

    opts keys (defaults):
      - count: int = 1
      - max_images_per_call: int = 8
      - seed: Optional[int]
      - rate_limiter: Optional[RateLimiter]
      - qa: bool = False
      - dedupe: bool = False
    Returns list of artifact dicts (ordered as requested).
    """
    if opts is None:
        opts = {}

    count = int(opts.get("count", 1))
    max_per = int(opts.get("max_images_per_call", 8))
    seed = opts.get("seed")
    rate_limiter: Optional[RateLimiter] = opts.get("rate_limiter")
    qa_enabled = bool(opts.get("qa", False))
    dedupe_enabled = bool(opts.get("dedupe", False))

    # Use an in-memory recent index for dedupe by default
    recent = opts.get("recent_index") or (RecentIndex() if dedupe_enabled else None)

    # Pre-render QA check (textual sampling); here we call a lightweight inspect hook
    if qa_enabled:
        qa_report = inspect_frames([], [prompt])
        if qa_report.get("reject"):
            raise PlanPolicyViolation("Prompt rejected by QA: %s" % qa_report)

    artifacts: List[Dict[str, Any]] = []
    global_index = 0

    # split into batches respecting max_per
    batches = [(i, min(max_per, count - i)) for i in range(0, count, max_per)]

    for batch_start, batch_size in batches:
        # Rate limiter hook (if provided) - ask permission to proceed
        if rate_limiter is not None:
            allowed = rate_limiter.allow(batch_size)
            if not allowed:
                raise RuntimeError("Rate limit exceeded")

        # call adapter for this batch
        batch_opts = {**opts, "count": batch_size, "seed": seed, "batch_start": batch_start}
        batch_results = render_images(prompt, batch_opts)

        # normalize and append preserving ordering
        for item_idx, item in enumerate(batch_results):
            # item is expected to be a dict with at least 'data' (bytes) and 'metadata'
            data = item.get("data")
            meta = dict(item.get("metadata", {}))
            meta.update({"global_index": global_index, "batch_index": batch_start, "item_index": item_idx})

            if dedupe_enabled and data is not None and recent is not None:
                h = compute_hash(data)
                canonical_uri = recent.get_or_add(h, f"inmem://{h}")
                artifact = {"uri": canonical_uri, "metadata": meta}
                # mark deduped when canonical existed
                if canonical_uri != f"inmem://{h}":
                    artifact["deduped"] = True
            else:
                artifact = {"data": data, "metadata": meta}

            artifacts.append(artifact)
            global_index += 1

    return artifacts


__all__ = ["render", "PlanPolicyViolation"]
