from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sparkle_motion.ratelimit import RateLimitDecision, RateLimiter
from sparkle_motion.utils.dedupe import RecentIndex, compute_hash

try:
    from function_tools.images_sdxl.entrypoint import render_images
except Exception:  # adapter may be stubbed in tests
    def render_images(prompt: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore
        raise RuntimeError("images_sdxl adapter not available")

try:
    from function_tools.qa_qwen2vl import entrypoint as _qa_entrypoint
except Exception:  # pragma: no cover - optional dependency
    _qa_entrypoint = None


def _inspect_frames(frames: List[bytes], prompts: List[str]) -> Dict[str, Any]:
    target = getattr(_qa_entrypoint, "inspect_frames", None)
    if callable(target):
        return target(frames, prompts)
    return {"ok": True}


class PlanPolicyViolation(RuntimeError):
    pass


class RateLimitError(RuntimeError):
    def __init__(self, message: str, decision: RateLimitDecision) -> None:
        super().__init__(message)
        self.decision = decision


class RateLimitQueued(RateLimitError):
    pass


class RateLimitExceeded(RateLimitError):
    pass


@dataclass(frozen=True)
class _BatchSpec:
    batch_index: int
    start: int
    size: int


class _PlanDeduper:
    def __init__(self, enabled: bool, recent_index: Optional[RecentIndex], uri_prefix: str = "inmem://") -> None:
        self._enabled = enabled
        self._recent_index = recent_index
        self._uri_prefix = uri_prefix
        self._plan_cache: Dict[str, str] = {}

    def apply(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        if not self._enabled:
            return artifact

        data = artifact.get("data")
        if data is None:
            return artifact

        digest = compute_hash(data)
        canonical_uri = artifact.get("uri") or f"{self._uri_prefix}{digest}"

        existing_uri = self._plan_cache.get(digest)
        if existing_uri is None and self._recent_index is not None:
            existing_uri = self._recent_index.get(digest)

        deduped = existing_uri is not None
        if deduped:
            canonical_uri = existing_uri  # reuse previously seen artifact
        else:
            if self._recent_index is not None:
                canonical_uri = self._recent_index.get_or_add(digest, canonical_uri)
            self._plan_cache[digest] = canonical_uri

        updated = dict(artifact)
        updated["uri"] = canonical_uri
        if deduped:
            updated.pop("data", None)
            meta = dict(updated.get("metadata") or {})
            meta["deduped"] = True
            updated["metadata"] = meta
            updated["duplicate_of"] = canonical_uri

        return updated


def _build_batches(count: int, max_per: int) -> List[_BatchSpec]:
    batches: List[_BatchSpec] = []
    batch_index = 0
    for start in range(0, count, max_per):
        size = min(max_per, count - start)
        batches.append(_BatchSpec(batch_index=batch_index, start=start, size=size))
        batch_index += 1
    return batches


def _coerce_frames(sources: Iterable[Any]) -> List[bytes]:
    frames: List[bytes] = []
    for source in sources:
        if isinstance(source, bytes):
            frames.append(source)
        elif isinstance(source, (str, Path, PathLike)):
            frames.append(Path(source).read_bytes())
        elif source is None:
            continue
        else:
            raise TypeError(f"Unsupported reference image type: {type(source)!r}")
    return frames


def _qa_status(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "ok"
    if report.get("reject") or report.get("status") == "reject":
        return "reject"
    if report.get("escalate") or report.get("status") == "escalate":
        return "escalate"
    frames = report.get("frames")
    if isinstance(frames, Sequence):
        for frame in frames:
            decision = str(frame.get("decision", "")).lower()
            if decision == "reject":
                return "reject"
            if decision == "escalate":
                return "escalate"
    return "ok"


def _ensure_qa_ok(report: Optional[Dict[str, Any]], *, stage: str) -> None:
    status = _qa_status(report)
    if status == "reject":
        raise PlanPolicyViolation(f"QA rejected {stage}: {report}")
    if status == "escalate":
        raise PlanPolicyViolation(f"QA escalation required for {stage}: {report}")


def _inspect_reference_images(reference_images: Iterable[Any], prompt: str) -> None:
    frames = _coerce_frames(reference_images)
    if not frames:
        return
    report = _inspect_frames(frames, [prompt] * len(frames))
    _ensure_qa_ok(report, stage="reference images")


def _evaluate_post_render_qa(prompt: str, artifacts: List[Dict[str, Any]]) -> None:
    frames: List[bytes] = []
    prompts: List[str] = []
    for artifact in artifacts:
        data = artifact.get("data")
        if isinstance(data, bytes):
            frames.append(data)
            prompts.append(prompt)
    if not frames:
        return
    report = _inspect_frames(frames, prompts)
    _ensure_qa_ok(report, stage="rendered frames")
    for artifact in artifacts:
        meta = dict(artifact.get("metadata") or {})
        meta.setdefault("qa_status", "ok")
        if report:
            meta.setdefault("qa_report", report)
        artifact["metadata"] = meta


def render(prompt: str, opts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Orchestrate image rendering with batching, optional QA and dedupe.

        opts keys (defaults):
      - count: int = 1
      - max_images_per_call: int = 8
      - seed: Optional[int]
      - rate_limiter: Optional[RateLimiter]
      - qa: bool = False
            - dedupe: bool = False
            - queue_allowed: bool = True
            - queue_ttl_s: float = 600.0
            - reference_images: Iterable[bytes | str | os.PathLike]
    Returns list of artifact dicts (ordered as requested).
    """
    if opts is None:
        opts = {}

    count = int(opts.get("count", 1))
    if count <= 0:
        raise ValueError("count must be positive")

    max_per = int(opts.get("max_images_per_call", 8))
    if max_per <= 0:
        raise ValueError("max_images_per_call must be positive")

    seed = opts.get("seed")
    rate_limiter: Optional[RateLimiter] = opts.get("rate_limiter")
    qa_enabled = bool(opts.get("qa", False))
    dedupe_enabled = bool(opts.get("dedupe", False))
    queue_allowed = bool(opts.get("queue_allowed", True))
    queue_ttl_s = float(opts.get("queue_ttl_s", 600.0))
    reference_images = opts.get("reference_images") or []

    # Use an in-memory recent index for dedupe by default
    recent = opts.get("recent_index") or (RecentIndex() if dedupe_enabled else None)
    deduper = _PlanDeduper(dedupe_enabled, recent)

    # Pre-render QA check (textual sampling); here we call a lightweight inspect hook
    if qa_enabled and reference_images:
        _inspect_reference_images(reference_images, prompt)

    ordered: List[Optional[Dict[str, Any]]] = [None for _ in range(count)]

    batches = _build_batches(count, max_per)

    for batch in batches:
        # Rate limiter hook (if provided) - ask permission to proceed
        if rate_limiter is not None:
            decision = rate_limiter.request(
                batch.size,
                queue_allowed=queue_allowed,
                ttl_s=queue_ttl_s,
            )
            if not decision.allowed:
                message = "Rate limit queued" if decision.queued else "Rate limit exceeded"
                exc_cls = RateLimitQueued if decision.queued else RateLimitExceeded
                raise exc_cls(message, decision)

        # call adapter for this batch
        batch_opts = {**opts, "count": batch.size, "seed": seed, "batch_start": batch.start}
        batch_results = render_images(prompt, batch_opts)

        if len(batch_results) != batch.size:
            raise RuntimeError(
                f"images_sdxl adapter returned {len(batch_results)} items for batch size {batch.size}"
            )

        if qa_enabled:
            _evaluate_post_render_qa(prompt, batch_results)

        # normalize and append preserving ordering
        for item_idx, item in enumerate(batch_results):
            # item is expected to be a dict with at least 'data' (bytes) and 'metadata'
            meta = dict(item.get("metadata", {}))
            global_index = batch.start + item_idx
            meta.update({"global_index": global_index, "batch_index": batch.batch_index, "item_index": item_idx, "batch_start": batch.start})

            artifact = {"metadata": meta}
            if "data" in item:
                artifact["data"] = item["data"]
            if "uri" in item:
                artifact["uri"] = item["uri"]

            artifact = deduper.apply(artifact)
            ordered[global_index] = artifact

    # typing helper (all entries filled)
    return [art for art in ordered if art is not None]


__all__ = ["render", "PlanPolicyViolation", "RateLimitExceeded", "RateLimitQueued"]
