from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sparkle_motion.ratelimit import RateLimitDecision, RateLimiter
from sparkle_motion.utils import dedupe

try:
    from function_tools.images_sdxl.entrypoint import render_images
except Exception:  # adapter may be stubbed in tests
    def render_images(prompt: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore
        raise RuntimeError("images_sdxl adapter not available")

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
    def __init__(
        self,
        enabled: bool,
        recent_index: Optional[dedupe.RecentIndexBackend],
        uri_prefix: str = "inmem://",
        phash_threshold: int = dedupe.DEFAULT_PHASH_DISTANCE_THRESHOLD,
    ) -> None:
        self._enabled = enabled
        self._recent_index = recent_index
        self._uri_prefix = uri_prefix
        self._phash_threshold = max(0, phash_threshold)
        self._plan_cache: Dict[str, str] = {}
        self._plan_phashes: list[tuple[str, str]] = []

    def apply(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        if not self._enabled:
            return artifact

        data = artifact.get("data")
        if data is None:
            return artifact

        metadata = artifact.get("metadata")
        meta_dict = metadata if isinstance(metadata, dict) else {}
        phash_value = meta_dict.get("phash")
        phash = str(phash_value).strip() if phash_value else None

        digest = dedupe.compute_hash(data)
        canonical_uri = artifact.get("uri") or f"{self._uri_prefix}{digest}"

        existing_uri = self._find_plan_match(phash)
        if existing_uri is None:
            existing_uri = self._plan_cache.get(digest)

        if existing_uri is None and self._recent_index is not None:
            key = phash or digest
            existing_uri = self._recent_index.get(key)

        deduped = existing_uri is not None
        if deduped:
            canonical_uri = existing_uri  # reuse previously seen artifact
        else:
            if self._recent_index is not None:
                key = phash or digest
                canonical_uri = self._recent_index.get_or_add(key, canonical_uri)
            self._plan_cache[digest] = canonical_uri
            if phash:
                self._remember_phash(phash, canonical_uri)

        updated = dict(artifact)
        updated["uri"] = canonical_uri
        if deduped:
            updated.pop("data", None)
            meta = dict(updated.get("metadata") or {})
            meta["deduped"] = True
            updated["metadata"] = meta
            updated["duplicate_of"] = canonical_uri
            if phash:
                self._remember_phash(phash, canonical_uri)
            self._plan_cache.setdefault(digest, canonical_uri)

        return updated

    def _find_plan_match(self, phash: Optional[str]) -> Optional[str]:
        if not phash:
            return None
        for existing_phash, uri in self._plan_phashes:
            try:
                if dedupe.hamming_distance(existing_phash, phash) <= self._phash_threshold:
                    return uri
            except ValueError:
                continue
        return None

    def _remember_phash(self, phash: str, canonical_uri: str) -> None:
        self._plan_phashes.append((phash, canonical_uri))


def _build_batches(count: int, max_per: int) -> List[_BatchSpec]:
    batches: List[_BatchSpec] = []
    batch_index = 0
    for start in range(0, count, max_per):
        size = min(max_per, count - start)
        batches.append(_BatchSpec(batch_index=batch_index, start=start, size=size))
        batch_index += 1
    return batches


def render(prompt: str, opts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Orchestrate image rendering with batching and optional dedupe.

    opts keys (defaults):
      - count: int = 1
      - max_images_per_call: int = 8
      - seed: Optional[int]
      - rate_limiter: Optional[RateLimiter]
      - dedupe: bool = False
      - dedupe_phash_threshold: int = DEFAULT_PHASH_DISTANCE_THRESHOLD
      - queue_allowed: bool = True
      - queue_ttl_s: float = 600.0
      - qa (deprecated): bool = False (ignored)
      - reference_images (deprecated): Iterable[bytes | str | os.PathLike] (ignored)

    Returns list of artifact dicts (ordered as requested).
    """
    if opts is None:
        opts = {}
    else:
        opts = dict(opts)

    qa_requested = bool(opts.pop("qa", False))
    reference_images = opts.pop("reference_images", None)
    if qa_requested or reference_images:
        # QA enforcement is temporarily disabled; keep these options for backward compatibility.
        pass

    count = int(opts.get("count", 1))
    if count <= 0:
        raise ValueError("count must be positive")

    max_per = int(opts.get("max_images_per_call", 8))
    if max_per <= 0:
        raise ValueError("max_images_per_call must be positive")

    seed = opts.get("seed")
    rate_limiter: Optional[RateLimiter] = opts.get("rate_limiter")
    dedupe_enabled = bool(opts.get("dedupe", False))
    phash_threshold_opt = opts.get("dedupe_phash_threshold")
    if phash_threshold_opt is None:
        dedupe_phash_threshold = dedupe.DEFAULT_PHASH_DISTANCE_THRESHOLD
    else:
        try:
            dedupe_phash_threshold = int(phash_threshold_opt)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("dedupe_phash_threshold must be an integer") from exc
        if dedupe_phash_threshold < 0:
            raise ValueError("dedupe_phash_threshold must be >= 0")
    queue_allowed = bool(opts.get("queue_allowed", True))
    queue_ttl_s = float(opts.get("queue_ttl_s", 600.0))

    recent = dedupe.resolve_recent_index(
        enabled=dedupe_enabled,
        backend=opts.get("recent_index"),
        use_sqlite=opts.get("recent_index_use_sqlite"),
        db_path=opts.get("recent_index_db_path"),
    )
    deduper = _PlanDeduper(
        dedupe_enabled,
        recent,
        phash_threshold=dedupe_phash_threshold,
    )

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


__all__ = ["render", "RateLimitExceeded", "RateLimitQueued"]
