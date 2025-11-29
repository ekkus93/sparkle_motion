from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Protocol, Sequence

_TRUTHY = {"1", "true", "yes", "on"}
_SQLITE_ENV_FLAG = "SPARKLE_RECENT_INDEX_SQLITE"


class RecentIndexBackend(Protocol):
    """Protocol for objects that can store/retrieve canonical artifact URIs."""

    def get(self, digest: str) -> Optional[str]:
        ...

    def get_or_add(self, digest: str, canonical: str) -> str:
        ...


def compute_hash(data: bytes) -> str:
    """Compute a stable hex hash for the given bytes (used for dedupe canonicalization)."""
    return hashlib.sha256(data).hexdigest()


def compute_phash(pixels: Sequence[Sequence[int]], width: int, height: int) -> str:
    """Compute a perceptual hash (average hash) for a flattened RGB pixel buffer."""

    if width <= 0 or height <= 0 or not pixels:
        return "0" * 16

    grayscale = [int(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in pixels]
    sample: list[int] = []
    for block_y in range(8):
        src_y = min(int(block_y * height / 8), height - 1)
        for block_x in range(8):
            src_x = min(int(block_x * width / 8), width - 1)
            sample.append(grayscale[src_y * width + src_x])
    avg = sum(sample) / len(sample)
    bits = 0
    for value in sample:
        bits = (bits << 1) | (1 if value >= avg else 0)
    return f"{bits:016x}"


class RecentIndex(RecentIndexBackend):
    """Simple in-memory canonical index mapping hash -> canonical URI."""

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}

    def get(self, digest: str) -> Optional[str]:
        return self._map.get(digest)

    def get_or_add(self, digest: str, canonical: str) -> str:
        existing = self._map.get(digest)
        if existing is not None:
            return existing
        self._map[digest] = canonical
        return canonical


def resolve_recent_index(
    *,
    enabled: bool,
    backend: Optional[RecentIndexBackend] = None,
    use_sqlite: Optional[bool] = None,
    db_path: Optional[str] = None,
    env_flag: str = _SQLITE_ENV_FLAG,
) -> Optional[RecentIndexBackend]:
    """Return either the provided backend, a SQLite store, or an in-memory index."""

    if not enabled:
        return None
    if backend is not None:
        return backend

    if use_sqlite is None:
        env_value = os.environ.get(env_flag, "")
        use_sqlite = env_value.strip().lower() in _TRUTHY or db_path is not None

    if use_sqlite:
        from .recent_index_sqlite import RecentIndexSqlite

        return RecentIndexSqlite(db_path)

    return RecentIndex()


def canonicalize_digest(
    *,
    digest: str,
    recent_index: Optional[RecentIndexBackend],
    candidate_uri: str,
) -> tuple[str, bool]:
    """Return the canonical URI for a digest plus whether it was deduped."""

    if recent_index is None:
        return candidate_uri, False
    existing = recent_index.get(digest)
    if existing is not None:
        return existing, True
    recent_index.get_or_add(digest, candidate_uri)
    return candidate_uri, False


__all__ = [
    "RecentIndexBackend",
    "RecentIndex",
    "compute_hash",
    "compute_phash",
    "resolve_recent_index",
    "canonicalize_digest",
]
