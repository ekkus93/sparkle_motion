from __future__ import annotations
import hashlib
from typing import Dict


def compute_hash(data: bytes) -> str:
    """Compute a stable hex hash for the given bytes (used for dedupe canonicalization)."""
    return hashlib.sha256(data).hexdigest()


class RecentIndex:
    """Simple in-memory canonical index mapping hash -> canonical uri.

    Not distributed. Designed for unit tests and local dev.
    """

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}

    def get(self, h: str) -> str | None:
        return self._map.get(h)

    def get_or_add(self, h: str, canonical: str) -> str:
        existing = self._map.get(h)
        if existing is not None:
            return existing
        self._map[h] = canonical
        return canonical


__all__ = ["compute_hash", "RecentIndex"]
