from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Any, Dict, Optional, Protocol

import imagehash
from PIL import Image

_TRUTHY = {"1", "true", "yes", "on"}
_SQLITE_ENV_FLAG = "SPARKLE_RECENT_INDEX_SQLITE"

PHASH_HEX_LENGTH = 16
DEFAULT_PHASH_DISTANCE_THRESHOLD = 6


class RecentIndexBackend(Protocol):
    """Protocol for objects that can store/retrieve canonical artifact URIs."""

    def get(self, digest: str) -> Optional[str]:
        ...

    def get_or_add(self, digest: str, canonical: str) -> str:
        ...


def compute_hash(data: bytes) -> str:
    """Compute a stable hex hash for the given bytes (used for dedupe canonicalization)."""
    return hashlib.sha256(data).hexdigest()


def compute_phash(pixels: Sequence[Any], width: int, height: int) -> str:
    """Compute a perceptual hash using the ImageHash implementation."""

    total_pixels = width * height
    if width <= 0 or height <= 0 or total_pixels <= 0 or not pixels:
        return "0" * PHASH_HEX_LENGTH

    rgb_pixels = _normalize_pixels(pixels, total_pixels)
    image = Image.new("RGB", (width, height))
    image.putdata(rgb_pixels)
    phash = imagehash.phash(image, hash_size=8)
    return _hash_to_hex(phash)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Return the Hamming distance between two hexadecimal perceptual hashes."""

    if not hash_a or not hash_b:
        raise ValueError("hash strings must be non-empty")

    length = max(len(hash_a), len(hash_b))
    padded_a = hash_a.rjust(length, "0")
    padded_b = hash_b.rjust(length, "0")

    try:
        value_a = int(padded_a, 16)
        value_b = int(padded_b, 16)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("hash strings must be hexadecimal") from exc

    return (value_a ^ value_b).bit_count()


_PIXEL_SEQUENCE_TYPES = (bytes, bytearray, str)


def _normalize_pixels(pixels: Sequence[Any], expected: int) -> list[tuple[int, int, int]]:
    normalized: list[tuple[int, int, int]] = []
    last_pixel = (0, 0, 0)
    for pixel in pixels:
        rgb = _coerce_pixel(pixel)
        normalized.append(rgb)
        last_pixel = rgb
        if len(normalized) >= expected:
            break
    if len(normalized) < expected:
        normalized.extend([last_pixel] * (expected - len(normalized)))
    return normalized[:expected]


def _coerce_pixel(pixel: Any) -> tuple[int, int, int]:
    if isinstance(pixel, Sequence) and not isinstance(pixel, _PIXEL_SEQUENCE_TYPES):
        components = list(pixel)
    else:
        components = [int(pixel)]
    if not components:
        return (0, 0, 0)
    r = _clamp_channel(components[0])
    g = _clamp_channel(components[1] if len(components) > 1 else components[0])
    b = _clamp_channel(components[2] if len(components) > 2 else components[-1])
    return (r, g, b)


def _clamp_channel(value: int) -> int:
    return max(0, min(255, int(value)))


def _hash_to_hex(value: imagehash.ImageHash) -> str:
    text = str(value)
    if len(text) >= PHASH_HEX_LENGTH:
        return text[:PHASH_HEX_LENGTH]
    return text.rjust(PHASH_HEX_LENGTH, "0")


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
    register_new: bool = True,
) -> tuple[str, bool]:
    """Return the canonical URI for a digest plus whether it was deduped.

    When ``register_new`` is False the function will not create a new index entry,
    which is useful for pre-publish duplicate checks where the artifact does not
    exist yet.
    """

    if recent_index is None:
        return candidate_uri, False

    existing = recent_index.get(digest)
    if existing is not None:
        # touch the entry to keep hit counts fresh
        recent_index.get_or_add(digest, existing)
        return existing, True

    if not register_new:
        return candidate_uri, False

    recent_index.get_or_add(digest, candidate_uri)
    return candidate_uri, False


__all__ = [
    "RecentIndexBackend",
    "RecentIndex",
    "compute_hash",
    "compute_phash",
    "hamming_distance",
    "PHASH_HEX_LENGTH",
    "DEFAULT_PHASH_DISTANCE_THRESHOLD",
    "resolve_recent_index",
    "canonicalize_digest",
]
