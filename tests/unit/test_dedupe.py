from __future__ import annotations

import pytest

from sparkle_motion.utils.dedupe import PHASH_HEX_LENGTH, compute_phash, hamming_distance


def _make_gradient(width: int, height: int) -> list[tuple[int, int, int]]:
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            value = (x * 7 + y * 13) % 256
            pixels.append((value, value, value))
    return pixels


def _offset_pixels(pixels: list[tuple[int, int, int]], delta: int) -> list[tuple[int, int, int]]:
    adjusted: list[tuple[int, int, int]] = []
    for r, g, b in pixels:
        value = _clamp(r + delta)
        adjusted.append((value, value, value))
    return adjusted


def _invert_pixels(pixels: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(255 - r, 255 - g, 255 - b) for r, g, b in pixels]


def _clamp(value: int) -> int:
    return max(0, min(255, value))



def test_compute_phash_handles_empty_input() -> None:
    assert compute_phash([], 0, 0) == "0" * PHASH_HEX_LENGTH


def test_compute_phash_consistent_for_same_pixels() -> None:
    pixels = _make_gradient(32, 32)
    phash_a = compute_phash(pixels, 32, 32)
    phash_b = compute_phash(list(pixels), 32, 32)
    assert phash_a == phash_b


def test_compute_phash_small_brightness_change_results_in_small_distance() -> None:
    base = _make_gradient(32, 32)
    brighter = _offset_pixels(base, 5)
    phash_base = compute_phash(base, 32, 32)
    phash_brighter = compute_phash(brighter, 32, 32)
    assert hamming_distance(phash_base, phash_brighter) <= 8


def test_compute_phash_detects_large_difference() -> None:
    base = _make_gradient(32, 32)
    inverted = _invert_pixels(base)
    phash_base = compute_phash(base, 32, 32)
    phash_inverted = compute_phash(inverted, 32, 32)
    assert hamming_distance(phash_base, phash_inverted) >= 20


def test_hamming_distance_rejects_invalid_hex() -> None:
    with pytest.raises(ValueError):
        hamming_distance("zz", "11")


def test_hamming_distance_zero_for_equal_values() -> None:
    value = "abcd1234abcd1234"
    assert hamming_distance(value, value) == 0
