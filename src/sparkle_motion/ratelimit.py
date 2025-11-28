from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Optional


class RateLimiter:
    """Pluggable rate limiter interface."""

    def allow(self, tokens: int = 1) -> bool:
        raise NotImplementedError()


@dataclass
class NoopRateLimiter(RateLimiter):
    def allow(self, tokens: int = 1) -> bool:  # pragma: no cover - trivial
        return True


class TokenBucketRateLimiter(RateLimiter):
    """Simple in-memory token bucket for testing and lightweight use.

    Not distributed. Intended for tests and single-process dev use.
    """

    def __init__(self, rate: float, capacity: int, now: Optional[float] = None) -> None:
        self.rate = float(rate)
        self.capacity = int(capacity)
        self._tokens = float(capacity)
        self._last = now if now is not None else time.monotonic()

    def allow(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        delta = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + delta * self.rate)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


__all__ = ["RateLimiter", "NoopRateLimiter", "TokenBucketRateLimiter"]
