from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Literal, Optional


@dataclass(frozen=True)
class RateLimitDecision:
    status: Literal["allowed", "queued", "rejected"]
    tokens: int
    retry_after_s: Optional[float] = None
    eta_epoch_s: Optional[float] = None
    ttl_deadline_s: Optional[float] = None
    reason: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.status == "allowed"

    @property
    def queued(self) -> bool:
        return self.status == "queued"


class RateLimiter:
    """Pluggable rate limiter interface with optional queue hints."""

    def allow(self, tokens: int = 1) -> bool:
        return self.request(tokens=tokens, queue_allowed=False).allowed

    def request(self, tokens: int = 1, *, queue_allowed: bool = False, ttl_s: float = 600.0) -> RateLimitDecision:
        raise NotImplementedError()


@dataclass
class NoopRateLimiter(RateLimiter):
    def allow(self, tokens: int = 1) -> bool:  # pragma: no cover - trivial
        return True

    def request(self, tokens: int = 1, *, queue_allowed: bool = False, ttl_s: float = 600.0) -> RateLimitDecision:
        return RateLimitDecision(status="allowed", tokens=tokens)


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
        return self._consume(tokens, now=time.monotonic())

    def request(
        self,
        tokens: int = 1,
        *,
        queue_allowed: bool = False,
        ttl_s: float = 600.0,
    ) -> RateLimitDecision:
        now_monotonic = time.monotonic()
        now_epoch = time.time()
        if self._consume(tokens, now=now_monotonic):
            return RateLimitDecision(status="allowed", tokens=tokens)

        retry_after: Optional[float]
        if self.rate <= 0:
            retry_after = None
        else:
            deficit = max(tokens - self._tokens, 0.0)
            retry_after = deficit / self.rate if deficit > 0 else 0.0

        eta = now_epoch + retry_after if retry_after is not None else None
        ttl_deadline = (now_epoch + ttl_s) if ttl_s and ttl_s > 0 else None

        if queue_allowed:
            return RateLimitDecision(
                status="queued",
                tokens=tokens,
                retry_after_s=retry_after,
                eta_epoch_s=eta,
                ttl_deadline_s=ttl_deadline,
                reason="rate_limited",
            )

        return RateLimitDecision(
            status="rejected",
            tokens=tokens,
            retry_after_s=retry_after,
            eta_epoch_s=eta,
            reason="rate_limited",
        )

    def _consume(self, tokens: int, *, now: float) -> bool:
        delta = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + delta * self.rate)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


__all__ = [
    "RateLimitDecision",
    "RateLimiter",
    "NoopRateLimiter",
    "TokenBucketRateLimiter",
]
