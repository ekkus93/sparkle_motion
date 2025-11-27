from __future__ import annotations

import math
import random
import time
from typing import Callable, Iterable, Optional


def exponential_backoff(base: float = 0.1, factor: float = 2.0, jitter: float = 0.1, max_backoff: float = 30.0) -> Callable[[int], float]:
    """Return a function that computes backoff delay for attempt index (0-based).

    deterministic when `jitter` is 0.0; otherwise adds uniform jitter in
    +/- jitter*delay.
    """

    def _delay(attempt: int) -> float:
        if attempt <= 0:
            return 0.0
        delay = base * (factor ** (attempt - 1))
        delay = min(delay, max_backoff)
        if jitter and jitter > 0:
            # uniform jitter in [-jitter*delay, +jitter*delay]
            delta = (random.random() * 2 - 1) * jitter * delay
            delay = max(0.0, delay + delta)
        return delay

    return _delay


def retry_call(fn: Callable, attempts: int = 3, backoff_fn: Optional[Callable[[int], float]] = None, on_retry: Optional[Callable[[int, Exception], None]] = None):
    """Call `fn()` with retries. Returns fn() result or raises last exception.

    The function sleeps according to `backoff_fn(attempt)` between retries.
    `attempts` is the total number of tries including the first.
    """
    last_exc = None
    backoff_fn = backoff_fn or exponential_backoff()
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if on_retry:
                try:
                    on_retry(attempt, e)
                except Exception:
                    pass
            if attempt == attempts:
                break
            delay = backoff_fn(attempt)
            time.sleep(delay)
    raise last_exc
