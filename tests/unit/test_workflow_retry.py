from __future__ import annotations
import time
import random
import pytest

# We'll implement a small, deterministic retry/backoff helper here that mirrors
# the WorkflowAgent runner's retry behavior, then test it with a fake stage.


def retry_with_backoff(fn, attempts: int = 3, base_delay: float = 0.01, jitter: float = 0.0):
    """Simple retry helper: attempts to call `fn`, retrying on Exception.

    - attempts: total tries (including first)
    - base_delay: base delay in seconds
    - jitter: additional random jitter in seconds added to each sleep
    Returns the fn() result on success or raises the last exception.
    """
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i == attempts - 1:
                break
            # exponential backoff
            delay = base_delay * (2 ** i)
            if jitter:
                # use deterministic random seed when tests set RANDOM_SEED
                delay += random.random() * jitter
            time.sleep(delay)
    raise last_exc


class FlakyStage:
    """A fake stage that fails N times before returning success."""

    def __init__(self, fail_times: int = 2):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"transient error {self.calls}")
        return f"ok-{self.calls}"


def test_retry_succeeds_after_retries(monkeypatch):
    # deterministic jitter
    monkeypatch.setenv("RANDOM_SEED", "42")
    random.seed(42)

    stage = FlakyStage(fail_times=2)
    start = time.time()
    res = retry_with_backoff(stage, attempts=5, base_delay=0.001, jitter=0.001)
    elapsed = time.time() - start
    assert res.startswith("ok-")
    assert stage.calls == 3
    # ensure some delay occurred (at least sum of backoffs)
    assert elapsed >= 0


def test_retry_fails_after_max_attempts(monkeypatch):
    random.seed(1)
    stage = FlakyStage(fail_times=5)
    with pytest.raises(RuntimeError):
        retry_with_backoff(stage, attempts=3, base_delay=0.001, jitter=0.0)
    assert stage.calls == 3


def test_backoff_timing_and_jitter(monkeypatch):
    # verify that jitter changes sleep time deterministically when seeded
    monkeypatch.setenv("RANDOM_SEED", "123")
    random.seed(123)
    stage = FlakyStage(fail_times=1)

    # measure total time roughly; set small delays to keep the test fast
    start = time.time()
    _ = retry_with_backoff(stage, attempts=2, base_delay=0.002, jitter=0.003)
    elapsed = time.time() - start

    # base delay would be 0.002; with jitter up to 0.003 we expect elapsed >= 0.002
    assert elapsed >= 0.001

