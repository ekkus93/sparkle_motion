from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

import pytest

from sparkle_motion import gpu_utils

from . import helpers

pytestmark = pytest.mark.gpu

try:  # pragma: no cover - torch only available on GPU test hosts
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional outside GPU envs
    torch = None  # type: ignore

_MB = 1024 * 1024


@dataclass
class _TensorHandle:
    tensor: Any

    def close(self) -> None:
        if self.tensor is not None:
            self.tensor = None


def _memory_state(*, sync: bool = True) -> tuple[float, float]:
    assert torch is not None, "torch is required for GPU context tests"
    if sync:
        torch.cuda.synchronize()
    allocated_mb = torch.cuda.memory_allocated() / _MB
    free_bytes, _ = torch.cuda.mem_get_info()
    return allocated_mb, free_bytes / _MB


def _post_cleanup_state() -> tuple[float, float]:
    assert torch is not None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return _memory_state(sync=False)


def _allocation_size(free_mb: float, *, fraction: float, minimum: int, maximum: int) -> int:
    dynamic = max(minimum, min(maximum, int(free_mb * fraction)))
    safe_cap = max(minimum, int(free_mb * 0.6))
    return min(dynamic, safe_cap)


def _make_loader(allocation_mb: int) -> Callable[[], _TensorHandle]:
    assert torch is not None

    def _loader() -> _TensorHandle:
        elements = max(1, (allocation_mb * _MB) // 4)
        tensor = torch.ones((elements,), dtype=torch.float32, device="cuda")
        tensor.mul_(2.0)
        return _TensorHandle(tensor=tensor)

    return _loader


def _exercise_model_cycle(
    *,
    model_key: str,
    allocation_mb: int,
    baseline_allocated: float,
    baseline_free: float,
) -> tuple[float, float]:
    loader = _make_loader(allocation_mb)
    with gpu_utils.model_context(
        model_key,
        loader=loader,
        weights=model_key,
        offload=True,
        keep_warm=False,
        block_until_gpu_free=True,
    ) as ctx:
        allocated_during, free_during = _memory_state()
        allocation_gain = allocated_during - baseline_allocated
        free_loss = baseline_free - free_during
        assert allocation_gain >= allocation_mb * 0.85, "model load should reserve expected VRAM"
        assert free_loss >= allocation_mb * 0.5, "free VRAM should drop while model is loaded"
        snapshot = ctx.report_memory()
        assert snapshot, "memory snapshot should include device metrics"
        assert any("cuda" in dev.lower() for dev in snapshot), "must capture CUDA stats"
        assert ctx.metadata.get("offload") is True

    assert not gpu_utils.gpu_is_busy()
    allocated_after, free_after = _post_cleanup_state()
    assert allocated_after <= baseline_allocated + 8.0, "cleanup should release tensors"
    tolerance_free = max(64.0, allocation_mb * 0.25)
    assert free_after >= baseline_free - tolerance_free, "free VRAM should rebound after cleanup"
    return allocated_after, free_after


@pytest.mark.gpu
def test_gpu_context_model_offload() -> None:
    helpers.require_gpu_available()
    gpu_utils.evict_cached_model()
    baseline_allocated, baseline_free = _post_cleanup_state()
    if baseline_free < 256:
        pytest.skip("Insufficient free VRAM for offload test")

    sdxl_allocation = _allocation_size(baseline_free, fraction=0.2, minimum=96, maximum=1024)
    wan_allocation = _allocation_size(baseline_free, fraction=0.15, minimum=64, maximum=768)

    baseline_allocated, baseline_free = _exercise_model_cycle(
        model_key="gpu-test-sdxl",
        allocation_mb=sdxl_allocation,
        baseline_allocated=baseline_allocated,
        baseline_free=baseline_free,
    )

    baseline_allocated, baseline_free = _exercise_model_cycle(
        model_key="gpu-test-wan",
        allocation_mb=wan_allocation,
        baseline_allocated=baseline_allocated,
        baseline_free=baseline_free,
    )

    _exercise_model_cycle(
        model_key="gpu-test-sdxl",
        allocation_mb=sdxl_allocation,
        baseline_allocated=baseline_allocated,
        baseline_free=baseline_free,
    )


def _context_worker(
    *,
    name: str,
    allocation_mb: int,
    hold_s: float,
    enter_times: Dict[str, float],
    exit_times: Dict[str, float],
    ready_event: threading.Event,
) -> None:
    loader = _make_loader(allocation_mb)
    with gpu_utils.model_context(
        f"gpu-test-{name}",
        loader=loader,
        weights=name,
        offload=True,
        keep_warm=False,
        block_until_gpu_free=True,
    ):
        enter_times[name] = time.perf_counter()
        ready_event.set()
        time.sleep(hold_s)
    exit_times[name] = time.perf_counter()


@pytest.mark.gpu
def test_gpu_context_concurrent_requests() -> None:
    helpers.require_gpu_available()
    gpu_utils.evict_cached_model()
    baseline_allocated, baseline_free = _post_cleanup_state()
    if baseline_free < 192:
        pytest.skip("Insufficient free VRAM for concurrency test")

    allocation_mb = _allocation_size(baseline_free, fraction=0.12, minimum=64, maximum=512)
    hold_s = 0.5
    enter_times: Dict[str, float] = {}
    exit_times: Dict[str, float] = {}

    first_ready = threading.Event()
    second_ready = threading.Event()

    first = threading.Thread(
        target=_context_worker,
        kwargs={
            "name": "sdxl",
            "allocation_mb": allocation_mb,
            "hold_s": hold_s,
            "enter_times": enter_times,
            "exit_times": exit_times,
            "ready_event": first_ready,
        },
        daemon=True,
    )
    second = threading.Thread(
        target=_context_worker,
        kwargs={
            "name": "wan",
            "allocation_mb": allocation_mb,
            "hold_s": hold_s,
            "enter_times": enter_times,
            "exit_times": exit_times,
            "ready_event": second_ready,
        },
        daemon=True,
    )

    first.start()
    assert first_ready.wait(timeout=5.0), "First context failed to acquire GPU"
    start_second = time.perf_counter()
    second.start()
    assert second_ready.wait(timeout=10.0), "Second context never entered"

    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive(), "First worker hung"
    assert not second.is_alive(), "Second worker hung"

    sdxl_enter = enter_times["sdxl"]
    sdxl_exit = exit_times["sdxl"]
    wan_enter = enter_times["wan"]
    wan_exit = exit_times["wan"]

    assert wan_enter > sdxl_exit - 0.01, "Second context entered before first released GPU lock"
    wait_duration = wan_enter - start_second
    assert wait_duration >= hold_s * 0.8, "Second context should wait while first holds the lock"
    assert wan_exit > wan_enter
    assert sdxl_exit - sdxl_enter >= hold_s * 0.9

    allocated_after, free_after = _post_cleanup_state()
    assert allocated_after <= baseline_allocated + 8.0
    assert free_after >= baseline_free - allocation_mb
    assert not gpu_utils.gpu_is_busy()


@pytest.mark.gpu
def test_gpu_oom_recovery() -> None:
    helpers.require_gpu_available()
    gpu_utils.evict_cached_model()
    baseline_allocated, baseline_free = _post_cleanup_state()
    if baseline_free < 160:
        pytest.skip("Insufficient free VRAM for OOM test")

    assert torch is not None

    attempted_size = max(256, int(baseline_free * 0.8))

    def _oom_loader() -> _TensorHandle:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory while allocating tensor")

    with pytest.raises(gpu_utils.ModelOOMError) as excinfo:
        with gpu_utils.model_context(
            "gpu-test-oom",
            loader=_oom_loader,
            weights="gpu-test-oom",
            offload=True,
            keep_warm=False,
            block_until_gpu_free=True,
        ):
            pass

    err = excinfo.value
    assert err.stage == "load"
    assert err.memory_snapshot
    assert not gpu_utils.gpu_is_busy()

    state = gpu_utils.OOMAttemptState(
        model_key="gpu-test-oom",
        stage="images",
        attempted_size=attempted_size,
        min_size=32,
        initial_size=attempted_size,
        shrink_factor=0.5,
        failure_count=1,
        error_message=str(err),
        history=(attempted_size,),
        memory_snapshot=err.memory_snapshot,
    )
    fallback_size = gpu_utils.suggest_shrink_for_oom(state)
    assert fallback_size < attempted_size

    baseline_allocated, baseline_free = _post_cleanup_state()
    allocation_cap = _allocation_size(baseline_free, fraction=0.25, minimum=32, maximum=512)
    allocation_mb = max(32, min(fallback_size, allocation_cap))

    _exercise_model_cycle(
        model_key="gpu-test-oom-retry",
        allocation_mb=allocation_mb,
        baseline_allocated=baseline_allocated,
        baseline_free=baseline_free,
    )
