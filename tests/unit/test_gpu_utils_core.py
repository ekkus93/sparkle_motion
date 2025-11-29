from __future__ import annotations

import time
import threading
from typing import Any

import pytest

from sparkle_motion import gpu_utils, telemetry


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def is_available(self) -> bool:  # pragma: no cover - helper
        return True

    def current_device(self) -> int:  # pragma: no cover - helper
        return 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def synchronize(self) -> None:  # pragma: no cover - helper
        return None

    def mem_get_info(self, index: int) -> tuple[int, int]:
        # return (free, total) bytes
        return 400 * 1024 * 1024, 800 * 1024 * 1024


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


@pytest.fixture(autouse=True)
def _ensure_fixture_mode(monkeypatch):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    telemetry.clear_events()
    gpu_utils.evict_cached_model()
    yield
    gpu_utils.evict_cached_model()
    telemetry.clear_events()


def test_model_context_runs_loader_and_cleanup(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    cleanup_called: dict[str, bool] = {"flag": False}

    class _Handle:
        def close(self) -> None:
            cleanup_called["flag"] = True

    loader_calls: list[int] = []

    def loader() -> Any:
        loader_calls.append(1)
        return _Handle()

    with gpu_utils.model_context("demo", loader=loader, device_map={"unet": "cuda:0"}) as ctx:
        assert isinstance(ctx.pipeline, _Handle)
        snapshot = ctx.report_memory()
        assert "cuda:0" in snapshot

    assert loader_calls
    assert cleanup_called["flag"]
    events = telemetry.get_events()
    names = [ev["name"] for ev in events]
    assert "gpu.model.load_start" in names
    assert "gpu.model.cleanup" in names


def test_model_context_timeout(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    def slow_loader() -> None:
        time.sleep(0.2)

    with pytest.raises(gpu_utils.ModelLoadTimeoutError):
        with gpu_utils.model_context("timeout-demo", loader=slow_loader, timeout_s=0.05):
            pass


def test_model_context_normalizes_oom(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    def oom_loader() -> None:
        raise RuntimeError("CUDA out of memory while allocating tensor")

    with pytest.raises(gpu_utils.ModelOOMError) as excinfo:
        with gpu_utils.model_context("oom-demo", loader=oom_loader):
            pass

    assert excinfo.value.stage == "load"


def test_model_context_keep_warm_reuses_handle(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    loader_calls: list[int] = []

    class _Handle:
        def __init__(self, idx: int) -> None:
            self.idx = idx

        def close(self) -> None:
            loader_calls.append(-1)

    counter = {"value": 0}

    def loader() -> Any:
        counter["value"] += 1
        loader_calls.append(counter["value"])
        return _Handle(counter["value"])

    with gpu_utils.model_context("demo", loader=loader, keep_warm=True, warm_ttl_s=60) as ctx:
        first_idx = ctx.pipeline.idx

    with gpu_utils.model_context("demo", loader=loader, keep_warm=True, warm_ttl_s=60) as ctx:
        assert ctx.pipeline.idx == first_idx

    assert loader_calls.count(1) == 1


def test_model_context_keep_warm_respects_ttl(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    loader_calls: list[int] = []

    def loader() -> Any:
        loader_calls.append(1)
        return object()

    with gpu_utils.model_context("demo", loader=loader, keep_warm=True, warm_ttl_s=0.05):
        pass
    time.sleep(0.1)
    with gpu_utils.model_context("demo", loader=loader, keep_warm=True, warm_ttl_s=0.05):
        pass
    assert loader_calls == [1, 1]


def test_model_context_busy_error(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    def loader() -> Any:
        return object()

    ready = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with gpu_utils.model_context("demo", loader=loader):
            ready.set()
            release.wait(timeout=1)

    thread = threading.Thread(target=holder)
    thread.start()
    ready.wait(timeout=1)
    with pytest.raises(gpu_utils.GpuBusyError):
        with gpu_utils.model_context("demo", loader=loader, block_until_gpu_free=False):
            pass
    release.set()
    thread.join(timeout=1)


def test_compute_device_map_presets():
    preset = gpu_utils.compute_device_map("a100-80gb")
    assert preset["unet"] == "cuda:0"
    overrides = gpu_utils.compute_device_map("rtx4090", overrides={"vae": "cuda:1"})
    assert overrides["vae"] == "cuda:1"
