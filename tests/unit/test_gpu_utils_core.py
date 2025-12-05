from __future__ import annotations

import asyncio
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


def _oom_state(**overrides: Any) -> gpu_utils.OOMAttemptState:
    base: dict[str, Any] = {
        "model_key": "wan",
        "stage": "render",
        "attempted_size": 64,
        "min_size": 8,
        "initial_size": 64,
        "shrink_factor": 0.5,
        "failure_count": 1,
        "error_message": "RuntimeError: CUDA out of memory",
    }
    base.update(overrides)
    return gpu_utils.OOMAttemptState(**base)


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
    assert "gpu.model.inference_start" in names
    assert "gpu.model.inference_end" in names
    assert "gpu.model.cleanup" in names
    inference_start = next(ev for ev in events if ev["name"] == "gpu.model.inference_start")
    assert inference_start["payload"]["model_key"] == "demo"
    assert inference_start["payload"]["label"] == "context"
    inference_end = next(ev for ev in events if ev["name"] == "gpu.model.inference_end")
    assert inference_end["payload"]["status"] == "ok"
    assert inference_end["payload"]["duration_ms"] >= 0
    cleanup_event = next(ev for ev in events if ev["name"] == "gpu.model.cleanup")
    snapshot = cleanup_event["payload"].get("snapshot")
    assert snapshot is not None
    assert "cuda:0" in snapshot


def test_model_context_timeout(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    def slow_loader() -> None:
        time.sleep(0.2)

    with pytest.raises(gpu_utils.ModelLoadTimeoutError) as excinfo:
        with gpu_utils.model_context("timeout-demo", loader=slow_loader, timeout_s=0.05):
            pass
    assert excinfo.value.retryable is True


def test_model_context_normalizes_oom(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    def oom_loader() -> None:
        raise RuntimeError("CUDA out of memory while allocating tensor")

    with pytest.raises(gpu_utils.ModelOOMError) as excinfo:
        with gpu_utils.model_context("oom-demo", loader=oom_loader):
            pass

    assert excinfo.value.stage == "load"
    assert excinfo.value.retryable is False


def test_model_context_wraps_generic_error(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    def broken_loader() -> None:
        raise ValueError("boom")

    with pytest.raises(gpu_utils.ModelLoadError) as excinfo:
        with gpu_utils.model_context("broken", loader=broken_loader):
            pass

    assert excinfo.value.retryable is False
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


def test_suggest_shrink_uses_torch_trace_values():
    message = (
        "RuntimeError: CUDA out of memory. Tried to allocate 6.0 GiB (GPU 0; "
        "24.0 GiB total capacity; 12.0 GiB already allocated; 3.0 GiB free; 12.5 GiB reserved)"
    )
    state = _oom_state(error_message=message, shrink_factor=0.7)
    result = gpu_utils.suggest_shrink_for_oom(state)
    assert result == 28


def test_suggest_shrink_falls_back_without_trace():
    state = _oom_state(attempted_size=48, initial_size=48)
    result = gpu_utils.suggest_shrink_for_oom(state)
    assert result == 24


def test_suggest_shrink_avoids_repeating_history():
    state = _oom_state(
        attempted_size=32,
        initial_size=40,
        shrink_factor=0.75,
        history=(48, 24),
    )
    result = gpu_utils.suggest_shrink_for_oom(state)
    assert result == 22


def test_suggest_shrink_uses_snapshot_when_free_missing():
    message = "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB"
    snapshot = {"cuda:0": {"free_mb": 512.0, "total_mb": 24576.0}}
    state = _oom_state(
        attempted_size=64,
        error_message=message,
        memory_snapshot=snapshot,
    )
    result = gpu_utils.suggest_shrink_for_oom(state)
    assert result == 14


def test_model_context_async_usage(monkeypatch):
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

    async def _exercise() -> None:
        async with gpu_utils.model_context("demo-async", loader=loader) as ctx:
            assert isinstance(ctx.pipeline, _Handle)

    asyncio.run(_exercise())

    assert loader_calls
    assert cleanup_called["flag"]


def test_report_memory_cpu_fallback(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        """MemTotal: 1024000 kB
MemFree: 204800 kB
MemAvailable: 512000 kB
"""
    )
    monkeypatch.setattr(gpu_utils, "_PROC_MEMINFO_PATH", meminfo)
    monkeypatch.setattr(gpu_utils, "torch", None)
    ctx = gpu_utils.ModelContext(
        model_key="cpu-only",
        pipeline=object(),
        weights=None,
        device_map=None,
        allocated_devices=("cpu",),
        metadata={},
    )
    snapshot = ctx.report_memory()
    assert "cpu" in snapshot
    assert snapshot["cpu"]["total_mb"] == pytest.approx(1000.0, rel=0.01)
    assert snapshot["cpu"]["used_mb"] == pytest.approx(500.0, rel=0.01)
    events = telemetry.get_events()
    mem_events = [ev for ev in events if ev["name"] == "gpu.model.memory_snapshot"]
    assert len(mem_events) == 1
    payload = mem_events[0]["payload"]
    assert payload["model_key"] == "cpu-only"
    assert payload["snapshot"] == snapshot
    assert "nvml" not in payload


def test_report_memory_gpu_and_cpu(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        """MemTotal: 1024000 kB
MemFree: 204800 kB
MemAvailable: 512000 kB
"""
    )
    monkeypatch.setattr(gpu_utils, "_PROC_MEMINFO_PATH", meminfo)
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)
    ctx = gpu_utils.ModelContext(
        model_key="combo",
        pipeline=object(),
        weights=None,
        device_map=None,
        allocated_devices=("cuda:0", "cpu"),
        metadata={},
    )
    snapshot = ctx.report_memory()
    assert "cuda:0" in snapshot
    assert snapshot["cuda:0"]["total_mb"] == pytest.approx(800.0, rel=0.01)
    assert "cpu" in snapshot
    events = telemetry.get_events()
    mem_events = [ev for ev in events if ev["name"] == "gpu.model.memory_snapshot"]
    assert len(mem_events) == 1
    payload = mem_events[0]["payload"]
    assert payload["model_key"] == "combo"
    assert payload["snapshot"] == snapshot
    assert "nvml" not in payload


def test_report_memory_emits_nvml_payload(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        """MemTotal: 1024000 kB
MemFree: 204800 kB
MemAvailable: 512000 kB
"""
    )
    monkeypatch.setattr(gpu_utils, "_PROC_MEMINFO_PATH", meminfo)
    fake_torch = _FakeTorch()
    monkeypatch.setattr(gpu_utils, "torch", fake_torch)

    def _fake_nvml(devices):
        return {"cuda:0": {"temperature_c": 42, "fan_percent": 30}}

    monkeypatch.setattr(gpu_utils, "_collect_nvml_metrics", lambda devices: _fake_nvml(devices))
    ctx = gpu_utils.ModelContext(
        model_key="nvml-demo",
        pipeline=object(),
        weights=None,
        device_map=None,
        allocated_devices=("cuda:0",),
        metadata={},
    )
    snapshot = ctx.report_memory()
    assert "cuda:0" in snapshot
    events = telemetry.get_events()
    mem_events = [ev for ev in events if ev["name"] == "gpu.model.memory_snapshot"]
    assert len(mem_events) == 1
    payload = mem_events[0]["payload"]
    assert payload["model_key"] == "nvml-demo"
    assert payload["snapshot"] == snapshot
    assert payload["nvml"] == {"cuda:0": {"temperature_c": 42, "fan_percent": 30}}


def test_compute_device_map_presets():
    preset = gpu_utils.compute_device_map("a100-80gb")
    assert preset["unet"] == "cuda:0"
    overrides = gpu_utils.compute_device_map("rtx4090", overrides={"vae": "cuda:1"})
    assert overrides["vae"] == "cuda:1"


def test_collect_nvml_metrics_uses_sampler(monkeypatch):
    class _Sampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self, devices):
            self.calls += 1
            return {"cuda:0": {"temperature_c": 55}}

    sampler = _Sampler()
    monkeypatch.setattr(gpu_utils, "_resolve_nvml_sampler", lambda: sampler)
    result = gpu_utils._collect_nvml_metrics(("cuda:0",))
    assert result == {"cuda:0": {"temperature_c": 55}}
    assert sampler.calls == 1


def test_collect_nvml_metrics_logs_error(monkeypatch):
    class _Sampler:
        def sample(self, devices):
            raise gpu_utils.NvmlError("nvml broke")

    monkeypatch.setattr(gpu_utils, "_resolve_nvml_sampler", lambda: _Sampler())
    result = gpu_utils._collect_nvml_metrics(("cuda:0",))
    assert result is None
    events = telemetry.get_events()
    assert any(ev["name"] == "gpu.model.nvml_error" for ev in events)
