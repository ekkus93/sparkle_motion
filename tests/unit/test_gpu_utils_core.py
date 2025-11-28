from __future__ import annotations

import time
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
    yield
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


def test_compute_device_map_presets():
    preset = gpu_utils.compute_device_map("a100-80gb")
    assert preset["unet"] == "cuda:0"
    overrides = gpu_utils.compute_device_map("rtx4090", overrides={"vae": "cuda:1"})
    assert overrides["vae"] == "cuda:1"
