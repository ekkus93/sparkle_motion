from __future__ import annotations

import contextlib
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterator, Mapping, MutableMapping, Optional, Sequence

from . import adk_helpers, telemetry

try:  # pragma: no cover - optional heavy dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - environments without torch
    torch = None  # type: ignore


_OOM_TOKENS = ("out of memory", "cuda error", "cublas_status_alloc_failed")
_CLEANUP_METHODS = ("close", "shutdown", "stop", "release", "unload", "dispose")

_DEVICE_MAP_PRESETS: dict[str, Mapping[str, str]] = {
    "a100-80gb": {"text_encoder": "cuda:0", "unet": "cuda:0", "vae": "cuda:0", "controlnet": "cuda:0"},
    "a100-40gb": {"text_encoder": "cuda:0", "unet": "cuda:0", "vae": "cuda:0"},
    "rtx4090": {"text_encoder": "cuda:0", "unet": "cuda:0", "vae": "cuda:0"},
}


class ModelContextError(RuntimeError):
    """Base class for gpu_utils errors."""


class ModelLoadTimeoutError(ModelContextError):
    """Raised when model loading exceeds the configured timeout."""

    def __init__(self, *, model_key: str, timeout_s: float) -> None:
        super().__init__(f"Timed out loading model '{model_key}' after {timeout_s:.2f}s")
        self.model_key = model_key
        self.timeout_s = timeout_s


class ModelOOMError(ModelContextError):
    """Raised when CUDA/NPU OOM conditions are detected."""

    def __init__(
        self,
        *,
        model_key: str,
        stage: str,
        message: str,
        device_map: Optional[Mapping[str, str]],
        memory_snapshot: Mapping[str, Mapping[str, float]],
    ) -> None:
        super().__init__(f"{model_key} OOM during {stage}: {message}")
        self.model_key = model_key
        self.stage = stage
        self.device_map = dict(device_map or {})
        self.memory_snapshot = dict(memory_snapshot)


@dataclass
class ModelContext:
    """Model handle wrapper returned by :func:`model_context`."""

    model_key: str
    pipeline: Any
    weights: Optional[str]
    device_map: Optional[Mapping[str, str]]
    allocated_devices: Sequence[str]
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def report_memory(self) -> Mapping[str, Mapping[str, float]]:
        return _collect_memory_snapshot(self.allocated_devices)

    def __getattr__(self, item: str) -> Any:
        # Allow legacy callers/tests to treat the context as the pipeline directly.
        return getattr(self.pipeline, item)

    @contextlib.contextmanager
    def inference_span(self, label: str = "default", extra: Optional[Mapping[str, Any]] = None) -> Iterator[Any]:
        payload = {"model_key": self.model_key, "label": label, **(extra or {})}
        _emit_gpu_event("gpu.model.inference_start", payload, self.allocated_devices)
        start = time.time()
        try:
            yield self.pipeline
        except RuntimeError as exc:  # pragma: no cover - inference spans optional
            raise _normalize_oom(exc, model_key=self.model_key, stage="inference", device_map=self.device_map) from exc
        finally:
            duration_ms = int((time.time() - start) * 1000)
            payload["duration_ms"] = duration_ms
            _emit_gpu_event("gpu.model.inference_end", payload, self.allocated_devices)


@contextlib.contextmanager
def model_context(
    model_key: str,
    *,
    loader: Optional[Callable[[], Any]] = None,
    weights: Optional[str] = None,
    offload: bool = True,
    xformers: bool = True,
    compile: bool = False,
    device_map: Optional[Mapping[str, str]] = None,
    low_cpu_mem_usage: bool = True,
    max_memory: Optional[Mapping[str, str]] = None,
    timeout_s: Optional[float] = None,
) -> Generator[ModelContext, None, None]:
    """Load a model via *loader* and guarantee deterministic cleanup."""

    actual_loader = loader
    if isinstance(model_key, str) and model_key.strip():
        actual_key = model_key
    else:
        raise ValueError("model_context requires a non-empty string model_key")

    if actual_loader is None:
        raise ValueError("model_context requires a loader callable")

    metadata = {
        "model_key": actual_key,
        "weights": weights,
        "offload": offload,
        "xformers": xformers,
        "compile": compile,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "max_memory": dict(max_memory or {}),
        "device_map": dict(device_map or {}),
    }
    load_start = time.time()
    allocated_devices = _derive_allocated_devices(device_map)
    _emit_gpu_event("gpu.model.load_start", metadata, allocated_devices)

    handle: Optional[Any] = None
    try:
        handle = _execute_loader(actual_loader, actual_key, timeout_s)
        ctx = ModelContext(
            model_key=actual_key,
            pipeline=handle,
            weights=weights,
            device_map=device_map,
            allocated_devices=allocated_devices,
            metadata=metadata,
        )
        _emit_gpu_event(
            "gpu.model.load_complete",
            {**metadata, "duration_ms": int((time.time() - load_start) * 1000)},
            allocated_devices,
        )
        yield ctx
    except RuntimeError as exc:
        raise _normalize_oom(exc, model_key=actual_key, stage="load", device_map=device_map) from exc
    finally:
        if handle is not None:
            _cleanup_handle(handle)
            _emit_gpu_event("gpu.model.cleanup", metadata, allocated_devices)


def compute_device_map(preset: str, overrides: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
    """Return a shallow copy of the registered device-map preset."""

    preset_key = preset.lower()
    if preset_key not in _DEVICE_MAP_PRESETS:
        raise KeyError(f"Unknown device preset '{preset}'")
    base = dict(_DEVICE_MAP_PRESETS[preset_key])
    if overrides:
        base.update(overrides)
    return base


def list_device_presets() -> Mapping[str, Mapping[str, str]]:
    """Return available device-map presets."""

    return {k: dict(v) for k, v in _DEVICE_MAP_PRESETS.items()}


def _execute_loader(loader: Callable[[], Any], model_key: str, timeout_s: Optional[float]) -> Any:
    if timeout_s is None:
        return loader()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(loader)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            future.cancel()
            raise ModelLoadTimeoutError(model_key=model_key, timeout_s=timeout_s) from exc


def _derive_allocated_devices(device_map: Optional[Mapping[str, str]]) -> Sequence[str]:
    if device_map:
        seen: list[str] = []
        for dev in device_map.values():
            if dev not in seen:
                seen.append(dev)
        return tuple(seen)

    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():  # pragma: no branch - depends on env
                return (f"cuda:{torch.cuda.current_device()}",)
        except Exception:
            pass
    return ("cpu",)


def _cleanup_handle(handle: Any) -> None:
    for name in _CLEANUP_METHODS:
        fn = getattr(handle, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():  # pragma: no branch - depends on env
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize()
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
        except Exception:
            pass

    with contextlib.suppress(Exception):
        gc.collect()


def _emit_gpu_event(name: str, payload: Mapping[str, Any], allocated_devices: Sequence[str]) -> None:
    data = dict(payload)
    data.setdefault("allocated_devices", list(allocated_devices))
    telemetry.emit_event(name, data)
    try:
        adk_helpers.write_memory_event(run_id=None, event_type=name, payload=data)
    except Exception:
        pass


def _collect_memory_snapshot(devices: Sequence[str]) -> Mapping[str, Mapping[str, float]]:
    snapshot: dict[str, Mapping[str, float]] = {}
    if torch is None or getattr(torch, "cuda", None) is None:
        return snapshot

    for dev in devices:
        if not dev.startswith("cuda"):
            continue
        try:
            index = int(dev.split(":", 1)[1]) if ":" in dev else torch.cuda.current_device()
            if torch.cuda.is_available():  # pragma: no branch - env dependent
                free, total = torch.cuda.mem_get_info(index)
                snapshot[dev] = {
                    "free_mb": round(free / (1024 * 1024), 2),
                    "total_mb": round(total / (1024 * 1024), 2),
                    "used_mb": round((total - free) / (1024 * 1024), 2),
                }
        except Exception:
            continue
    return snapshot


def _normalize_oom(
    exc: RuntimeError,
    *,
    model_key: str,
    stage: str,
    device_map: Optional[Mapping[str, str]],
) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if any(token in lowered for token in _OOM_TOKENS):
        snapshot = _collect_memory_snapshot(tuple(device_map.values()) if device_map else ("cuda:0", "cpu"))
        return ModelOOMError(
            model_key=model_key,
            stage=stage,
            message=message,
            device_map=device_map,
            memory_snapshot=snapshot,
        )
    return exc
