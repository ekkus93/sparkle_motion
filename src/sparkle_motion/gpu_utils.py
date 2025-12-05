from __future__ import annotations

import asyncio
import contextlib
import ctypes
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Generator, Iterator, Mapping, MutableMapping, Optional, Sequence

from ctypes.util import find_library

from . import adk_helpers, telemetry

try:  # pragma: no cover - optional heavy dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - environments without torch
    torch = None  # type: ignore


_OOM_TOKENS = ("out of memory", "cuda error", "cublas_status_alloc_failed")
_CLEANUP_METHODS = ("close", "shutdown", "stop", "release", "unload", "dispose")
_PROC_MEMINFO_PATH = Path("/proc/meminfo")

_DEVICE_MAP_PRESETS: dict[str, Mapping[str, str]] = {
    "a100-80gb": {"text_encoder": "cuda:0", "unet": "cuda:0", "vae": "cuda:0", "controlnet": "cuda:0"},
    "a100-40gb": {"text_encoder": "cuda:0", "unet": "cuda:0", "vae": "cuda:0"},
    "rtx4090": {"text_encoder": "cuda:0", "unet": "cuda:0", "vae": "cuda:0"},
}


class NvmlError(RuntimeError):
    """Raised when NVML sampling fails."""


_NVML_STATE_UNKNOWN = "unknown"
_NVML_STATE_READY = "ready"
_NVML_STATE_ERROR = "error"
_NVML_STATE = _NVML_STATE_UNKNOWN
_NVML_SAMPLER: Optional["_NvmlSampler"] = None
_NVML_STATE_LOCK = Lock()
_NVML_ERROR_REPORTED = False


class ModelContextError(RuntimeError):
    """Base class for gpu_utils errors."""


class ModelLoadError(ModelContextError):
    """Base class for load-time errors (retryable flag for callers)."""

    def __init__(self, *, message: str, model_key: str, stage: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.model_key = model_key
        self.stage = stage
        self.retryable = retryable


class ModelLoadTimeoutError(ModelLoadError):
    """Raised when model loading exceeds the configured timeout."""

    def __init__(self, *, model_key: str, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        super().__init__(
            message=f"Timed out loading model '{model_key}' after {timeout_s:.2f}s",
            model_key=model_key,
            stage="load",
            retryable=True,
        )


class ModelOOMError(ModelLoadError):
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
        super().__init__(
            message=f"{model_key} OOM during {stage}: {message}",
            model_key=model_key,
            stage=stage,
            retryable=False,
        )
        self.device_map = dict(device_map or {})
        self.memory_snapshot = dict(memory_snapshot)


class GpuBusyError(ModelContextError):
    """Raised when the shared GPU lock cannot be acquired immediately."""

    def __init__(self, *, model_key: str) -> None:
        super().__init__(f"GPU busy while loading '{model_key}'")
        self.model_key = model_key


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
        payload = _build_memory_payload(self.model_key, self.allocated_devices)
        snapshot = payload.get("snapshot")
        if len(payload) > 1:
            _emit_gpu_event("gpu.model.memory_snapshot", payload, self.allocated_devices)
        return snapshot or {}

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


@dataclass
class _CachedModelHandle:
    model_key: str
    handle: Any
    weights: Optional[str]
    device_map: Optional[Mapping[str, str]]
    allocated_devices: Sequence[str]
    metadata: MutableMapping[str, Any]
    warm_ttl_s: Optional[float]
    last_used_s: float


_GPU_LOCK = Lock()
_CACHED_MODEL: Optional[_CachedModelHandle] = None


class _ModelContextManager(
    contextlib.ContextDecorator,
    contextlib.AbstractContextManager[ModelContext],
    contextlib.AbstractAsyncContextManager[ModelContext],
):
    """Adapter that exposes both sync and async context manager protocols."""

    def __init__(self, factory: Callable[[], contextlib.AbstractContextManager[ModelContext]]) -> None:
        self._factory = factory
        self._context = factory()

    def __enter__(self) -> ModelContext:
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, exc_tb) -> bool | None:
        return self._context.__exit__(exc_type, exc, exc_tb)

    async def __aenter__(self) -> ModelContext:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.__enter__)

    async def __aexit__(self, exc_type, exc, exc_tb) -> bool | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.__exit__(exc_type, exc, exc_tb))


def gpu_is_busy() -> bool:
    """Return True if another model_context currently owns the GPU lock."""

    return _GPU_LOCK.locked()


def evict_cached_model(model_key: Optional[str] = None) -> bool:
    """Drop the in-memory cached model, if present.

    Returns True when a cached model was released.
    """

    with _GPU_LOCK:
        entry = _CACHED_MODEL
        if entry is None or (model_key and entry.model_key != model_key):
            return False
        _evict_cached_entry(entry, reason="manual")
        return True


def _maybe_take_cached_model(model_key: str, *, keep_warm: bool, now_s: float) -> Optional[_CachedModelHandle]:
    global _CACHED_MODEL
    entry = _CACHED_MODEL
    if entry is None:
        return None
    expired = _entry_expired(entry, now_s)
    if expired or entry.model_key != model_key or not keep_warm:
        reason = "expired" if expired else ("replacement" if entry.model_key != model_key else "caller_disabled")
        _evict_cached_entry(entry, reason=reason)
        return None
    return entry


def _evict_conflicting_cache(model_key: str, *, keep_warm: bool) -> None:
    entry = _CACHED_MODEL
    if entry is None:
        return
    if entry.model_key != model_key or not keep_warm:
        reason = "replacement" if entry.model_key != model_key else "caller_disabled"
        _evict_cached_entry(entry, reason=reason)


def _store_cached_model(
    *,
    model_key: str,
    handle: Any,
    weights: Optional[str],
    device_map: Optional[Mapping[str, str]],
    allocated_devices: Sequence[str],
    metadata: MutableMapping[str, Any],
    warm_ttl_s: Optional[float],
    last_used_s: float,
) -> None:
    global _CACHED_MODEL
    _CACHED_MODEL = _CachedModelHandle(
        model_key=model_key,
        handle=handle,
        weights=weights,
        device_map=device_map,
        allocated_devices=allocated_devices,
        metadata=metadata,
        warm_ttl_s=warm_ttl_s,
        last_used_s=last_used_s,
    )
    _emit_gpu_event(
        "gpu.model.cache_store",
        {"model_key": model_key, "ttl_s": warm_ttl_s},
        allocated_devices,
    )


def _evict_cached_entry(entry: _CachedModelHandle, *, reason: str) -> None:
    global _CACHED_MODEL
    _cleanup_handle(entry.handle)
    _emit_gpu_event(
        "gpu.model.cache_evict",
        {"model_key": entry.model_key, "reason": reason},
        entry.allocated_devices,
    )
    if _CACHED_MODEL is entry:
        _CACHED_MODEL = None


def _entry_expired(entry: _CachedModelHandle, now_s: float) -> bool:
    return entry.warm_ttl_s is not None and (now_s - entry.last_used_s) > entry.warm_ttl_s


@contextlib.contextmanager
def _model_context_sync(
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
    keep_warm: bool = False,
    warm_ttl_s: Optional[float] = 900.0,
    block_until_gpu_free: bool = True,
) -> Generator[ModelContext, None, None]:
    """Load a model via *loader* and guarantee deterministic cleanup."""

    actual_loader = loader
    if isinstance(model_key, str) and model_key.strip():
        actual_key = model_key
    else:
        raise ValueError("model_context requires a non-empty string model_key")

    if actual_loader is None:
        raise ValueError("model_context requires a loader callable")

    normalized_ttl = None if warm_ttl_s is not None and warm_ttl_s <= 0 else warm_ttl_s
    acquired = _GPU_LOCK.acquire(blocking=block_until_gpu_free)
    if not acquired:
        raise GpuBusyError(model_key=actual_key)

    metadata: MutableMapping[str, Any] = {
        "model_key": actual_key,
        "weights": weights,
        "offload": offload,
        "xformers": xformers,
        "compile": compile,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "max_memory": dict(max_memory or {}),
        "device_map": dict(device_map or {}),
    }
    allocated_devices: Sequence[str] = ()
    handle: Optional[Any] = None
    cached_entry: Optional[_CachedModelHandle] = None
    reused = False
    success = False
    context_span_payload: Optional[dict[str, Any]] = None
    context_span_started_at = 0.0

    try:
        now = time.time()
        cached_entry = _maybe_take_cached_model(actual_key, keep_warm=keep_warm, now_s=now)
        if cached_entry is not None:
            reused = True
            handle = cached_entry.handle
            allocated_devices = cached_entry.allocated_devices
            metadata = dict(cached_entry.metadata)
            idle_ms = int((now - cached_entry.last_used_s) * 1000)
            _emit_gpu_event(
                "gpu.model.cache_reuse",
                {**metadata, "idle_ms": idle_ms},
                allocated_devices,
            )
        else:
            _evict_conflicting_cache(actual_key, keep_warm=keep_warm)
            allocated_devices = _derive_allocated_devices(device_map)
            load_start = time.time()
            _emit_gpu_event("gpu.model.load_start", metadata, allocated_devices)
            handle = _execute_loader(actual_loader, actual_key, timeout_s)
            _emit_gpu_event(
                "gpu.model.load_complete",
                {**metadata, "duration_ms": int((time.time() - load_start) * 1000)},
                allocated_devices,
            )

        ctx = ModelContext(
            model_key=actual_key,
            pipeline=handle,
            weights=weights,
            device_map=device_map,
            allocated_devices=allocated_devices,
            metadata=metadata,
        )
        context_span_payload = {"model_key": actual_key, "label": "context"}
        context_span_started_at = time.time()
        _emit_gpu_event("gpu.model.inference_start", context_span_payload, allocated_devices)
        yield ctx
        success = True
    except Exception as exc:
        raise _wrap_load_exception(exc, model_key=actual_key, stage="load", device_map=device_map) from exc
    finally:
        if context_span_payload is not None:
            span_payload = dict(context_span_payload)
            span_payload["duration_ms"] = int((time.time() - context_span_started_at) * 1000)
            span_payload["status"] = "ok" if success else "error"
            _emit_gpu_event("gpu.model.inference_end", span_payload, allocated_devices)
        now = time.time()
        if success and keep_warm:
            if reused and cached_entry is not None:
                cached_entry.last_used_s = now
                cached_entry.warm_ttl_s = normalized_ttl if normalized_ttl is not None else cached_entry.warm_ttl_s
                cached_entry.metadata = dict(metadata)
            elif handle is not None:
                _store_cached_model(
                    model_key=actual_key,
                    handle=handle,
                    weights=weights,
                    device_map=device_map,
                    allocated_devices=allocated_devices,
                    metadata=dict(metadata),
                    warm_ttl_s=normalized_ttl,
                    last_used_s=now,
                )
                handle = None
        if handle is not None:
            cleanup_payload = dict(metadata)
            memory_payload = _build_memory_payload(actual_key, allocated_devices)
            for key, value in memory_payload.items():
                if key == "model_key":
                    continue
                cleanup_payload[key] = value
            _cleanup_handle(handle)
            _emit_gpu_event("gpu.model.cleanup", cleanup_payload, allocated_devices)
        if not success and reused and cached_entry is not None:
            _evict_cached_entry(cached_entry, reason="error")
        _GPU_LOCK.release()


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
    keep_warm: bool = False,
    warm_ttl_s: Optional[float] = 900.0,
    block_until_gpu_free: bool = True,
) -> _ModelContextManager:
    """Return a context manager usable from sync or async call sites."""

    return _ModelContextManager(
        lambda: _model_context_sync(
            model_key,
            loader=loader,
            weights=weights,
            offload=offload,
            xformers=xformers,
            compile=compile,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            max_memory=max_memory,
            timeout_s=timeout_s,
            keep_warm=keep_warm,
            warm_ttl_s=warm_ttl_s,
            block_until_gpu_free=block_until_gpu_free,
        )
    )


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


def _system_memory_snapshot() -> Optional[Mapping[str, float]]:
    try:
        text = _PROC_MEMINFO_PATH.read_text()
    except Exception:
        return None

    values: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            kb_value = float(parts[0])
        except ValueError:
            continue
        values[key.strip()] = kb_value * 1024  # convert to bytes

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    free = values.get("MemFree") or available
    if total is None or free is None:
        return None

    used = total - (available if available is not None else free)

    def _to_mb(val: float) -> float:
        return round(val / (1024 * 1024), 2)

    snapshot: dict[str, float] = {
        "total_mb": _to_mb(total),
        "free_mb": _to_mb(free),
        "used_mb": _to_mb(used),
    }
    if available is not None:
        snapshot["available_mb"] = _to_mb(available)
    return snapshot


def _collect_memory_snapshot(devices: Sequence[str]) -> Mapping[str, Mapping[str, float]]:
    snapshot: dict[str, Mapping[str, float]] = {}
    normalized = tuple(devices) if devices else ("system",)
    torch_cuda = getattr(torch, "cuda", None)
    cuda_available = False
    if torch_cuda is not None:
        try:
            cuda_available = bool(torch_cuda.is_available())
        except Exception:
            cuda_available = False

    for dev in normalized:
        if dev is None:
            continue
        lowered = dev.lower()
        if lowered.startswith("cuda") and cuda_available:
            try:
                index = int(dev.split(":", 1)[1]) if ":" in dev else torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info(index)
            except Exception:
                continue
            snapshot[dev] = {
                "free_mb": round(free / (1024 * 1024), 2),
                "total_mb": round(total / (1024 * 1024), 2),
                "used_mb": round((total - free) / (1024 * 1024), 2),
            }
        elif lowered.startswith("cpu") or lowered == "system":
            system_stats = _system_memory_snapshot()
            if system_stats:
                snapshot[dev] = system_stats

    if snapshot:
        return snapshot

    system_stats = _system_memory_snapshot()
    if system_stats:
        snapshot["system"] = system_stats
    return snapshot


def _collect_nvml_metrics(devices: Sequence[str]) -> Optional[Mapping[str, Mapping[str, Any]]]:
    sampler = _resolve_nvml_sampler()
    if sampler is None:
        return None
    try:
        metrics = sampler.sample(devices)
    except NvmlError as exc:
        _invalidate_nvml(reason=str(exc), devices=devices)
        return None
    if metrics:
        return metrics
    return None


def _build_memory_payload(model_key: str, devices: Sequence[str]) -> Mapping[str, Any]:
    snapshot = _collect_memory_snapshot(devices)
    nvml_stats = _collect_nvml_metrics(devices)
    payload: dict[str, Any] = {"model_key": model_key}
    if snapshot:
        payload["snapshot"] = snapshot
    if nvml_stats:
        payload["nvml"] = nvml_stats
    return payload


def _resolve_nvml_sampler() -> Optional["_NvmlSampler"]:
    global _NVML_STATE, _NVML_SAMPLER
    if _NVML_STATE == _NVML_STATE_ERROR:
        return None
    if _NVML_STATE == _NVML_STATE_READY and _NVML_SAMPLER is not None:
        return _NVML_SAMPLER
    with _NVML_STATE_LOCK:
        if _NVML_STATE == _NVML_STATE_READY and _NVML_SAMPLER is not None:
            return _NVML_SAMPLER
        sampler = _NvmlSampler.try_create()
        if sampler is None:
            _NVML_STATE = _NVML_STATE_ERROR
            _NVML_SAMPLER = None
            return None
        _NVML_SAMPLER = sampler
        _NVML_STATE = _NVML_STATE_READY
        return sampler


def _invalidate_nvml(*, reason: str, devices: Sequence[str]) -> None:
    global _NVML_STATE, _NVML_SAMPLER, _NVML_ERROR_REPORTED
    _NVML_STATE = _NVML_STATE_ERROR
    _NVML_SAMPLER = None
    if not _NVML_ERROR_REPORTED:
        _emit_gpu_event(
            "gpu.model.nvml_error",
            {"reason": reason},
            tuple(devices),
        )
        _NVML_ERROR_REPORTED = True


def _extract_cuda_indexes(devices: Sequence[str]) -> tuple[int, ...]:
    indexes: set[int] = set()
    for dev in devices:
        if not dev:
            continue
        lowered = dev.lower()
        if not lowered.startswith("cuda"):
            continue
        idx = 0
        if ":" in dev:
            try:
                idx = int(dev.split(":", 1)[1])
            except ValueError:
                idx = 0
        if idx >= 0:
            indexes.add(idx)
    return tuple(sorted(indexes))


def _resolve_nvml_library_path() -> Optional[str]:
    override = os.environ.get("NVML_LIBRARY_PATH")
    if override:
        return override
    candidate = find_library("nvidia-ml")
    if candidate:
        return candidate
    common_paths = (
        Path("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"),
        Path("/usr/lib64/libnvidia-ml.so.1"),
        Path("/usr/lib/libnvidia-ml.so.1"),
    )
    for path in common_paths:
        if path.exists():
            return str(path)
    return None


class _NvmlSampler:
    _TEMPERATURE_SENSOR_GPU = 0
    _MAX_NAME_LENGTH = 96

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib
        self._device_t = ctypes.c_void_p

        class _NvmlUtilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        class _NvmlMemory(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong),
            ]

        self._utilization_t = _NvmlUtilization
        self._memory_t = _NvmlMemory

        self._init = self._bind("nvmlInit_v2") or self._bind("nvmlInit", required=True)
        self._shutdown = self._bind("nvmlShutdown", required=True)
        self._get_count = self._bind("nvmlDeviceGetCount_v2") or self._bind("nvmlDeviceGetCount", required=True)
        self._get_handle = self._bind("nvmlDeviceGetHandleByIndex_v2") or self._bind(
            "nvmlDeviceGetHandleByIndex",
            required=True,
        )
        self._get_name = self._bind("nvmlDeviceGetName")
        self._get_temperature = self._bind("nvmlDeviceGetTemperature")
        self._get_fan_speed = self._bind("nvmlDeviceGetFanSpeed")
        self._get_utilization = self._bind("nvmlDeviceGetUtilizationRates")
        self._get_power_usage = self._bind("nvmlDeviceGetPowerUsage")
        self._get_memory_info = self._bind("nvmlDeviceGetMemoryInfo")
        self._error_string_fn = getattr(self._lib, "nvmlErrorString", None)
        if self._error_string_fn is not None:
            self._error_string_fn.restype = ctypes.c_char_p

    @classmethod
    def try_create(cls) -> Optional["_NvmlSampler"]:
        path = _resolve_nvml_library_path()
        if not path:
            return None
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            return None
        try:
            return cls(lib)
        except NvmlError:
            return None

    def sample(self, devices: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        indexes = _extract_cuda_indexes(devices)
        data: dict[str, Mapping[str, Any]] = {}
        self._check(self._init(), "nvmlInit")
        try:
            total = self._device_count()
            targets = indexes or tuple(range(total))
            for idx in targets:
                if idx >= total:
                    continue
                metrics = self._snapshot_device(idx)
                if metrics:
                    data[f"cuda:{idx}"] = metrics
        finally:
            self._shutdown()
        return data

    def _device_count(self) -> int:
        count = ctypes.c_uint()
        self._check(self._get_count(ctypes.byref(count)), "nvmlDeviceGetCount")
        return int(count.value)

    def _snapshot_device(self, index: int) -> Optional[Mapping[str, Any]]:
        handle = ctypes.c_void_p()
        self._check(
            self._get_handle(ctypes.c_uint(index), ctypes.byref(handle)),
            "nvmlDeviceGetHandleByIndex",
        )
        metrics: dict[str, Any] = {"index": index}
        if self._get_name is not None:
            buffer = ctypes.create_string_buffer(self._MAX_NAME_LENGTH)
            status = self._get_name(handle, buffer, ctypes.c_uint(len(buffer)))
            if self._call_optional(status, "nvmlDeviceGetName"):
                metrics["name"] = buffer.value.decode("utf-8", "replace").strip()
        if self._get_temperature is not None:
            value = ctypes.c_uint()
            status = self._get_temperature(
                handle,
                ctypes.c_uint(self._TEMPERATURE_SENSOR_GPU),
                ctypes.byref(value),
            )
            if self._call_optional(status, "nvmlDeviceGetTemperature"):
                metrics["temperature_c"] = int(value.value)
        if self._get_fan_speed is not None:
            fan = ctypes.c_uint()
            status = self._get_fan_speed(handle, ctypes.byref(fan))
            if self._call_optional(status, "nvmlDeviceGetFanSpeed"):
                metrics["fan_percent"] = int(fan.value)
        if self._get_utilization is not None:
            util = self._utilization_t()
            status = self._get_utilization(handle, ctypes.byref(util))
            if self._call_optional(status, "nvmlDeviceGetUtilizationRates"):
                metrics["utilization_gpu_percent"] = int(util.gpu)
                metrics["utilization_mem_percent"] = int(util.memory)
        if self._get_memory_info is not None:
            mem = self._memory_t()
            status = self._get_memory_info(handle, ctypes.byref(mem))
            if self._call_optional(status, "nvmlDeviceGetMemoryInfo"):
                metrics["memory_total_mb"] = round(mem.total / (1024 * 1024), 2)
                metrics["memory_used_mb"] = round(mem.used / (1024 * 1024), 2)
        if self._get_power_usage is not None:
            power = ctypes.c_uint()
            status = self._get_power_usage(handle, ctypes.byref(power))
            if self._call_optional(status, "nvmlDeviceGetPowerUsage"):
                metrics["power_w"] = round(power.value / 1000.0, 2)
        return metrics

    def _bind(self, name: str, *, required: bool = False):
        fn = getattr(self._lib, name, None)
        if fn is None:
            if required:
                raise NvmlError(f"NVML function '{name}' is unavailable")
            return None
        fn.restype = ctypes.c_int
        return fn

    def _check(self, status: int, fn_name: str) -> None:
        if status == 0:
            return
        raise NvmlError(f"{fn_name} failed: {self._error_string(status)}")

    def _call_optional(self, status: int, fn_name: str) -> bool:
        if status == 0:
            return True
        message = self._error_string(status)
        lowered = message.lower()
        if "not supported" in lowered or "no data" in lowered or "insufficient permissions" in lowered:
            return False
        raise NvmlError(f"{fn_name} failed: {message}")

    def _error_string(self, status: int) -> str:
        if self._error_string_fn is None:
            return f"status={status}"
        raw = self._error_string_fn(status)
        return raw.decode("utf-8", "replace") if raw else f"status={status}"


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


def _wrap_load_exception(
    exc: Exception,
    *,
    model_key: str,
    stage: str,
    device_map: Optional[Mapping[str, str]],
) -> ModelContextError:
    if isinstance(exc, ModelContextError):
        return exc
    if isinstance(exc, RuntimeError):
        maybe_oom = _normalize_oom(exc, model_key=model_key, stage=stage, device_map=device_map)
        if isinstance(maybe_oom, ModelContextError):
            return maybe_oom
    return ModelLoadError(
        message=f"{model_key} failed during {stage}: {exc}",
        model_key=model_key,
        stage=stage,
        retryable=False,
    )
