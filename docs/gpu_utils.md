# GPU Utilities Reference

This guide explains how to work with `sparkle_motion.gpu_utils`, focusing on
`model_context`, the shared device presets, emitted telemetry, and the most common
failure modes you will encounter while running heavyweight adapters.

## Why `model_context` exists

All FunctionTools and adapters that touch GPU-bound pipelines must wrap their load
and inference phases inside `gpu_utils.model_context`. Doing so ensures:

- **Deterministic lifecycle** — models load once per `with` block and are fully
  cleaned up (CUDA sync, cache flush, GC) even when errors bubble out.
- **Shared GPU lock** — only one heavy pipeline may own the device at a time,
  preventing VRAM contention in multi-tool workflows.
- **Structured telemetry** — every load, inference, cache event, and cleanup gets
  recorded via `telemetry.emit_event()` and `adk_helpers.write_memory_event()`.
- **Warm caching** — adapters can opt in to `keep_warm=True` to reuse a pipeline
  for subsequent invocations without reloading weights.
- **Uniform error semantics** — OOMs, timeouts, and lock conflicts are surfaced as
  dedicated exception types (`ModelOOMError`, `ModelLoadTimeoutError`, `GpuBusyError`).

## Quick-start usage

```python
from sparkle_motion.gpu_utils import model_context, ModelOOMError

MODEL_ID = "stabilityai/sdxl-base-1.0"

try:
    with model_context(
        "images/sdxl",
        loader=lambda: load_sdxl_pipeline(MODEL_ID),
        weights=MODEL_ID,
        device_map={"unet": "cuda:0"},
        keep_warm=True,
        warm_ttl_s=600,
    ) as ctx:
        pipe = ctx.pipeline
        with ctx.inference_span(label="denoise", extra={"steps": 30}):
            images = pipe(prompt="A hero shot", num_inference_steps=30)
        publish_images(images)
except ModelOOMError as exc:
    log.warning("OOM for %s on %s", exc.model_key, exc.device_map)
    raise
```

Key takeaways from this example:

1. Always pass a **loader callable**. `model_context` never instantiates models for you.
2. The `model_key` identifies telemetry streams *and* cache entries, so choose a
   stable string per logical pipeline (e.g. `tts/chatterbox`).
3. Wrap expensive inference chunks inside `ctx.inference_span(...)` to emit timing
   events and to benefit from automatic OOM normalization.

## API reference

### `model_context()` parameters

| Parameter | Type | Purpose |
| --- | --- | --- |
| `model_key` | `str` | Logical identifier used for telemetry, caching, and error messages. Must be non-empty. |
| `loader` | `Callable[[], Any]` | Required factory that returns the fully initialized pipeline/model. |
| `weights` | `Optional[str]` | Informational tag (HF repo, checkpoint path) recorded in metadata. |
| `offload` | `bool` (default `True`) | Hint for adapters to enable CPU/GPU offload utilities. Stored in metadata for observability. |
| `xformers` | `bool` (default `True`) | Whether callers intend to enable xFormers attention optimizations. |
| `compile` | `bool` | Records whether `torch.compile()` was requested. |
| `device_map` | `Optional[Mapping[str, str]]` | Explicit placement for model submodules. Drives both telemetry and `allocated_devices`. |
| `low_cpu_mem_usage` | `bool` | Set to `True` for HF pipelines that support streaming weight loading. |
| `max_memory` | `Optional[Mapping[str, str]]` | Memory budgets passed to HF loaders (e.g. `{0: "74GiB", "cpu": "120GiB"}`). |
| `timeout_s` | `Optional[float]` | Max seconds to wait for the loader. Raises `ModelLoadTimeoutError` on expiry. |
| `keep_warm` | `bool` | When `True`, keeps the pipeline cached after exit. Future calls with the same `model_key` can reuse it. |
| `warm_ttl_s` | `Optional[float]` | Cache lifetime in seconds. `None` disables expiry; `<=0` disables caching. |
| `block_until_gpu_free` | `bool` | Controls whether the shared GPU lock waits. When `False`, raises `GpuBusyError` immediately if the lock is held. |

### Returned `ModelContext`

`ModelContext` instances contain:

- `pipeline`: whatever the loader returned (usually your `diffusers` or custom pipeline).
- `device_map`: the resolved placement mapping.
- `allocated_devices`: tuple of devices that currently hold weights (derived from `device_map` or best effort).
- `metadata`: mutable dict echoed into telemetry events (seeded with the keyword arguments).
- `report_memory()`: captures `{device: {free_mb, used_mb, total_mb}}` via CUDA APIs when available.
- `inference_span(label="default", extra=None)`: nested context manager to log inference start/end events and wrap runtime OOMs into `ModelOOMError`.

### Helper utilities

- `gpu_utils.gpu_is_busy()` — returns `True` if another `model_context` owns the GPU lock.
- `gpu_utils.evict_cached_model(model_key: Optional[str] = None)` — force-drop the warm cache globally, optionally filtering by key.
- `gpu_utils.compute_device_map(preset, overrides=None)` — copy a registered preset and optionally tweak placements.
- `gpu_utils.list_device_presets()` — read-only view of all preset names and mappings.

## Device-map presets

`compute_device_map()` exposes a few turnkey placements tuned for the hardware we use in CI and smoke rigs:

| Preset | Mapping |
| --- | --- |
| `a100-80gb` | `{text_encoder: cuda:0, unet: cuda:0, vae: cuda:0, controlnet: cuda:0}` |
| `a100-40gb` | `{text_encoder: cuda:0, unet: cuda:0, vae: cuda:0}` |
| `rtx4090` | `{text_encoder: cuda:0, unet: cuda:0, vae: cuda:0}` |

Guidance:

1. Treat preset keys as case-insensitive. `compute_device_map("A100-80GB")` works.
2. Use `overrides` to pin individual submodules to different devices or to add CPU offload entries. Overrides win over preset defaults.
3. Log the final device map in adapter metadata so QA can correlate VRAM usage with placement decisions.

## Telemetry lifecycle

`gpu_utils` fires structured events for every notable step. All payloads include
`model_key`, `allocated_devices`, and the metadata captured at entry.

| Event | When it fires | Notable fields |
| --- | --- | --- |
| `gpu.model.load_start` | Right before executing the loader | `weights`, `device_map`, `max_memory`, `offload`, `xformers` |
| `gpu.model.load_complete` | Loader returned successfully | `duration_ms`, `allocated_devices` |
| `gpu.model.load_error` | Loader raised (surfaced via exception) | Emitted implicitly through `ModelOOMError` / exception path |
| `gpu.model.cache_store` | A context exits successfully with `keep_warm=True` | `ttl_s` |
| `gpu.model.cache_reuse` | A cached handle gets reused | `idle_ms`, `metadata` snapshot |
| `gpu.model.cache_evict` | Cache gets dropped (expiry, manual, error) | `reason` |
| `gpu.model.inference_start` | `ctx.inference_span()` entered | `label`, optional user-supplied extras |
| `gpu.model.inference_end` | `ctx.inference_span()` exited | `duration_ms` |
| `gpu.model.cleanup` | Context exits and a handle is actually destroyed | `device_map`, `offload`, memory snapshot via `report_memory()` |

Every telemetry emission is mirrored into the ADK memory log with
`adk_helpers.write_memory_event()` so you can audit GPU usage after the fact.

## Warm cache semantics

- Only one cached model may exist at a time (global singleton).
- `keep_warm=True` stores the handle along with `warm_ttl_s` and `metadata`.
- The cache is invalidated when:
  - A different `model_key` is requested.
  - TTL expires (`now - last_used_s > warm_ttl_s`).
  - Callers pass `keep_warm=False` (forces eviction after the run).
  - An error occurs while reusing the cached handle.
- Use `evict_cached_model()` when switching between incompatible adapters or after running gated smoke tests to reclaim VRAM proactively.

## Troubleshooting

### `GpuBusyError`
- **Cause**: Another adapter currently holds the GPU lock.
- **Fix**: Either set `block_until_gpu_free=True` (default) to wait, or catch the
  exception and retry after a backoff. Monitoring `gpu_utils.gpu_is_busy()` can
  help preflight your workload.

### `ModelLoadTimeoutError`
- **Cause**: Loader exceeded `timeout_s`.
- **Fix**: Instrument your loader to log progress and ensure weight downloads are
  local before invoking the FunctionTool. Increase `timeout_s` only after
  verifying the pipeline is healthy.

### `ModelOOMError`
- **Cause**: CUDA reported OOM during load or inference.
- **Fix**:
  1. Inspect `exc.memory_snapshot` for per-device usage.
  2. Shrink batch/chunk sizes or lower resolution before retrying.
  3. Enable offload/xformers/`low_cpu_mem_usage` flags if they were disabled.
  4. Consider switching to a different device preset (e.g., offload VAE to CPU).

### Cache thrash or stale weights
- Use `evict_cached_model()` between smoke tests to drop lingering handles.
- When switching between fixtures and real adapters, prefer distinct `model_key`
  strings so cached fixtures cannot be mistaken for real weights.

### Missing CUDA metrics
- `report_memory()` best-effort queries CUDA. On CPU-only hosts it simply returns
  `{}`. Adapters should fall back to conservative defaults instead of assuming
  the snapshot will be populated.

## Adapter checklist

1. Wrap every heavy inference call in `with model_context(...):`.
2. Pass an explicit `loader` that **only** builds the pipeline (no side effects).
3. Log the `ctx.metadata` along with adapter-specific info (prompt, plan_id, etc.).
4. Use `ctx.inference_span()` for every chunk to get consistent telemetry.
5. Catch `ModelOOMError` to implement adaptive retries where appropriate.
6. Call `gpu_utils.compute_device_map()` for known hardware instead of ad-hoc
   strings; keep overrides localized so future hardware additions remain easy.
7. Provide configuration switches (`SMOKE_*`) that flip `keep_warm`/`warm_ttl_s`
   according to local dev vs. CI needs.

## Further reading

- `src/sparkle_motion/gpu_utils.py` — authoritative implementation.
- `tests/unit/test_gpu_and_adapters.py` and `tests/unit/test_device_map.py` — cover
  cache behavior, error normalization, and preset validation.
- `docs/IMPLEMENTATION_TASKS.md` (§ "Cross-cutting: gpu_utils.model_context") — architectural
  deep dive and future enhancements (NVML, adaptive shrink helpers).
