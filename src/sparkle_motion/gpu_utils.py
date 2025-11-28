from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterator, Optional


@contextmanager
def model_context(name: str, *, weights: Optional[str] = None, **kwargs) -> Iterator[Any]:
    """Lightweight context manager to load/unload model resources safely.

    This is a scaffold. Real implementations MUST perform lazy imports inside
    the context (so module import does not require heavy deps) and must
    implement deterministic CUDA/device cleanup (torch.cuda.empty_cache(),
    context timely deletion, etc.).

    Usage (example):
        with model_context('sdxl', weights='stabilityai/sdxl-base-1.0') as ctx:
            pipe = ctx.pipeline  # adapter-specific
            outputs = pipe(...)

    The returned context object is intentionally minimal; adapters may return
    whatever handle they need (pipeline, model, tokenizer, etc.).
    """
    # NOTE: keep imports lazy to avoid requiring torch/diffusers on import.
    class _DummyCtx:
        def __init__(self) -> None:
            self.model = None

        def __enter__(self) -> "_DummyCtx":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return False

    # A real implementation would look like:
    # try:
    #     import torch
    #     from diffusers import DiffusionPipeline
    #     # load pipeline with specified dtype and device placement
    #     pipe = DiffusionPipeline.from_pretrained(weights, torch_dtype=torch.float16)
    #     pipe.to('cuda')
    #     ctx = SimpleNamespace(pipeline=pipe)
    #     yield ctx
    # finally:
    #     # cleanup: delete, torch.cuda.empty_cache(), etc.
    #     del pipe
    #     torch.cuda.empty_cache()

    yield _DummyCtx()
from __future__ import annotations

import contextlib
import gc
import os
from typing import Any, Callable, Generator, Optional

try:
    import torch
except Exception:
    torch = None  # optional, only used when available


@contextlib.contextmanager
def model_context(load_fn: Callable[[], Any], name: Optional[str] = None) -> Generator[Any, None, None]:
    """Context manager that loads a model/resource and ensures cleanup.

    - `load_fn` is a callable that returns a loaded model or resource handle.
    - On exit, attempts to call common teardown methods and clears CUDA cache
      if `torch` is available.
    - In test/fixture mode (`ADK_USE_FIXTURE=1`) the loader is called but
      teardown becomes a no-op to keep unit tests deterministic.
    """
    fixture = os.environ.get("ADK_USE_FIXTURE") == "1"
    handle = None
    try:
        handle = load_fn()
        yield handle
    finally:
        if handle is None:
            return

        # Note: even in fixture mode we still attempt light-weight teardown
        # (call `close()`/`shutdown()` if present). Tests expect the model's
        # `close` to be invoked for deterministic cleanup. Heavy GPU cache
        # clearing remains gated by torch.cuda.is_available().

        # Try common close/shutdown methods
        for meth in ("close", "shutdown", "stop", "release", "unload", "dispose"):
            fn = getattr(handle, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

        # Attempt to clear CUDA caches if torch is present
        try:
            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

        # Force a GC pass
        try:
            gc.collect()
        except Exception:
            pass
