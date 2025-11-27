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
