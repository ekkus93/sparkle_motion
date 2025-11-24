from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import tempfile


@dataclass
class StageEvent:
    run_id: str
    stage: str
    status: str  # begin | success | fail
    timestamp: float
    attempt: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": self.status,
            "timestamp": self.timestamp,
            "attempt": self.attempt,
            "error": self.error,
            "metadata": self.metadata or {},
        }


class RunManifest:
    """Simple run manifest that records stage events and can persist to disk.

    Usage:
      m = RunManifest(run_id='r1', path=Path('runs/r1/manifest.json'))
      m.add_event(StageEvent(...))
      m.save()
    """

    def __init__(self, run_id: str, path: Optional[Path] = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path is not None else None
        self.events: List[Dict[str, Any]] = []

    def add_event(self, event: StageEvent) -> None:
        self.events.append(event.to_dict())

    def save(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path is not None else self.path
        if p is None:
            raise RuntimeError("No path provided for saving manifest")
        # Ensure parent exists
        p.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically: write to a temp file in the same directory, fsync,
        # then atomically replace the target path. This avoids partial-file
        # states if the process is killed while writing.
        data = json.dumps({"run_id": self.run_id, "events": self.events}, indent=2)
        dirpath = str(p.parent)
        fd = None
        tmp_path = None
        try:
            # Create a named temporary file in the same directory
            fd, tmp_path = tempfile.mkstemp(prefix=".manifest-", dir=dirpath)
            # Write bytes and force to disk
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = None
            # Atomically replace the target file
            os.replace(tmp_path, str(p))
            tmp_path = None
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        m = cls(run_id=data.get("run_id", ""), path=path)
        m.events = data.get("events", [])
        return m

    def last_status_for_stage(self, stage: str) -> Optional[str]:
        """Return the most recent status for a given stage, or None if no events.

        Status values are strings like "begin", "success", "fail" as recorded
        in the events list. This helper scans events in reverse to find the
        latest matching stage entry.
        """
        for ev in reversed(self.events):
            if ev.get("stage") == stage:
                return ev.get("status")
        return None


def retry(
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: float = 0.2,
    stage_name: Optional[str] = None,
):
    """Decorator that retries a function and optionally logs events to a RunManifest.

    The decorated function may accept a keyword argument `manifest` (RunManifest)
    to which begin/fail/success events will be written. If not provided, no
    manifest events are recorded.
    """

    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            manifest: Optional[RunManifest] = kwargs.get("manifest")
            run_id = getattr(manifest, "run_id", "unknown") if manifest else "unknown"
            stage = stage_name or getattr(fn, "__name__", "stage")

            attempt = 0
            last_exc: Optional[BaseException] = None
            if manifest:
                manifest.add_event(StageEvent(run_id=run_id, stage=stage, status="begin", timestamp=time.time(), attempt=0, metadata={}))
            while attempt < max_attempts:
                attempt += 1
                try:
                    result = fn(*args, **kwargs)
                    if manifest:
                        manifest.add_event(StageEvent(run_id=run_id, stage=stage, status="success", timestamp=time.time(), attempt=attempt, metadata={}))
                    return result
                except Exception as e:
                    last_exc = e
                    if manifest:
                        manifest.add_event(
                            StageEvent(
                                run_id=run_id,
                                stage=stage,
                                status="fail",
                                timestamp=time.time(),
                                attempt=attempt,
                                error=str(e),
                            )
                        )
                    if attempt >= max_attempts:
                        break
                    # exponential backoff with jitter
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay = delay * (1.0 + jitter * (random.random() * 2 - 1))
                    time.sleep(max(0.0, delay))
            # if we reach here, all attempts failed
            raise last_exc

        # preserve function identity
        _wrapped.__name__ = fn.__name__
        _wrapped.__doc__ = fn.__doc__
        return _wrapped

    return _decorator


__all__ = ["StageEvent", "RunManifest", "retry"]
