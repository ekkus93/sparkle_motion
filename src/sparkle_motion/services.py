from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class SessionContext:
    """Lightweight session context used by the orchestrator and services."""

    run_id: str
    run_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionService:
    """Creates per-run directories and contextual metadata.

    This is intentionally simple so alternate implementations (e.g., ADK's
    SessionService or a remote run manager) can be swapped in later.
    """

    def __init__(self, runs_root: Path | str = "runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def create_session(self, run_id: Optional[str] = None) -> SessionContext:
        rid = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        run_dir = self.runs_root / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        return SessionContext(run_id=rid, run_dir=run_dir)


class ArtifactService:
    """Tracks artifacts (files, directories, URIs) produced during a run."""

    def __init__(self, session: SessionContext) -> None:
        self.session = session
        self.registry_path = session.run_dir / "artifacts.json"
        self._registry: Dict[str, Dict[str, Any]] = {}
        if self.registry_path.exists():
            try:
                self._registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                self._registry = {}

    def register(self, *, name: str, path: Path | str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._registry[name] = {"path": str(path), "metadata": metadata or {}}
        self._sync()

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._registry.get(name)

    def _sync(self) -> None:
        self.registry_path.write_text(json.dumps(self._registry, indent=2), encoding="utf-8")


class MemoryService:
    """Persists run-level notes, QA decisions, and human feedback."""

    def __init__(self, session: SessionContext) -> None:
        self.session = session
        self.log_path = session.run_dir / "memory_log.json"
        if self.log_path.exists():
            try:
                self.entries = json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                self.entries = []
        else:
            self.entries = []

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload,
        }
        self.entries.append(entry)
        self.log_path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        return entry

    def record_human_feedback(self, *, stage: str, decision: str, notes: Optional[str] = None) -> Dict[str, Any]:
        return self.record_event(
            "human_feedback",
            {
                "stage": stage,
                "decision": decision,
                "notes": notes,
            },
        )

    def list_events(self) -> list[Dict[str, Any]]:
        """Return a shallow copy of the recorded events for external aggregation."""

        return list(self.entries)


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable
    config: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Minimal Tool catalog to mirror ADK's FunctionTool registry."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, *, name: str, description: str, func: Callable, config: Optional[Dict[str, Any]] = None) -> None:
        self._tools[name] = ToolSpec(name=name, description=description, func=func, config=config or {})

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def invoke(self, name: str, *args, **kwargs):
        spec = self.get(name)
        return spec.func(*args, **kwargs)

    def list_tools(self) -> Dict[str, ToolSpec]:
        return dict(self._tools)