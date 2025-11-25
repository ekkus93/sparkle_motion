from __future__ import annotations

import asyncio
import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session as ADKSession
from google.adk.tools.function_tool import FunctionTool
from google.genai import types


def _run_async(coro):
    """Helper to run ADK async APIs from the synchronous orchestrator."""

    return asyncio.run(coro)


def _guess_mime_type(path: Path) -> Optional[str]:
    mime, _ = mimetypes.guess_type(str(path))
    return mime


def _artifact_part_from_path(path: Path, embed_threshold_bytes: int = 1_000_000) -> types.Part:
    """Create an ADK artifact part for a filesystem path.

    We treat small text files (<1MB) as inline payloads; larger files are
    referenced by absolute path so we avoid duplicating multi-GB media.
    """

    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if size <= embed_threshold_bytes:
        mime = _guess_mime_type(path) or "application/octet-stream"
        data = path.read_bytes()
        return types.Part.from_bytes(data=data, mime_type=mime)

    return types.Part(text=str(path))


@dataclass
class SessionContext:
    """Session metadata backed by the official ADK Session model."""

    run_id: str
    run_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    adk_session: ADKSession | None = None


class SessionService:
    """Creates per-run directories while delegating to ADK's SessionService."""

    def __init__(
        self,
        runs_root: Path | str = "runs",
        *,
        app_name: str = "sparkle_motion",
        user_id: str = "local_user",
        adk_service: Optional[InMemorySessionService] = None,
    ) -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.app_name = app_name
        self.user_id = user_id
        self._adk_service = adk_service or InMemorySessionService()

    def create_session(self, run_id: Optional[str] = None) -> SessionContext:
        try:
            adk_session = _run_async(
                self._adk_service.create_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=run_id,
                    state={"session": {"requested_run_id": run_id}},
                )
            )
        except AlreadyExistsError:
            if not run_id:
                raise
            existing = _run_async(
                self._adk_service.get_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=run_id,
                )
            )
            if existing is None:
                raise
            adk_session = existing
        rid = adk_session.id
        run_dir = self.runs_root / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        return SessionContext(run_id=rid, run_dir=run_dir, metadata=adk_session.state, adk_session=adk_session)

    @property
    def adk_service(self) -> InMemorySessionService:
        return self._adk_service


class ArtifactService:
    """Tracks artifacts using ADK's artifact service plus a local index."""

    def __init__(
        self,
        session: SessionContext,
        *,
        app_name: str = "sparkle_motion",
        user_id: str = "local_user",
        adk_artifact_service: Optional[InMemoryArtifactService] = None,
    ) -> None:
        self.session = session
        self.app_name = app_name
        self.user_id = user_id
        self._adk_artifact_service = adk_artifact_service or InMemoryArtifactService()
        self.registry_path = session.run_dir / "artifacts.json"
        self._registry: Dict[str, Dict[str, Any]] = {}
        if self.registry_path.exists():
            try:
                self._registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                self._registry = {}

    def register(self, *, name: str, path: Path | str, metadata: Optional[Dict[str, Any]] = None) -> None:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _artifact_part_from_path(artifact_path)
        custom_metadata = dict(metadata or {})
        custom_metadata.setdefault("path", str(artifact_path))
        version = _run_async(
            self._adk_artifact_service.save_artifact(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session.run_id,
                filename=name,
                artifact=payload,
                custom_metadata=custom_metadata,
            )
        )
        self._registry[name] = {
            "path": str(artifact_path),
            "metadata": custom_metadata,
            "adk_version": version,
        }
        self._sync()

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._registry.get(name)

    def _sync(self) -> None:
        self.registry_path.write_text(json.dumps(self._registry, indent=2), encoding="utf-8")

    @property
    def adk_artifact_service(self) -> InMemoryArtifactService:
        return self._adk_artifact_service


class MemoryService:
    """Persists run-level notes locally and via ADK's memory service."""

    def __init__(
        self,
        session: SessionContext,
        *,
        memory_service: Optional[InMemoryMemoryService] = None,
    ) -> None:
        self.session = session
        self.log_path = session.run_dir / "memory_log.json"
        if self.log_path.exists():
            try:
                self.entries = json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                self.entries = []
        else:
            self.entries = []
        self._adk_memory = memory_service or InMemoryMemoryService()

    def _persist_entry(self, entry: Dict[str, Any]) -> None:
        self.entries.append(entry)
        self.log_path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")

    def _emit_to_adk_memory(self, entry: Dict[str, Any]) -> None:
        if not self.session.adk_session:
            return
        content = types.Content(parts=[types.Part(text=json.dumps(entry, ensure_ascii=False))])
        event = Event(author="runner", content=content)
        self.session.adk_session.events.append(event)
        self.session.adk_session.last_update_time = entry["timestamp"]
        _run_async(self._adk_memory.add_session_to_memory(self.session.adk_session))

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload,
        }
        self._persist_entry(entry)
        self._emit_to_adk_memory(entry)
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
        return list(self.entries)

    @property
    def adk_memory_service(self) -> InMemoryMemoryService:
        return self._adk_memory


@dataclass
class ToolSpec:
    name: str
    description: str
    function_tool: FunctionTool
    config: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registers stage callables as ADK FunctionTools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, *, name: str, description: str, func: Callable, config: Optional[Dict[str, Any]] = None) -> None:
        function_tool = FunctionTool(func)
        function_tool.name = name
        function_tool.description = description
        metadata = dict(config or {})
        if metadata:
            function_tool.custom_metadata = metadata
        self._tools[name] = ToolSpec(name=name, description=description, function_tool=function_tool, config=metadata)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def invoke(self, name: str, *args, **kwargs):
        spec = self.get(name)
        # Direct invocation keeps the pipeline synchronous while still storing
        # the ADK FunctionTool metadata for future orchestration.
        return spec.function_tool.func(*args, **kwargs)

    def list_tools(self) -> Dict[str, ToolSpec]:
        return dict(self._tools)