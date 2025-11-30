from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Callable, Dict, List, Literal, Optional

from . import adk_helpers, telemetry

RunStatus = Literal["pending", "running", "paused", "stopped", "failed", "succeeded", "queued"]
QAMode = Literal["full", "skip"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ArtifactEntry:
    stage: str
    artifact_type: str
    name: str
    artifact_uri: str
    media_type: Optional[str] = None
    local_path: Optional[str] = None
    download_url: Optional[str] = None
    storage_hint: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    duration_s: Optional[float] = None
    frame_rate: Optional[float] = None
    resolution_px: Optional[str] = None
    checksum_sha256: Optional[str] = None
    qa_report_uri: Optional[str] = None
    qa_passed: Optional[bool] = None
    qa_mode: Optional[str] = None
    playback_ready: Optional[bool] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "artifact_type": self.artifact_type,
            "name": self.name,
            "artifact_uri": self.artifact_uri,
            "media_type": self.media_type,
            "local_path": self.local_path,
            "download_url": self.download_url,
            "storage_hint": self.storage_hint,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "duration_s": self.duration_s,
            "frame_rate": self.frame_rate,
            "resolution_px": self.resolution_px,
            "checksum_sha256": self.checksum_sha256,
            "qa_report_uri": self.qa_report_uri,
            "qa_passed": self.qa_passed,
            "qa_mode": self.qa_mode,
            "playback_ready": self.playback_ready,
            "notes": self.notes,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class AsyncControlGate:
    """Dual-mode gate that mirrors state between threading and asyncio events."""

    def __init__(self) -> None:
        self._sync_event = threading.Event()
        self._sync_event.set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_event: Optional[asyncio.Event] = None
        self._gate_lock = threading.RLock()

    def wait_sync(self) -> None:
        self._sync_event.wait()

    async def wait_async(self) -> None:
        event = self._ensure_async_event()
        await event.wait()

    def set(self) -> None:
        self._sync_event.set()
        self._notify_async("set")

    def clear(self) -> None:
        self._sync_event.clear()
        self._notify_async("clear")

    def _ensure_async_event(self) -> asyncio.Event:
        loop = asyncio.get_running_loop()
        with self._gate_lock:
            if self._async_event is None or self._loop is not loop:
                self._async_event = asyncio.Event()
                self._loop = loop
                if self._sync_event.is_set():
                    self._async_event.set()
            return self._async_event

    def _notify_async(self, action: Literal["set", "clear"]) -> None:
        with self._gate_lock:
            event = self._async_event
            loop = self._loop
        if not event or not loop or loop.is_closed():
            return
        callback = event.set if action == "set" else event.clear
        loop.call_soon_threadsafe(callback)


@dataclass
class ControlState:
    pause_requested: bool = False
    stop_requested: bool = False
    gate: AsyncControlGate = field(default_factory=AsyncControlGate)
    last_command: Optional[Dict[str, Any]] = None


@dataclass
class RunState:
    run_id: str
    plan_id: str
    plan_title: str
    mode: Literal["dry", "run"]
    status: RunStatus = "pending"
    started_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    current_stage: Optional[str] = None
    progress: float = 0.0
    expected_steps: Optional[int] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, List[ArtifactEntry]] = field(default_factory=dict)
    last_error: Optional[str] = None
    control: ControlState = field(default_factory=ControlState)
    render_profile: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    qa_mode: QAMode = "full"
    schema_uri: Optional[str] = None

    def append_step(self, record: Dict[str, Any]) -> None:
        self.steps.append(record)
        self.current_stage = record.get("step_id")
        self.progress = self._calculate_progress()
        self.updated_at = _now_iso()

    def _calculate_progress(self) -> float:
        if not self.expected_steps:
            return min(0.99, len(self.steps) / max(len(self.steps) + 1, 1))
        return min(1.0, len(self.steps) / max(self.expected_steps, 1))


class RunHalted(RuntimeError):
    pass


class RunRegistry:
    def __init__(self) -> None:
        self._runs: Dict[str, RunState] = {}
        self._lock = threading.RLock()

    def start_run(
        self,
        *,
        run_id: str,
        plan_id: str,
        plan_title: str,
        mode: Literal["dry", "run"],
        expected_steps: Optional[int] = None,
        render_profile: Optional[Dict[str, Any]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        qa_mode: QAMode = "full",
        schema_uri: Optional[str] = None,
    ) -> RunState:
        with self._lock:
            state = RunState(
                run_id=run_id,
                plan_id=plan_id,
                plan_title=plan_title,
                mode=mode,
                status="running" if mode == "run" else "pending",
                current_stage=None,
                expected_steps=expected_steps,
                render_profile=dict(render_profile or {}),
                metadata=dict(run_metadata or {}),
                qa_mode=qa_mode,
                schema_uri=schema_uri,
            )
            self._runs[run_id] = state
        self._emit_event("run.start", state)
        return state

    def mark_queued(self, run_id: str, ticket_id: str) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return
            state.status = "queued"
            state.updated_at = _now_iso()
        self._emit_event("run.queued", state, extra={"ticket_id": ticket_id})

    def complete_run(self, run_id: str) -> None:
        self._update_status(run_id, "succeeded")

    def fail_run(self, run_id: str, *, error: str) -> None:
        self._update_status(run_id, "failed", error=error)

    def stop_run(self, run_id: str, *, reason: str) -> None:
        self._update_status(run_id, "stopped", error=reason)

    def _update_status(self, run_id: str, status: RunStatus, *, error: Optional[str] = None) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return
            state.status = status
            state.updated_at = _now_iso()
            state.last_error = error
        self._emit_event("run.status", state)

    def discard_run(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]

    def record_step(self, run_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return
            state.append_step(record)
            state.status = "running"
            artifact_uri = record.get("artifact_uri")
            if artifact_uri:
                meta = record.get("meta") or {}
                entry = ArtifactEntry(
                    stage=record.get("step_type", "unknown"),
                    artifact_type=meta.get("artifact_type", "step_artifact"),
                    name=record.get("step_id", "artifact"),
                    artifact_uri=str(artifact_uri),
                    media_type=meta.get("media_type"),
                    local_path=meta.get("local_path"),
                    download_url=meta.get("download_url"),
                    storage_hint=meta.get("storage_hint"),
                    mime_type=meta.get("mime_type"),
                    size_bytes=meta.get("size_bytes"),
                    duration_s=meta.get("duration_s"),
                    frame_rate=meta.get("frame_rate"),
                    resolution_px=meta.get("resolution_px"),
                    checksum_sha256=meta.get("checksum_sha256"),
                    qa_report_uri=meta.get("qa_report_uri"),
                    qa_passed=meta.get("qa_passed"),
                    qa_mode=meta.get("qa_mode"),
                    playback_ready=meta.get("playback_ready"),
                    notes=meta.get("notes"),
                    metadata=dict(meta),
                )
                state.artifacts.setdefault(entry.stage, []).append(entry)
        self._emit_event("run.step", state, extra={"record": record})

    def record_artifact(self, run_id: str, entry: ArtifactEntry) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                return
            state.artifacts.setdefault(entry.stage, []).append(entry)
            state.updated_at = _now_iso()
        self._emit_event("run.artifact", state, extra={"artifact": entry.as_dict()})

    def get_status(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                raise KeyError(run_id)
            return self._serialize_state(state)

    def get_artifacts(self, run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                raise KeyError(run_id)
            if stage:
                return [entry.as_dict() for entry in state.artifacts.get(stage, [])]
            result: List[Dict[str, Any]] = []
            for entries in state.artifacts.values():
                result.extend(entry.as_dict() for entry in entries)
            return result

    def get_artifacts_by_stage(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                raise KeyError(run_id)
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for stage_name, entries in state.artifacts.items():
                grouped[stage_name] = [entry.as_dict() for entry in entries]
            return grouped

    def pre_step_hook(self, run_id: str) -> Callable[[str], None]:
        def _hook(step_id: str) -> None:
            with self._lock:
                state = self._runs.get(run_id)
                if not state:
                    return
                state.current_stage = step_id
                state.status = "paused" if state.control.pause_requested else "running"
                state.updated_at = _now_iso()
                control = state.control
            if control.stop_requested:
                raise RunHalted(f"Run {run_id} stopped")
            if control.pause_requested:
                control.gate.wait_sync()
                with self._lock:
                    resumed_state = self._runs.get(run_id)
                    if resumed_state:
                        resumed_state.status = "running"
                        resumed_state.updated_at = _now_iso()
                if control.stop_requested:
                    raise RunHalted(f"Run {run_id} stopped")

        return _hook

    def build_progress_handler(self, run_id: str) -> Callable[[Dict[str, Any]], None]:
        def _handler(record_dict: Dict[str, Any]) -> None:
            self.record_step(run_id, record_dict)

        return _handler

    def request_pause(self, run_id: str) -> Dict[str, Any]:
        return self._update_control(run_id, command="pause")

    def request_resume(self, run_id: str) -> Dict[str, Any]:
        return self._update_control(run_id, command="resume")

    def request_stop(self, run_id: str) -> Dict[str, Any]:
        return self._update_control(run_id, command="stop")

    def _update_control(self, run_id: str, *, command: Literal["pause", "resume", "stop"]) -> Dict[str, Any]:
        with self._lock:
            state = self._runs.get(run_id)
            if not state:
                raise KeyError(run_id)
            control = state.control
            if command == "pause":
                control.pause_requested = True
                control.gate.clear()
                state.status = "paused"
            elif command == "resume":
                control.pause_requested = False
                control.gate.set()
                state.status = "running"
            elif command == "stop":
                control.stop_requested = True
                control.gate.set()
                state.status = "stopped"
            control.last_command = {"command": command, "at": _now_iso()}
            state.updated_at = _now_iso()
        self._emit_event("run.control", state, extra={"command": command})
        return self._serialize_state(state)

    def _serialize_state(self, state: RunState) -> Dict[str, Any]:
        timeline = [self._format_timeline_entry(record, state.qa_mode) for record in state.steps]
        return {
            "run_id": state.run_id,
            "plan_id": state.plan_id,
            "plan_title": state.plan_title,
            "mode": state.mode,
            "status": state.status,
            "started_at": state.started_at,
            "updated_at": state.updated_at,
            "current_stage": state.current_stage,
            "progress": state.progress,
            "expected_steps": state.expected_steps,
            "last_error": state.last_error,
            "steps": list(state.steps),
            "artifact_counts": {stage: len(entries) for stage, entries in state.artifacts.items()},
            "control": {
                "pause_requested": state.control.pause_requested,
                "stop_requested": state.control.stop_requested,
                "last_command": dict(state.control.last_command or {}),
            },
            "metadata": dict(state.metadata),
            "render_profile": dict(state.render_profile),
            "qa_mode": state.qa_mode,
            "schema_uri": state.schema_uri,
            "timeline": timeline,
            "log": timeline,
        }

    def _format_timeline_entry(self, record: Dict[str, Any], qa_mode: QAMode) -> Dict[str, Any]:
        meta = dict(record.get("meta") or {})
        artifacts: List[str] = []
        artifact_uri = record.get("artifact_uri")
        if artifact_uri:
            artifacts.append(str(artifact_uri))
        extra_artifacts = meta.get("artifacts")
        if isinstance(extra_artifacts, list):
            artifacts.extend(str(item) for item in extra_artifacts)
        return {
            "step_id": record.get("step_id"),
            "stage": record.get("step_type"),
            "status": record.get("status"),
            "started_at": record.get("start_time"),
            "completed_at": record.get("end_time"),
            "duration_s": record.get("duration_s"),
            "attempts": record.get("attempts"),
            "model_id": record.get("model_id"),
            "device": record.get("device"),
            "artifact_uri": artifact_uri,
            "artifacts": artifacts,
            "qa_mode": qa_mode,
            "meta": meta,
        }

    def _emit_event(self, event_type: str, state: RunState, *, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "run_id": state.run_id,
            "plan_id": state.plan_id,
            "status": state.status,
            "current_stage": state.current_stage,
            "progress": state.progress,
        }
        if extra:
            payload.update(extra)
        try:
            adk_helpers.write_memory_event(run_id=state.run_id, event_type=f"production_agent.{event_type}", payload=payload)
        except adk_helpers.MemoryWriteError:
            pass
        try:
            telemetry.emit_event(f"production_agent.{event_type}", payload)
        except Exception:
            pass


_registry = RunRegistry()


def get_run_registry() -> RunRegistry:
    return _registry


__all__ = [
    "ArtifactEntry",
    "AsyncControlGate",
    "RunRegistry",
    "RunHalted",
    "get_run_registry",
]
