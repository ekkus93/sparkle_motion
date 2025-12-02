from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from . import adk_helpers, telemetry
from .filesystem_artifacts.config import FilesystemArtifactsConfig
from .filesystem_artifacts.models import ArtifactRecord
from .filesystem_artifacts.storage import FilesystemArtifactStore
from .schemas import StageManifest
from .utils.env import filesystem_backend_enabled

RunStatus = Literal["pending", "running", "paused", "stopped", "failed", "succeeded", "queued"]
QAMode = Literal["full", "skip"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_FILESYSTEM_FETCH_LIMIT = 200
_filesystem_store: Optional[FilesystemArtifactStore] = None
_filesystem_store_config: Optional[FilesystemArtifactsConfig] = None


def _get_filesystem_store() -> FilesystemArtifactStore:
    global _filesystem_store, _filesystem_store_config
    config = FilesystemArtifactsConfig.from_env()
    if _filesystem_store is None or _filesystem_store_config != config:
        _filesystem_store = FilesystemArtifactStore(config)
        _filesystem_store_config = config
    return _filesystem_store


def _reset_filesystem_store_for_tests() -> None:  # pragma: no cover - helper
    global _filesystem_store, _filesystem_store_config
    _filesystem_store = None
    _filesystem_store_config = None


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
    qa_skipped: Optional[bool] = None
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
            "qa_skipped": self.qa_skipped,
            "playback_ready": self.playback_ready,
            "notes": self.notes,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


def _artifact_identity(entry: ArtifactEntry) -> str:
    candidate = entry.artifact_uri or ""
    if candidate:
        return candidate
    return f"{entry.stage}:{entry.name}:{entry.created_at}"


def _iso_from_epoch(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value:
        return value
    return _now_iso()


def _entry_from_stage_manifest(manifest: StageManifest, *, record: Optional[ArtifactRecord] = None) -> ArtifactEntry:
    metadata = dict(manifest.metadata or {})
    artifact_uri = manifest.artifact_uri
    local_path = manifest.local_path
    download_url = manifest.download_url
    storage_hint: Optional[str] = manifest.storage_hint
    media_type = manifest.media_type or manifest.mime_type
    if record is not None:
        metadata.setdefault("filesystem_manifest_path", record.storage.manifest_path)
        metadata.setdefault("filesystem_relative_path", record.storage.relative_path)
        metadata["storage_backend"] = "filesystem"
        metadata.setdefault("artifact_uri", record.artifact_uri)
        artifact_uri = record.artifact_uri
        storage_hint = "filesystem"
        local_path = record.storage.absolute_path
        download_url = record.manifest.get("download_url") or download_url
        if not media_type:
            media_type = record.mime_type
    checksum = manifest.checksum_sha256
    if checksum is None and record is not None:
        checksum = (record.manifest.get("checksum") or {}).get("sha256")
    return ArtifactEntry(
        stage=manifest.stage_id,
        artifact_type=manifest.artifact_type,
        name=manifest.name,
        artifact_uri=artifact_uri,
        media_type=media_type,
        local_path=local_path,
        download_url=download_url,
        storage_hint=storage_hint,
        mime_type=manifest.mime_type,
        size_bytes=manifest.size_bytes,
        duration_s=manifest.duration_s,
        frame_rate=manifest.frame_rate,
        resolution_px=manifest.resolution_px,
        checksum_sha256=checksum,
        qa_report_uri=manifest.qa_report_uri,
        qa_passed=manifest.qa_passed,
        qa_mode=manifest.qa_mode,
        qa_skipped=manifest.qa_skipped,
        playback_ready=manifest.playback_ready,
        notes=manifest.notes,
        metadata=metadata,
        created_at=manifest.created_at,
    )


def _entry_from_artifact_record(record: ArtifactRecord) -> ArtifactEntry:
    manifest_payload = record.manifest
    metadata = dict(manifest_payload.get("metadata") or record.metadata or {})
    snapshot = metadata.get("stage_manifest_snapshot")
    if isinstance(snapshot, dict):
        try:
            stage_manifest = StageManifest.model_validate(snapshot)
            return _entry_from_stage_manifest(stage_manifest, record=record)
        except Exception:
            pass
    checksum_payload = manifest_payload.get("checksum")
    checksum = checksum_payload.get("sha256") if isinstance(checksum_payload, dict) else None
    metadata.setdefault("filesystem_manifest_path", record.storage.manifest_path)
    metadata.setdefault("filesystem_relative_path", record.storage.relative_path)
    metadata.setdefault("storage_backend", "filesystem")
    metadata.setdefault("artifact_uri", record.artifact_uri)
    local_path = manifest_payload.get("local_path") or metadata.get("local_path") or record.storage.absolute_path
    download_url = manifest_payload.get("download_url") or metadata.get("download_url")
    created_at = _iso_from_epoch(manifest_payload.get("created_at") or record.created_at)
    return ArtifactEntry(
        stage=record.stage,
        artifact_type=record.artifact_type,
        name=metadata.get("name") or record.artifact_type,
        artifact_uri=record.artifact_uri,
        media_type=manifest_payload.get("mime_type") or record.mime_type,
        local_path=local_path,
        download_url=download_url,
        storage_hint="filesystem",
        mime_type=record.mime_type,
        size_bytes=manifest_payload.get("size_bytes"),
        duration_s=metadata.get("duration_s"),
        frame_rate=metadata.get("frame_rate"),
        resolution_px=metadata.get("resolution_px"),
        checksum_sha256=checksum,
        qa_report_uri=metadata.get("qa_report_uri"),
        qa_passed=metadata.get("qa_passed"),
        qa_mode=metadata.get("qa_mode"),
        qa_skipped=metadata.get("qa_skipped"),
        playback_ready=metadata.get("playback_ready"),
        notes=metadata.get("notes"),
        metadata=metadata,
        created_at=created_at,
    )


def _entry_to_stage_manifest(entry: ArtifactEntry, run_id: str) -> Dict[str, Any]:
    manifest = StageManifest(
        run_id=run_id,
        stage_id=entry.stage,
        artifact_type=entry.artifact_type,
        name=entry.name,
        artifact_uri=entry.artifact_uri,
        media_type=entry.media_type,
        local_path=entry.local_path,
        download_url=entry.download_url,
        storage_hint=entry.storage_hint,
        mime_type=entry.mime_type,
        size_bytes=entry.size_bytes,
        duration_s=entry.duration_s,
        frame_rate=entry.frame_rate,
        resolution_px=entry.resolution_px,
        checksum_sha256=entry.checksum_sha256,
        qa_report_uri=entry.qa_report_uri,
        qa_passed=entry.qa_passed,
        qa_mode=entry.qa_mode,
        qa_skipped=entry.qa_skipped,
        playback_ready=entry.playback_ready,
        notes=entry.notes,
        metadata=dict(entry.metadata),
        created_at=entry.created_at,
    )
    return manifest.model_dump()

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
    qa_skipped: bool = False
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
                qa_skipped=qa_mode == "skip",
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
                    qa_skipped=meta.get("qa_skipped"),
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
                if filesystem_backend_enabled():
                    return self._filesystem_status_payload(run_id)
                raise KeyError(run_id)
            payload = self._serialize_state(state)
            memory_groups = self._snapshot_artifacts(state, None)
        payload["artifact_counts"] = self._compute_artifact_counts(run_id, memory_groups)
        return payload

    def get_artifacts(self, run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = self._collect_artifact_entries(run_id, stage)
        return [entry.as_dict() for entry in entries]

    def get_artifacts_by_stage(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        groups = self._collect_artifact_groups(run_id, None)
        return {stage_name: [entry.as_dict() for entry in entries] for stage_name, entries in groups.items()}

    def list_artifacts(self, run_id: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = self._collect_artifact_entries(run_id, stage)
        return [_entry_to_stage_manifest(entry, run_id) for entry in entries]

    def _filesystem_status_payload(self, run_id: str) -> Dict[str, Any]:
        entries = self._load_filesystem_artifacts(run_id, None)
        if not entries:
            raise KeyError(run_id)
        stage_groups = _group_entries_by_stage(entries)
        artifact_counts = {stage: len(items) for stage, items in stage_groups.items()}
        started_at = _select_timestamp(entries, prefer_min=True) or _now_iso()
        updated_at = _select_timestamp(entries, prefer_min=False) or started_at
        plan_id = _first_metadata_value(entries, "plan_id") or run_id
        plan_title = _first_metadata_value(entries, "plan_title") or plan_id
        qa_mode = _first_metadata_value(entries, "qa_mode") or "full"
        qa_skipped_flag = _first_metadata_value(entries, "qa_skipped")
        qa_skipped = bool(qa_skipped_flag)
        schema_uri = _first_metadata_value(entries, "schema_uri")
        completed = _filesystem_run_completed(entries)
        current_stage = _last_stage_name(stage_groups)
        steps = self._build_filesystem_step_records(stage_groups)
        timeline = [self._format_timeline_entry(record, qa_mode) for record in steps]
        render_profile: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {"source": "filesystem_fallback"}
        context_payload = _load_run_context_payload(entries)
        if context_payload:
            plan_title = context_payload.get("plan_title") or plan_title
            context_render = context_payload.get("render_profile")
            if isinstance(context_render, dict):
                render_profile = dict(context_render)
            context_meta = context_payload.get("metadata")
            if isinstance(context_meta, dict):
                metadata.update(context_meta)
            schema_uri = context_payload.get("schema_uri") or schema_uri
        metadata.setdefault("plan_id", plan_id)
        progress = 1.0 if completed else (0.0 if not steps else min(0.95, len(steps) / max(len(steps) + 1, 1)))
        expected_steps = len(stage_groups) or None
        status = "succeeded" if completed else "unknown"
        payload = {
            "run_id": run_id,
            "plan_id": plan_id,
            "plan_title": plan_title,
            "mode": "run",
            "status": status,
            "started_at": started_at,
            "updated_at": updated_at,
            "current_stage": current_stage,
            "progress": progress,
            "expected_steps": expected_steps,
            "last_error": None,
            "steps": steps,
            "artifact_counts": artifact_counts,
            "control": {
                "pause_requested": False,
                "stop_requested": False,
                "last_command": {},
            },
            "metadata": metadata,
            "render_profile": render_profile,
            "qa_mode": qa_mode,
            "qa_skipped": qa_skipped,
            "schema_uri": schema_uri,
            "timeline": timeline,
            "log": list(timeline),
        }
        return payload

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

    def build_progress_handler(self, run_id: str) -> Callable[[Any], None]:
        """Return a progress callback that tolerates dataclass or dict input."""

        def _handler(record_like: Any) -> None:
            if hasattr(record_like, "as_dict"):
                record_dict = record_like.as_dict()
            elif isinstance(record_like, dict):
                record_dict = record_like
            else:  # pragma: no cover - defensive conversion guard
                raise TypeError(
                    f"Progress handler expected StepExecutionRecord or dict, got {type(record_like)!r}"
                )
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
            "qa_skipped": state.qa_skipped,
            "schema_uri": state.schema_uri,
            "timeline": timeline,
            "log": timeline,
        }

    def _snapshot_artifacts(self, state: Optional[RunState], stage_filter: Optional[str]) -> Dict[str, List[ArtifactEntry]]:
        if state is None:
            return {}
        if stage_filter:
            entries = list(state.artifacts.get(stage_filter, []))
            return {stage_filter: entries} if entries else {}
        return {stage: list(entries) for stage, entries in state.artifacts.items()}

    def _collect_artifact_groups(self, run_id: str, stage_filter: Optional[str]) -> Dict[str, List[ArtifactEntry]]:
        normalized_stage = stage_filter.strip() if stage_filter else None
        with self._lock:
            state = self._runs.get(run_id)
            if not state and not filesystem_backend_enabled():
                raise KeyError(run_id)
            memory_groups = self._snapshot_artifacts(state, normalized_stage)
        merged = self._merge_with_filesystem(run_id, normalized_stage, memory_groups)
        if normalized_stage:
            return {normalized_stage: list(merged.get(normalized_stage, []))}
        return merged

    def _collect_artifact_entries(self, run_id: str, stage_filter: Optional[str]) -> List[ArtifactEntry]:
        groups = self._collect_artifact_groups(run_id, stage_filter)
        if stage_filter:
            normalized = stage_filter.strip()
            return list(groups.get(normalized, []))
        entries: List[ArtifactEntry] = []
        for stage_entries in groups.values():
            entries.extend(stage_entries)
        return entries

    def _merge_with_filesystem(
        self,
        run_id: str,
        stage_filter: Optional[str],
        base: Dict[str, List[ArtifactEntry]],
    ) -> Dict[str, List[ArtifactEntry]]:
        grouped: Dict[str, List[ArtifactEntry]] = {stage: list(entries) for stage, entries in base.items()}
        if not filesystem_backend_enabled():
            return grouped
        seen: Dict[str, set[str]] = {
            stage: {_artifact_identity(entry) for entry in entries} for stage, entries in grouped.items()
        }
        filesystem_entries = self._load_filesystem_artifacts(run_id, stage_filter)
        for entry in filesystem_entries:
            stage_name = entry.stage
            if stage_filter and stage_name != stage_filter:
                continue
            bucket = grouped.setdefault(stage_name, [])
            stage_seen = seen.setdefault(stage_name, set())
            identity = _artifact_identity(entry)
            if identity in stage_seen:
                continue
            bucket.append(entry)
            stage_seen.add(identity)
        for entries in grouped.values():
            entries.sort(key=lambda item: item.created_at)
        return grouped

    def _compute_artifact_counts(self, run_id: str, base: Dict[str, List[ArtifactEntry]]) -> Dict[str, int]:
        merged = self._merge_with_filesystem(run_id, None, base)
        return {stage: len(entries) for stage, entries in merged.items()}

    def _build_filesystem_step_records(self, stage_groups: Dict[str, List[ArtifactEntry]]) -> List[Dict[str, Any]]:
        ordered: List[Tuple[str, Dict[str, Any]]] = []
        for stage, entries in stage_groups.items():
            if not entries:
                continue
            sorted_entries = sorted(entries, key=lambda item: item.created_at or "")
            start_time = sorted_entries[0].created_at
            end_time = sorted_entries[-1].created_at
            record = {
                "step_id": stage,
                "step_type": stage,
                "status": "succeeded",
                "start_time": start_time,
                "end_time": end_time,
                "duration_s": _duration_between(start_time, end_time),
                "attempts": 1,
                "artifact_uri": sorted_entries[-1].artifact_uri,
                "meta": {
                    "storage": "filesystem",
                    "artifacts": [entry.artifact_uri for entry in sorted_entries],
                    "artifact_types": sorted({entry.artifact_type for entry in sorted_entries}),
                },
            }
            ordered.append((start_time or "", record))
        ordered.sort(key=lambda item: item[0])
        return [record for _, record in ordered]

    def _load_filesystem_artifacts(self, run_id: str, stage_filter: Optional[str]) -> List[ArtifactEntry]:
        if not filesystem_backend_enabled():
            return []
        try:
            store = _get_filesystem_store()
        except Exception:
            return []
        results: List[ArtifactEntry] = []
        page_marker: Optional[Tuple[int, str]] = None
        while True:
            try:
                records, _ = store.list_artifacts(
                    run_id=run_id,
                    stage=stage_filter,
                    artifact_type=None,
                    limit=_FILESYSTEM_FETCH_LIMIT,
                    order="asc",
                    page_marker=page_marker,
                )
            except Exception:
                break
            if not records:
                break
            for record in records:
                results.append(_entry_from_artifact_record(record))
            if len(records) < _FILESYSTEM_FETCH_LIMIT:
                break
            last = records[-1]
            page_marker = (last.created_at, last.artifact_id)
        return results

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
            "qa_skipped": qa_mode == "skip",
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


def _group_entries_by_stage(entries: Iterable[ArtifactEntry]) -> Dict[str, List[ArtifactEntry]]:
    grouped: Dict[str, List[ArtifactEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.stage, []).append(entry)
    for bucket in grouped.values():
        bucket.sort(key=lambda item: item.created_at or "")
    return grouped


def _select_timestamp(entries: Iterable[ArtifactEntry], *, prefer_min: bool) -> Optional[str]:
    stamps = [entry.created_at for entry in entries if entry.created_at]
    if not stamps:
        return None
    return min(stamps) if prefer_min else max(stamps)


def _first_metadata_value(entries: Iterable[ArtifactEntry], key: str) -> Any:
    for entry in entries:
        value = entry.metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _last_stage_name(stage_groups: Dict[str, List[ArtifactEntry]]) -> Optional[str]:
    latest: Optional[ArtifactEntry] = None
    for bucket in stage_groups.values():
        if not bucket:
            continue
        candidate = bucket[-1]
        if latest is None or (candidate.created_at or "") > (latest.created_at or ""):
            latest = candidate
    return latest.stage if latest else None


def _duration_between(start_iso: Optional[str], end_iso: Optional[str]) -> Optional[float]:
    start_dt = _coerce_datetime(start_iso)
    end_dt = _coerce_datetime(end_iso)
    if not start_dt or not end_dt:
        return None
    delta = (end_dt - start_dt).total_seconds()
    return max(0.0, delta)


def _coerce_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    token = value
    if token.endswith("Z"):
        token = f"{token[:-1]}+00:00"
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        return None


def _filesystem_run_completed(entries: Iterable[ArtifactEntry]) -> bool:
    return any(entry.stage == "finalize" and entry.artifact_type == "video_final" for entry in entries)


def _load_run_context_payload(entries: Iterable[ArtifactEntry]) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if entry.artifact_type != "plan_run_context":
            continue
        local_path = entry.local_path
        if not local_path:
            continue
        payload = _safe_read_json(local_path)
        if isinstance(payload, dict):
            return payload
    return None


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        data = Path(path).read_text(encoding="utf-8")
        payload = json.loads(data)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
