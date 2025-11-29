from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from typing_extensions import Literal, Protocol, TypedDict

from . import adk_helpers, observability, telemetry
from .gpu_utils import ModelOOMError
from .utils import dedupe


class VideoAgentError(RuntimeError):
    """Base error for videos_agent orchestration issues."""


class PlanPolicyViolation(VideoAgentError):
    """Raised when a prompt violates content or safety policy."""


class ChunkExecutionError(VideoAgentError):
    """Raised when a chunk exhausts retries/fallbacks."""


class CallbackEvent(TypedDict, total=False):
    plan_id: str
    step_id: str
    chunk_index: int
    frame_index: int
    num_frames: int
    progress: float
    eta_s: Optional[float]
    device: Optional[str]
    phase: str


@dataclass(frozen=True)
class VideoAgentConfig:
    num_frames: int
    chunk_length_frames: int = 64
    chunk_overlap_frames: int = 4
    min_chunk_frames: int = 8
    max_retries_per_chunk: int = 2
    adaptive_shrink_factor: float = 0.5
    allow_cpu_fallback: bool = True
    artifact_type: str = "videos_agent_clip"
    reassembly_mode: Literal["trim"] = "trim"
    debug_frames: bool = False

    @classmethod
    def from_opts(cls, opts: Mapping[str, Any]) -> "VideoAgentConfig":
        num_frames = int(opts.get("num_frames", 0))
        if num_frames <= 0:
            raise ValueError("opts['num_frames'] must be a positive integer")

        chunk_length = int(opts.get("chunk_length_frames", cls.chunk_length_frames))
        min_chunk = int(opts.get("min_chunk_frames", cls.min_chunk_frames))
        overlap = int(opts.get("chunk_overlap_frames", cls.chunk_overlap_frames))
        if chunk_length < min_chunk:
            chunk_length = min_chunk
        max_retries = int(opts.get("max_retries_per_chunk", cls.max_retries_per_chunk))
        adaptive = float(opts.get("adaptive_shrink_factor", cls.adaptive_shrink_factor))
        allow_cpu = bool(opts.get("allow_cpu_fallback", cls.allow_cpu_fallback))
        artifact_type = str(opts.get("artifact_type", cls.artifact_type))
        debug_frames = bool(opts.get("debug_frames", cls.debug_frames))
        return cls(
            num_frames=num_frames,
            chunk_length_frames=chunk_length,
            chunk_overlap_frames=max(0, overlap),
            min_chunk_frames=min_chunk,
            max_retries_per_chunk=max(0, max_retries),
            adaptive_shrink_factor=adaptive if adaptive > 0 else cls.adaptive_shrink_factor,
            allow_cpu_fallback=allow_cpu,
            artifact_type=artifact_type,
            debug_frames=debug_frames,
        )


@dataclass(frozen=True)
class ChunkSpec:
    chunk_index: int
    start_frame: int
    end_frame: int
    overlap_left: int
    overlap_right: int
    render_start: int
    render_end: int
    seed: Optional[int]

    @property
    def logical_length(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def render_length(self) -> int:
        return self.render_end - self.render_start + 1


@dataclass
class ChunkRenderResult:
    chunk: ChunkSpec
    frames: Sequence[Any]
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    artifact_path: Optional[Path] = None


@dataclass(frozen=True)
class VideoAdapterContext:
    prompt: str
    start_frames: Sequence[bytes]
    end_frames: Sequence[bytes]
    opts: Mapping[str, Any]
    progress_callback: Optional[Callable[[CallbackEvent], None]] = None


class VideoChunkRenderer(Protocol):
    def __call__(self, chunk: ChunkSpec, context: VideoAdapterContext) -> ChunkRenderResult:
        ...


class _ProgressDispatcher:
    def __init__(
        self,
        *,
        plan_id: str,
        step_id: str,
        run_id: str,
        callback: Optional[Callable[[CallbackEvent], None]],
    ) -> None:
        self._plan_id = plan_id
        self._step_id = step_id
        self._run_id = run_id
        self._callback = callback

    def emit(self, payload: Mapping[str, Any]) -> None:
        enriched: CallbackEvent = CallbackEvent(
            plan_id=self._plan_id,
            step_id=self._step_id,
            **{k: v for k, v in payload.items() if k not in {"plan_id", "step_id"}},
        )
        if self._callback:
            self._callback(enriched)
        try:
            adk_helpers.write_memory_event(
                run_id=self._run_id,
                event_type="videos_agent.progress",
                payload=dict(enriched),
            )
        except adk_helpers.MemoryWriteError:
            pass

    def adapter_callback(self, chunk_index: int) -> Callable[[CallbackEvent], None]:
        def _cb(event: CallbackEvent) -> None:
            merged = dict(event)
            merged.setdefault("chunk_index", chunk_index)
            self.emit(merged)

        return _cb


def render_video(
    start_frames: Iterable[Any],
    end_frames: Iterable[Any],
    prompt: str,
    opts: Optional[Mapping[str, Any]] = None,
    *,
    on_progress: Optional[Callable[[CallbackEvent], None]] = None,
    adapter: Optional[VideoChunkRenderer] = None,
) -> adk_helpers.ArtifactRef:
    """Render a clip by chunking requests to the configured adapter."""

    options: MutableMapping[str, Any] = dict(opts or {})
    config = VideoAgentConfig.from_opts(options)
    plan_id = str(options.get("plan_id") or "plan-unknown")
    step_id = str(options.get("step_id") or "videos_agent")
    run_id = str(options.get("run_id") or observability.get_session_id())
    _validate_prompt(prompt)

    dedupe_enabled = bool(options.get("dedupe", False))
    recent_index = dedupe.resolve_recent_index(
        enabled=dedupe_enabled,
        backend=options.get("recent_index"),
        use_sqlite=options.get("recent_index_use_sqlite"),
        db_path=options.get("recent_index_db_path"),
    )

    dispatcher = _ProgressDispatcher(plan_id=plan_id, step_id=step_id, run_id=run_id, callback=on_progress or options.get("on_progress"))

    start_payload = _coerce_frames(start_frames)
    end_payload = _coerce_frames(end_frames)
    chunk_renderer: VideoChunkRenderer = adapter or options.get("chunk_renderer") or _deterministic_chunk_renderer

    chunk_specs = _build_chunks(
        num_frames=config.num_frames,
        chunk_length=config.chunk_length_frames,
        overlap=config.chunk_overlap_frames,
        seed=options.get("seed"),
    )

    telemetry.emit_event(
        "videos_agent.render.start",
        {"plan_id": plan_id, "step_id": step_id, "chunks": len(chunk_specs)},
    )

    chunk_records: List[MutableMapping[str, Any]] = []
    assembled_frames: List[Any] = []
    adapter_base_opts = dict(options.get("adapter_opts") or {})
    chunk_attempts: List[dict[str, Any]] = []

    for spec in chunk_specs:
        result, attempt_record = _execute_chunk(
            spec,
            chunk_renderer,
            dispatcher,
            prompt,
            start_payload,
            end_payload,
            adapter_base_opts,
            config,
        )
        chunk_attempts.append(attempt_record)
        trimmed_frames = _trim_frames(result.frames, spec)
        assembled_frames.extend(trimmed_frames)
        record = {
            "chunk_index": spec.chunk_index,
            "chunk_start_frame": spec.start_frame,
            "chunk_end_frame": spec.end_frame,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "attempts": attempt_record["attempts"],
            "cpu_fallback": attempt_record["cpu_fallback"],
            "seed": spec.seed,
        }
        if result.metadata:
            record["adapter_metadata"] = dict(result.metadata)
        chunk_records.append(record)

    payload = {
        "plan_id": plan_id,
        "step_id": step_id,
        "prompt": prompt,
        "frame_count": len(assembled_frames),
        "chunks": chunk_records,
    }
    if config.debug_frames:
        payload["frames"] = list(assembled_frames)

    payload_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    digest = dedupe.compute_hash(payload_bytes)

    artifact_metadata: dict[str, Any] = {
        "plan_id": plan_id,
        "step_id": step_id,
        "prompt": prompt,
        "num_frames": config.num_frames,
        "chunks": chunk_records,
    }
    if config.debug_frames:
        artifact_metadata["assembled_frames"] = assembled_frames
    artifact_metadata["attempts"] = chunk_attempts

    if dedupe_enabled and recent_index is not None:
        canonical = recent_index.get(digest)
        if canonical:
            recent_index.get_or_add(digest, canonical)
            metadata = dict(artifact_metadata)
            metadata["deduped"] = True
            metadata["duplicate_of"] = canonical
            artifact = {
                "uri": canonical,
                "artifact_type": config.artifact_type,
                "media_type": "application/json",
                "metadata": metadata,
                "storage": "adk" if canonical.startswith("artifact://") else "local",
                "run_id": run_id,
            }
            telemetry.emit_event(
                "videos_agent.render.completed",
                {"plan_id": plan_id, "step_id": step_id, "artifact_uri": canonical, "deduped": True},
            )
            return artifact

    output_path = _write_result_file(
        plan_id=plan_id,
        step_id=step_id,
        payload_bytes=payload_bytes,
        opts=options,
    )

    artifact = adk_helpers.publish_artifact(
        local_path=output_path,
        artifact_type=config.artifact_type,
        metadata=artifact_metadata,
    )
    if recent_index is not None:
        canonical = recent_index.get_or_add(digest, artifact["uri"])
        if canonical != artifact["uri"]:
            artifact["metadata"] = dict(artifact.get("metadata") or {})
            artifact["metadata"]["deduped"] = True
            artifact["metadata"]["duplicate_of"] = canonical
            artifact["uri"] = canonical
            artifact["storage"] = "adk" if canonical.startswith("artifact://") else artifact.get("storage", "local")
    telemetry.emit_event(
        "videos_agent.render.completed",
        {"plan_id": plan_id, "step_id": step_id, "artifact_uri": artifact["uri"]},
    )
    return artifact


def _execute_chunk(
    spec: ChunkSpec,
    renderer: VideoChunkRenderer,
    dispatcher: _ProgressDispatcher,
    prompt: str,
    start_payload: Sequence[bytes],
    end_payload: Sequence[bytes],
    adapter_base_opts: Mapping[str, Any],
    config: VideoAgentConfig,
) -> tuple[ChunkRenderResult, MutableMapping[str, Any]]:
    max_attempts = max(1, 1 + config.max_retries_per_chunk)
    adaptive_length = spec.logical_length
    cpu_fallback = False
    attempt_records: List[dict[str, Any]] = []
    total_attempts = 0
    mode_attempts = 0

    def _record(outcome: str, opts_snapshot: Mapping[str, Any]) -> None:
        attempt_records.append(
            {
                "attempt": total_attempts,
                "chunk_length_frames": opts_snapshot["chunk_length_frames"],
                "device": opts_snapshot.get("device", "auto"),
                "outcome": outcome,
            }
        )

    while True:
        total_attempts += 1
        mode_attempts += 1
        attempt_opts = dict(adapter_base_opts)
        attempt_opts.setdefault("chunk_length_frames", adaptive_length)
        attempt_opts.setdefault("chunk_index", spec.chunk_index)
        attempt_opts.setdefault("render_start", spec.render_start)
        attempt_opts.setdefault("render_end", spec.render_end)
        if cpu_fallback:
            attempt_opts["device"] = "cpu"

        dispatcher.emit({"chunk_index": spec.chunk_index, "phase": "chunk.start", "progress": 0.0})
        context = VideoAdapterContext(
            prompt=prompt,
            start_frames=start_payload,
            end_frames=end_payload,
            opts=attempt_opts,
            progress_callback=dispatcher.adapter_callback(spec.chunk_index),
        )
        try:
            result = renderer(spec, context)
            dispatcher.emit({"chunk_index": spec.chunk_index, "phase": "chunk.complete", "progress": 1.0})
            _record("success", attempt_opts)
            meta = {
                "attempts": total_attempts,
                "cpu_fallback": cpu_fallback,
                "attempt_history": attempt_records,
            }
            if result.metadata:
                result.metadata.setdefault("attempts", total_attempts)
            return result, meta
        except ModelOOMError as exc:
            _record("oom", attempt_opts)
            adaptive_length = max(int(adaptive_length * config.adaptive_shrink_factor), config.min_chunk_frames)
            if mode_attempts >= max_attempts:
                if config.allow_cpu_fallback and not cpu_fallback:
                    cpu_fallback = True
                    mode_attempts = 0
                    adaptive_length = spec.logical_length
                    continue
                raise ChunkExecutionError(f"Chunk {spec.chunk_index} OOM after {total_attempts} attempts") from exc
        except Exception as exc:
            _record("error", attempt_opts)
            if mode_attempts >= max_attempts:
                if config.allow_cpu_fallback and not cpu_fallback:
                    cpu_fallback = True
                    mode_attempts = 0
                    continue
                raise ChunkExecutionError(f"Chunk {spec.chunk_index} failed: {exc}") from exc



def _build_chunks(
    *,
    num_frames: int,
    chunk_length: int,
    overlap: int,
    seed: Optional[Any],
) -> List[ChunkSpec]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")

    specs: List[ChunkSpec] = []
    start = 0
    chunk_index = 0
    overlap = max(0, overlap)
    while start < num_frames:
        end = min(start + chunk_length - 1, num_frames - 1)
        overlap_left = overlap if chunk_index > 0 else 0
        overlap_right = overlap if end < num_frames - 1 else 0
        render_start = max(0, start - overlap_left)
        render_end = min(num_frames - 1, end + overlap_right)
        chunk_seed = _derive_chunk_seed(seed, chunk_index)
        specs.append(
            ChunkSpec(
                chunk_index=chunk_index,
                start_frame=start,
                end_frame=end,
                overlap_left=overlap_left,
                overlap_right=overlap_right,
                render_start=render_start,
                render_end=render_end,
                seed=chunk_seed,
            )
        )
        start = end + 1
        chunk_index += 1
    return specs


def _derive_chunk_seed(seed: Optional[Any], chunk_index: int) -> Optional[int]:
    if seed is None:
        return None
    data = f"{seed}:{chunk_index}".encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    return int(digest[:8], 16)


def _trim_frames(frames: Sequence[Any], spec: ChunkSpec) -> List[Any]:
    trimmed = list(frames)
    if spec.overlap_left > 0:
        trimmed = trimmed[spec.overlap_left :]
    if spec.overlap_right > 0:
        trimmed = trimmed[: -spec.overlap_right]
    expected_length = spec.logical_length
    if len(trimmed) > expected_length:
        trimmed = trimmed[:expected_length]
    return trimmed


def _coerce_frames(sources: Iterable[Any]) -> List[bytes]:
    frames: List[bytes] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, bytes):
            frames.append(source)
        elif isinstance(source, (str, Path)):
            frames.append(Path(source).read_bytes())
        else:
            raise TypeError(f"Unsupported frame reference: {type(source)!r}")
    return frames


def _validate_prompt(prompt: str) -> None:
    lowered = prompt.lower()
    banned = {"weaponized", "forbidden"}
    if any(word in lowered for word in banned):
        raise PlanPolicyViolation("Prompt contains banned content")


def _write_result_file(
    *,
    plan_id: str,
    step_id: str,
    payload_bytes: bytes,
    opts: Mapping[str, Any],
) -> Path:
    output_path = opts.get("output_path")
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(opts.get("output_dir", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        dest = base / f"{plan_id}-{step_id}-video.json"

    dest.write_bytes(payload_bytes)
    return dest


def _deterministic_chunk_renderer(chunk: ChunkSpec, context: VideoAdapterContext) -> ChunkRenderResult:
    frames = list(range(chunk.render_start, chunk.render_end + 1))
    metadata = {
        "seed": chunk.seed,
        "render_length": chunk.render_length,
    }
    return ChunkRenderResult(chunk=chunk, frames=frames, metadata=metadata)


__all__ = [
    "render_video",
    "PlanPolicyViolation",
    "ChunkExecutionError",
    "VideoAgentError",
    "VideoChunkRenderer",
    "CallbackEvent",
    "ChunkRenderResult",
    "VideoAdapterContext",
    "ChunkSpec",
]
