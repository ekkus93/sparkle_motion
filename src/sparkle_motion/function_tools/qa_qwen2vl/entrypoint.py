from __future__ import annotations

"""FunctionTool entrypoint for Qwen2-VL powered visual QA checks.

Purpose & usage:
- Validates base images, lipsync frames, or assembled videos against the
    studio’s QA policy. Invoke whenever a workflow stage needs automated approval
    or escalation before publishing artifacts.

Request payload (`QaQwen2VlRequest`):
- `frames` (list[QaFramePayload], required): each item supplies `id`, `uri` or
    inline `data_b64`, plus an optional per-frame `prompt`.
- `prompt` (str, optional): fallback textual description applied to frames that
    omit their own prompt.
- `plan_id`, `run_id`, `step_id`, `movie_title`, `metadata`: provenance fields
    embedded in QA artifacts and telemetry.
- `options` (`QaQwen2VlOptions`): controls inference backend (fixture vs real),
    Qwen model id, dtype/attention strategy, policy path, pixel limits, cache TTL,
    download limits, and nested metadata overrides.

Response dictionary (`QaQwen2VlResponse`):
- `status`: always "success" when the tool returns normally.
- `request_id`: unique call identifier.
- `decision`: "approve", "regenerate", "escalate", or "pending" per QA policy.
- `artifact_uri`: URI pointing to the persisted QAReport artifact.
- `metadata`: structured engine/options/analysis metadata for auditing.
- `report`: serialized QAReport (summary + per-shot findings).
- `human_task_id`: present when escalation created a human-review task.
"""

import asyncio
import base64
import binascii
import logging
import os
import time
import uuid
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Tuple
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sparkle_motion import adk_factory, observability, telemetry
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion.function_tools.qa_qwen2vl import adapter
from sparkle_motion.function_tools.qa_qwen2vl.models import (
    QaFramePayload,
    QaQwen2VlOptions,
    QaQwen2VlRequest,
    QaQwen2VlResponse,
)

LOG = logging.getLogger("qa_qwen2vl.entrypoint")
LOG.setLevel(logging.INFO)

DEFAULT_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_DOWNLOAD_TIMEOUT_S = 10.0


def _safe_send_telemetry(event: str, payload: dict[str, Any]) -> None:
    try:
        send_telemetry(event, payload)
    except Exception:
        LOG.debug("send_telemetry failed", exc_info=True)


def _safe_emit(event: str, payload: dict[str, Any]) -> None:
    try:
        telemetry.emit_event(event, payload)
    except Exception:
        LOG.debug("telemetry.emit_event failed", exc_info=True)


def _sanitize_validation_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for err in errors:
        payload = dict(err)
        ctx = payload.get("ctx")
        if isinstance(ctx, dict):
            payload["ctx"] = {k: (str(v) if isinstance(v, BaseException) else v) for k, v in ctx.items()}
        sanitized.append(payload)
    return sanitized


def _options_to_dict(options: QaQwen2VlOptions | None) -> Dict[str, Any]:
    if not options:
        return {}
    return options.model_dump(exclude_none=True)


# Preserve historical attribute names for shared entrypoint tests.
RequestModel = QaQwen2VlRequest
ResponseModel = QaQwen2VlResponse


def make_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.shutting_down = False
        app.state.inflight = 0
        app.state.lock = Lock()
        try:
            delay = float(os.environ.get("MODEL_LOAD_DELAY", "0"))
        except Exception:
            delay = 0.0
        if delay > 0:
            LOG.info("Warmup: delay=%s", delay)
            await asyncio.sleep(delay)
        try:
            model_spec = os.environ.get("QA_QWEN2VL_MODEL", "qwen-2vl-default")
            seed = int(os.environ.get("QA_QWEN2VL_SEED")) if os.environ.get("QA_QWEN2VL_SEED") else None
            app.state.agent = adk_factory.get_agent("qa_qwen2vl", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="qa_qwen2vl")
            except Exception:
                LOG.debug("record_seed failed", exc_info=True)
            _safe_emit("agent.created", {"tool": "qa_qwen2vl", "model_spec": model_spec, "seed": seed})
        except Exception:
            LOG.exception("failed to construct ADK agent for qa_qwen2vl")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("qa_qwen2vl ready (agent attached)")
        _safe_send_telemetry("tool.ready", {"tool": "qa_qwen2vl"})
        _safe_emit("tool.ready", {"tool": "qa_qwen2vl"})
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="qa_qwen2vl Entrypoint", lifespan=lifespan)
    app.state.lock = Lock()
    app.state.ready = False
    app.state.shutting_down = False
    app.state.inflight = 0

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        LOG.debug("validation error", exc_info=exc)
        return JSONResponse(status_code=400, content={"detail": _sanitize_validation_errors(exc.errors())})

    @app.get("/health")
    def health() -> dict[str, str]:
        if getattr(app.state, "shutting_down", False):
            return {"status": "shutting_down"}
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"ready": bool(getattr(app.state, "ready", False)), "shutting_down": bool(getattr(app.state, "shutting_down", False))}

    @app.post("/invoke")
    def invoke(req: QaQwen2VlRequest) -> dict[str, Any]:
        _ensure_ready(app)
        request_id = uuid.uuid4().hex
        LOG.info("invoke.received", extra={"request_id": request_id})
        _safe_send_telemetry("invoke.received", {"tool": "qa_qwen2vl", "request_id": request_id})
        _safe_emit("invoke.received", {"tool": "qa_qwen2vl", "request_id": request_id})
        with app.state.lock:
            app.state.inflight += 1
        try:
            options_dict = _options_to_dict(req.options)
            download_limits = _download_limits(options_dict)
            frames, prompts, frame_ids = _materialize_frames(req, download_limits=download_limits)
            adapter_opts = _build_adapter_opts(req, options_dict)
            adapter_opts["frame_ids"] = frame_ids
            try:
                result = adapter.inspect_frames(frames, prompts, opts=adapter_opts)
            except adapter.QAWiringError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            resp = QaQwen2VlResponse(
                status="success",
                request_id=request_id,
                decision=result.decision,
                artifact_uri=result.artifact_uri,
                metadata=result.metadata,
                report=result.report.model_dump(mode="json"),
                human_task_id=result.human_task_id,
            )
            _safe_emit(
                "invoke.completed",
                {"tool": "qa_qwen2vl", "request_id": request_id, "artifact_uri": result.artifact_uri, "decision": result.decision},
            )
            return resp.model_dump()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


def _ensure_ready(app: FastAPI) -> None:
    if getattr(app.state, "shutting_down", False):
        raise HTTPException(status_code=503, detail="shutting down")
    if getattr(app.state, "ready", False):
        return
    delay = float(os.environ.get("MODEL_LOAD_DELAY", "0") or 0.0)
    if delay == 0.0:
        app.state.ready = True
        return
    raise HTTPException(status_code=503, detail="tool not ready")


def _materialize_frames(req: QaQwen2VlRequest, *, download_limits: Tuple[int, float]) -> tuple[list[bytes], list[str], list[str]]:
    max_bytes, timeout_s = download_limits
    frames: list[bytes] = []
    prompts: list[str] = []
    frame_ids: list[str] = []
    default_prompt = (req.prompt or "").strip()
    for idx, frame in enumerate(req.frames):
        prompt = (frame.prompt or default_prompt).strip()
        if not prompt:
            raise HTTPException(status_code=400, detail=f"frame[{idx}] missing prompt")
        frame_id = (frame.id or f"frame_{idx:04d}").strip() or f"frame_{idx:04d}"
        frames.append(_decode_frame(frame, idx, max_bytes=max_bytes, timeout_s=timeout_s))
        prompts.append(prompt)
        frame_ids.append(frame_id)
    return frames, prompts, frame_ids


def _download_limits(options: Mapping[str, Any]) -> Tuple[int, float]:
    max_bytes = int(options.get("max_download_bytes") or os.environ.get("QA_QWEN2VL_MAX_DOWNLOAD_BYTES") or DEFAULT_MAX_DOWNLOAD_BYTES)
    timeout_s = float(options.get("download_timeout_s") or os.environ.get("QA_QWEN2VL_DOWNLOAD_TIMEOUT_S") or DEFAULT_DOWNLOAD_TIMEOUT_S)
    max_bytes = max(1024, max_bytes)
    timeout_s = max(1.0, timeout_s)
    return max_bytes, timeout_s


def _decode_frame(frame: QaFramePayload, idx: int, *, max_bytes: int, timeout_s: float) -> bytes:
    if frame.data_base64:
        try:
            data = base64.b64decode(frame.data_base64, validate=True)
        except binascii.Error as exc:
            raise HTTPException(status_code=400, detail=f"frame[{idx}] data_base64 invalid") from exc
        if len(data) > max_bytes:
            raise HTTPException(status_code=400, detail=f"frame[{idx}] exceeds max_download_bytes ({len(data)} > {max_bytes})")
        return data
    if frame.uri:
        return _read_frame_uri(frame.uri, idx, max_bytes=max_bytes, timeout_s=timeout_s)
    raise HTTPException(status_code=400, detail=f"frame[{idx}] missing data")


def _read_frame_uri(uri: str, idx: int, *, max_bytes: int, timeout_s: float) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme else uri).expanduser()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"frame[{idx}] not found: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise HTTPException(status_code=400, detail=f"frame[{idx}] exceeds max_download_bytes ({size} > {max_bytes})")
        return path.read_bytes()
    if parsed.scheme in ("http", "https"):
        return _fetch_remote_bytes(uri, idx, max_bytes=max_bytes, timeout_s=timeout_s)
    raise HTTPException(status_code=400, detail=f"frame[{idx}] unsupported uri scheme {parsed.scheme}")


def _fetch_remote_bytes(uri: str, idx: int, *, max_bytes: int, timeout_s: float) -> bytes:
    request = urllib.request.Request(uri, headers={"User-Agent": "sparkle-motion-qa/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # nosec - controlled destination
            length = getattr(response, "length", None)
            if length and length > max_bytes:
                raise HTTPException(status_code=400, detail=f"frame[{idx}] exceeds max_download_bytes ({length} > {max_bytes})")
            chunks: list[bytes] = []
            total = 0
            while True:
                budget = max(1, min(65536, max_bytes - total))
                chunk = response.read(budget)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=400, detail=f"frame[{idx}] exceeds max_download_bytes ({total} > {max_bytes})")
                chunks.append(chunk)
            return b"".join(chunks)
    except HTTPException:
        raise
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=400, detail=f"frame[{idx}] download failed: {exc.reason}") from exc
    except Exception as exc:  # pragma: no cover - unexpected network issues
        raise HTTPException(status_code=400, detail=f"frame[{idx}] download failed: {exc}") from exc


def _build_adapter_opts(req: QaQwen2VlRequest, options_dict: Dict[str, Any]) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "plan_id": req.plan_id,
        "run_id": req.run_id,
        "step_id": req.step_id,
        "movie_title": req.movie_title,
        "metadata": dict(req.metadata or {}),
    }
    options_copy = dict(options_dict)
    options_metadata = options_copy.pop("metadata", None)
    if isinstance(options_metadata, dict):
        opts.setdefault("metadata", {}).update(options_metadata)
    for key, value in options_copy.items():
        if key in {"max_download_bytes", "download_timeout_s"}:
            continue
        opts[key] = value
    return opts


app = make_app()
