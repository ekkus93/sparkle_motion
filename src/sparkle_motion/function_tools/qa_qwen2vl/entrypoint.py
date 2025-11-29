from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from sparkle_motion import adk_factory, observability, telemetry
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion.function_tools.qa_qwen2vl import adapter

LOG = logging.getLogger("qa_qwen2vl.entrypoint")
LOG.setLevel(logging.INFO)


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


class FramePayload(BaseModel):
    id: str | None = None
    uri: str | None = None
    data_base64: str | None = Field(default=None, alias="data_b64")
    prompt: str | None = None

    @model_validator(mode="after")
    def _ensure_source(self) -> "FramePayload":
        if not self.uri and not self.data_base64:
            raise ValueError("each frame must provide uri or data_base64")
        return self


class OptionsModel(BaseModel):
    fixture_only: bool | None = None
    max_new_tokens: int | None = Field(default=None, ge=16, le=1024)
    policy_path: str | None = None
    fixture_seed: int | None = None
    model_id: str | None = None
    metadata: Dict[str, Any] | None = None


class RequestModel(BaseModel):
    frames: List[FramePayload]
    prompt: str | None = None
    plan_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    movie_title: str | None = None
    metadata: Dict[str, Any] | None = None
    options: OptionsModel | None = None

    @model_validator(mode="after")
    def _validate_frames(self) -> "RequestModel":
        if not self.frames:
            raise ValueError("frames must be provided")
        has_prompt = bool((self.prompt or "").strip()) or any((frame.prompt or "").strip() for frame in self.frames)
        if not has_prompt:
            raise ValueError("provide prompt either globally or per frame")
        return self


class ResponseModel(BaseModel):
    status: Literal["success"]
    request_id: str
    decision: Literal["approve", "regenerate", "escalate", "pending"]
    artifact_uri: str
    metadata: Dict[str, Any]
    report: Dict[str, Any]
    human_task_id: str | None = None


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
    def invoke(req: RequestModel) -> dict[str, Any]:
        _ensure_ready(app)
        request_id = uuid.uuid4().hex
        LOG.info("invoke.received", extra={"request_id": request_id})
        _safe_send_telemetry("invoke.received", {"tool": "qa_qwen2vl", "request_id": request_id})
        _safe_emit("invoke.received", {"tool": "qa_qwen2vl", "request_id": request_id})
        with app.state.lock:
            app.state.inflight += 1
        try:
            frames, prompts = _materialize_frames(req)
            adapter_opts = _build_adapter_opts(req)
            try:
                result = adapter.inspect_frames(frames, prompts, opts=adapter_opts)
            except adapter.QAWiringError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            resp = ResponseModel(
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


def _materialize_frames(req: RequestModel) -> tuple[list[bytes], list[str]]:
    frames: list[bytes] = []
    prompts: list[str] = []
    default_prompt = (req.prompt or "").strip()
    for idx, frame in enumerate(req.frames):
        prompt = (frame.prompt or default_prompt).strip()
        if not prompt:
            raise HTTPException(status_code=400, detail=f"frame[{idx}] missing prompt")
        frames.append(_decode_frame(frame, idx))
        prompts.append(prompt)
    return frames, prompts


def _decode_frame(frame: FramePayload, idx: int) -> bytes:
    if frame.data_base64:
        try:
            return base64.b64decode(frame.data_base64, validate=True)
        except binascii.Error as exc:
            raise HTTPException(status_code=400, detail=f"frame[{idx}] data_base64 invalid") from exc
    if frame.uri:
        return _read_frame_uri(frame.uri, idx)
    raise HTTPException(status_code=400, detail=f"frame[{idx}] missing data")


def _read_frame_uri(uri: str, idx: int) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme else uri).expanduser()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"frame[{idx}] not found: {path}")
        return path.read_bytes()
    raise HTTPException(status_code=400, detail=f"frame[{idx}] unsupported uri scheme {parsed.scheme}")


def _build_adapter_opts(req: RequestModel) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "plan_id": req.plan_id,
        "run_id": req.run_id,
        "step_id": req.step_id,
        "movie_title": req.movie_title,
        "metadata": dict(req.metadata or {}),
    }
    if req.options:
        if req.options.fixture_only is not None:
            opts["fixture_only"] = req.options.fixture_only
        if req.options.max_new_tokens is not None:
            opts["max_new_tokens"] = req.options.max_new_tokens
        if req.options.policy_path:
            opts["policy_path"] = req.options.policy_path
        if req.options.fixture_seed is not None:
            opts["fixture_seed"] = req.options.fixture_seed
        if req.options.model_id:
            opts["model_id"] = req.options.model_id
        if req.options.metadata:
            opts.setdefault("metadata", {}).update(req.options.metadata)
    return opts


app = make_app()
