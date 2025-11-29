from __future__ import annotations
from typing import Any, Dict, Literal, Mapping
from pathlib import Path
import os
import logging
import uuid
import time
import asyncio
from threading import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion import adk_factory, adk_helpers, observability, telemetry
from sparkle_motion.function_tools.videos_wan import adapter

LOG = logging.getLogger("videos_wan.entrypoint")
LOG.setLevel(logging.INFO)


class OptionsModel(BaseModel):
    num_inference_steps: int | None = Field(default=None, ge=1, le=128)
    guidance_scale: float | None = Field(default=None, ge=0.0)
    negative_prompt: str | None = None
    motion_bucket_id: int | None = Field(default=None, ge=0)
    megapixels: float | None = Field(default=None, ge=0.0)
    fixture_only: bool | None = None


class RequestModel(BaseModel):
    prompt: str
    plan_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    seed: int | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=1)
    num_frames: int = Field(default=64, ge=1, le=2048)
    fps: int = Field(default=24, ge=1, le=120)
    width: int = Field(default=1280, ge=64, le=4096)
    height: int = Field(default=720, ge=64, le=4096)
    metadata: Dict[str, Any] | None = None
    options: OptionsModel | None = None
    start_frame_uri: str | None = None
    end_frame_uri: str | None = None

    @model_validator(mode="after")
    def _validate_prompt(self) -> "RequestModel":
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt is required")
        return self


class ResponseModel(BaseModel):
    status: Literal["success", "error"]
    artifact_uri: str | None
    request_id: str
    metadata: Dict[str, Any] | None = None


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

        # Eagerly construct the per-tool ADK agent. This follows the project's
        # fail‑loud semantics: if the SDK or credentials are missing, startup
        # should fail rather than silently falling back. In fixture mode the
        # factory returns a lightweight dummy agent.
        try:
            model_spec = os.environ.get("VIDEOS_WAN_MODEL", "Wan-AI/Wan2.1-I2V-14B-720P")
            # attach a seed if provided via env for deterministic tests
            seed = int(os.environ.get("VIDEOS_WAN_SEED")) if os.environ.get("VIDEOS_WAN_SEED") else None
            app.state.agent = adk_factory.get_agent("videos_wan", model_spec=model_spec, mode="per-tool", seed=seed)
            # record seed & emit agent lifecycle event
            try:
                observability.record_seed(seed, tool_name="videos_wan")
                telemetry.emit_event("agent.created", {"tool": "videos_wan", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception as e:
            LOG.exception("failed to construct ADK agent for videos_wan: %s", e)
            # re-raise to prevent the app from starting silently
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("videos_wan ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "videos_wan"})
        except Exception:
            pass
        try:
            telemetry.emit_event("tool.ready", {"tool": "videos_wan"})
        except Exception:
            pass
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="videos_wan Entrypoint (scaffold)", lifespan=lifespan)
    app.state.lock = Lock()
    app.state.ready = False
    app.state.shutting_down = False
    app.state.inflight = 0

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        try:
            LOG.debug("validation error", exc_info=exc)
        except Exception:
            pass
        return JSONResponse(status_code=400, content={"detail": _serialize_validation_errors(exc.errors())})

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
        if not getattr(app.state, "ready", False):
            try:
                delay = float(os.environ.get("MODEL_LOAD_DELAY", "0"))
            except Exception:
                delay = 0.0
            if delay == 0.0:
                app.state.ready = True
            else:
                raise HTTPException(status_code=503, detail="tool not ready")
        if getattr(app.state, "shutting_down", False):
            raise HTTPException(status_code=503, detail="shutting down")

        request_id = uuid.uuid4().hex
        LOG.info("invoke.received", extra={"request_id": request_id})
        try:
            send_telemetry("invoke.received", {"tool": "videos_wan", "request_id": request_id})
        except Exception:
            pass
        try:
            telemetry.emit_event("invoke.received", {"tool": "videos_wan", "request_id": request_id})
        except Exception:
            pass
        with app.state.lock:
            app.state.inflight += 1
        try:
            try:
                result = _invoke_adapter(req)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except adapter.VideoRenderError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            metadata = _build_metadata(req, result, request_id)
            artifact_uri = _publish_artifact(result.path, metadata)

            try:
                send_telemetry(
                    "invoke.completed",
                    {"tool": "videos_wan", "request_id": request_id, "artifact_uri": artifact_uri, "engine": metadata.get("engine")},
                )
            except Exception:
                pass
            try:
                telemetry.emit_event(
                    "invoke.completed",
                    {"tool": "videos_wan", "request_id": request_id, "artifact_uri": artifact_uri, "engine": metadata.get("engine")},
                )
            except Exception:
                pass

            resp = ResponseModel(status="success", artifact_uri=artifact_uri, request_id=request_id, metadata=metadata)
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()


def _invoke_adapter(req: RequestModel) -> adapter.VideoRenderResult:
    options = req.options.model_dump(exclude_none=True) if req.options else {}
    metadata = dict(req.metadata or {})
    start_frame = _load_frame(req.start_frame_uri)
    end_frame = _load_frame(req.end_frame_uri)
    return adapter.render_clip(
        prompt=req.prompt,
        num_frames=req.num_frames,
        fps=req.fps,
        width=req.width,
        height=req.height,
        seed=req.seed,
        plan_id=req.plan_id,
        chunk_index=req.chunk_index,
        chunk_count=req.chunk_count,
        metadata=metadata,
        options=options,
        start_frame=start_frame,
        end_frame=end_frame,
        output_dir=_artifacts_dir(),
    )


def _build_metadata(req: RequestModel, result: adapter.VideoRenderResult, request_id: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = dict(result.metadata)
    metadata.setdefault("engine", result.engine)
    metadata["duration_s"] = result.duration_s
    metadata["frame_count"] = result.frame_count
    metadata["request_id"] = request_id
    metadata["local_path"] = str(result.path)
    if req.plan_id:
        metadata.setdefault("plan_id", req.plan_id)
    if req.step_id:
        metadata["step_id"] = req.step_id
    if req.run_id:
        metadata["run_id"] = req.run_id
    return metadata


def _publish_artifact(path: Path, metadata: Mapping[str, Any]) -> str:
    try:
        artifact = adk_helpers.publish_artifact(
            local_path=path,
            artifact_type="videos_wan_clip",
            media_type="video/mp4",
            metadata=metadata,
        )
        uri = artifact.get("uri")  # type: ignore[index]
    except Exception:
        uri = None
    if not uri:
        uri = f"file://{path}"
    if os.environ.get("ADK_USE_FIXTURE", "0") == "1" and isinstance(uri, str) and uri.startswith("artifact://"):
        return f"file://{path}"
    return str(uri)


def _load_frame(uri: str | None) -> bytes | None:
    if not uri:
        return None
    path = Path(uri).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"frame not found: {path}")
    return path.read_bytes()


def _artifacts_dir() -> Path:
    base = Path(os.environ.get("ARTIFACTS_DIR", Path.cwd() / "artifacts"))
    target = base / "videos_wan"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _serialize_validation_errors(errors: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    cleaned: list[Dict[str, Any]] = []
    for err in errors:
        data = dict(err)
        ctx = data.get("ctx")
        if isinstance(ctx, dict):
            data["ctx"] = {k: (str(v) if isinstance(v, BaseException) else v) for k, v in ctx.items()}
        cleaned.append(data)
    return cleaned
