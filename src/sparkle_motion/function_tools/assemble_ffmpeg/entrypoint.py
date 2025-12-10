"""FunctionTool entrypoint that stitches rendered clips with FFmpeg.

Purpose & usage:
- Called by the production workflow when individual shot clips (and an optional
    master audio bed) must be concatenated, transitioned, and re-encoded into a
    single deliverable.
- Use this tool whenever the script/production agents hand back a manifest of
    clip URIs that should become the final movie or preview reel.

Request payload (`AssembleRequest` → POST /invoke JSON):
- `plan_id`, `run_id`, `step_id` (str, optional): tracing identifiers copied
    into telemetry and published artifacts.
- `seed` (int, optional): fixture determinism knob for retries/tests.
- `clips` (list[AssembleClip], required): each entry sets `uri`, optional
    `start_s`, `end_s`, `metadata`, and `transition` instructions for ffmpeg.
- `audio` (AssembleAudio, optional): timed audio bed (`uri`, timing bounds,
    metadata, gain) mixed across the rendered video.
- `options` (AssembleOptions, optional): codec overrides (`video_codec`,
    `audio_codec`, `pix_fmt`, `crf`, `preset`, `audio_bitrate`, `timeout_s`,
    `retries`, `fixture_only`).
- `metadata` (dict, optional): arbitrary context persisted in the response.

Response dictionary (`AssembleResponse`):
- `status`: "success" or "error" to signal ffmpeg outcome.
- `artifact_uri`: file/GS URI for the muxed video when successful.
- `request_id`: server-generated correlation identifier.
- `metadata`: adapter-supplied diagnostics (frame count, timings, retries, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping
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
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion import adk_factory, adk_helpers, observability, telemetry
from sparkle_motion.utils.env import fixture_mode_enabled
from sparkle_motion.function_tools.assemble_ffmpeg import adapter
from sparkle_motion.function_tools.assemble_ffmpeg.models import (
    AssembleAudio,
    AssembleClip,
    AssembleOptions,
    AssembleRequest,
    AssembleResponse,
)

LOG = logging.getLogger("assemble_ffmpeg.entrypoint")
LOG.setLevel(logging.INFO)


ClipModel = AssembleClip
AudioModel = AssembleAudio
OptionsModel = AssembleOptions
RequestModel = AssembleRequest
ResponseModel = AssembleResponse


def _serialize_validation_errors(errors: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    cleaned: list[Dict[str, Any]] = []
    for err in errors:
        data = dict(err)
        ctx = data.get("ctx")
        if isinstance(ctx, dict):
            data["ctx"] = {k: (str(v) if isinstance(v, BaseException) else v) for k, v in ctx.items()}
        cleaned.append(data)
    return cleaned


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
        # Eagerly construct per-tool ADK agent
        try:
            model_spec = os.environ.get("ASSEMBLE_FFMPEG_MODEL", "ffmpeg-assembler")
            seed = int(os.environ.get("ASSEMBLE_FFMPEG_SEED")) if os.environ.get("ASSEMBLE_FFMPEG_SEED") else None
            app.state.agent = adk_factory.get_agent("assemble_ffmpeg", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="assemble_ffmpeg")
                telemetry.emit_event("agent.created", {"tool": "assemble_ffmpeg", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception:
            LOG.exception("failed to construct ADK agent for assemble_ffmpeg")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("assemble_ffmpeg ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "assemble_ffmpeg"})
        except Exception:
            pass
        try:
            telemetry.emit_event("tool.ready", {"tool": "assemble_ffmpeg"})
        except Exception:
            pass
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="assemble_ffmpeg Entrypoint (scaffold)", lifespan=lifespan)
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
            send_telemetry("invoke.received", {"tool": "assemble_ffmpeg", "request_id": request_id})
        except Exception:
            pass
        try:
            telemetry.emit_event("invoke.received", {"tool": "assemble_ffmpeg", "request_id": request_id})
        except Exception:
            pass
        with app.state.lock:
            app.state.inflight += 1
        try:
            result = _invoke_adapter(req)
            metadata = _build_metadata(req, result, request_id)
            artifact_uri = _publish_artifact(result.path, metadata)

            try:
                send_telemetry(
                    "invoke.completed",
                    {"tool": "assemble_ffmpeg", "request_id": request_id, "artifact_uri": artifact_uri, "engine": metadata.get("engine")},
                )
            except Exception:
                pass
            try:
                telemetry.emit_event(
                    "invoke.completed",
                    {"tool": "assemble_ffmpeg", "request_id": request_id, "artifact_uri": artifact_uri, "engine": metadata.get("engine")},
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


def _invoke_adapter(req: RequestModel) -> adapter.AssemblyResult:
    options = _options_dict(req.options)
    clips = [_clip_spec_from_model(clip) for clip in req.clips]
    audio = _audio_spec_from_model(req.audio) if req.audio else None
    try:
        return adapter.assemble_movie(
            clips=clips,
            audio=audio,
            plan_id=req.plan_id,
            options=options,
            seed=req.seed,
            output_dir=_artifacts_dir(),
        )
    except adapter.AssemblyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except adapter.CommandTimeoutError as exc:
        raise HTTPException(status_code=503, detail="ffmpeg timed out") from exc
    except adapter.CommandError as exc:
        raise HTTPException(status_code=500, detail="ffmpeg command failed") from exc


def _clip_spec_from_model(model: ClipModel) -> adapter.ClipSpec:
    path = Path(model.uri).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"clip not found: {path}")
    return adapter.ClipSpec(
        uri=path,
        start_s=model.start_s,
        end_s=model.end_s,
        metadata=model.metadata,
        transition=model.transition,
    )


def _audio_spec_from_model(model: AudioModel | None) -> adapter.AudioSpec | None:
    if model is None:
        return None
    path = Path(model.uri).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"audio not found: {path}")
    return adapter.AudioSpec(
        uri=path,
        start_s=model.start_s,
        end_s=model.end_s,
        metadata=model.metadata,
        gain_db=model.gain_db,
    )


def _options_dict(options: OptionsModel | None) -> Dict[str, Any]:
    return options.model_dump(exclude_none=True) if options else {}


def _artifacts_dir() -> Path:
    base = Path(os.environ.get("ARTIFACTS_DIR", Path.cwd() / "artifacts"))
    target = base / "assemble_ffmpeg"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _build_metadata(req: RequestModel, result: adapter.AssemblyResult, request_id: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = dict(result.metadata)
    metadata.setdefault("engine", result.engine)
    metadata["request_id"] = request_id
    metadata["local_path"] = str(result.path)
    metadata["duration_s"] = result.duration_s
    if req.plan_id:
        metadata.setdefault("plan_id", req.plan_id)
    if req.step_id:
        metadata["step_id"] = req.step_id
    if req.run_id:
        metadata["run_id"] = req.run_id
    if req.metadata:
        user_meta = dict(req.metadata)
        metadata.setdefault("user_metadata", user_meta)
    return metadata


def _publish_artifact(path: Path, metadata: Mapping[str, Any]) -> str:
    try:
        artifact = adk_helpers.publish_artifact(
            local_path=path,
            artifact_type="video_final",
            media_type="video/mp4",
            metadata=metadata,
        )
        uri = artifact.get("uri")  # type: ignore[index]
    except Exception:
        uri = None
    if not uri:
        uri = f"file://{path}"
    if fixture_mode_enabled() and isinstance(uri, str) and adk_helpers.is_adk_artifact_uri(uri):
        return f"file://{path}"
    return str(uri)
