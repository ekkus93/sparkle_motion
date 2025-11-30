from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Optional, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sparkle_motion import adk_factory, adk_helpers, gpu_utils, observability, telemetry
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion.function_tools.tts_chatterbox import adapter as chatterbox_adapter
from sparkle_motion.utils.env import fixture_mode_enabled
from sparkle_motion.function_tools.tts_chatterbox.models import TTSChatterboxRequest, TTSChatterboxResponse

LOG = logging.getLogger("tts_chatterbox.entrypoint")
LOG.setLevel(logging.INFO)

# Preserve historical attribute names so shared entrypoint tests keep passing.
RequestModel = TTSChatterboxRequest
ResponseModel = TTSChatterboxResponse


def _serialize_validation_errors(errors: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    cleaned: list[Mapping[str, Any]] = []
    for err in errors:
        data = dict(err)
        ctx = data.get("ctx")
        if isinstance(ctx, Mapping):
            data["ctx"] = {k: (str(v) if isinstance(v, BaseException) else v) for k, v in ctx.items()}
        cleaned.append(data)
    return cleaned


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
        # Eagerly construct per-tool ADK agent (fixture mode returns dummy agent)
        try:
            model_spec = os.environ.get("TTS_CHATTERBOX_MODEL", "tts-default")
            seed = int(os.environ.get("TTS_CHATTERBOX_SEED")) if os.environ.get("TTS_CHATTERBOX_SEED") else None
            app.state.agent = adk_factory.get_agent("tts_chatterbox", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="tts_chatterbox")
                telemetry.emit_event("agent.created", {"tool": "tts_chatterbox", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception:
            LOG.exception("failed to construct ADK agent for tts_chatterbox")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("tts_chatterbox ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "tts_chatterbox"})
        except Exception:
            pass
        try:
            telemetry.emit_event("tool.ready", {"tool": "tts_chatterbox"})
        except Exception:
            pass
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="tts_chatterbox Entrypoint (scaffold)", lifespan=lifespan)
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
    def invoke(req: TTSChatterboxRequest) -> dict[str, Any]:
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
            send_telemetry("invoke.received", {"tool": "tts_chatterbox", "request_id": request_id})
        except Exception:
            pass
        try:
            telemetry.emit_event("invoke.received", {"tool": "tts_chatterbox", "request_id": request_id})
        except Exception:
            pass
        with app.state.lock:
            app.state.inflight += 1
        try:
            result = _run_synthesis(req)
            metadata = _build_metadata(req, result, request_id)
            artifact_uri = _publish_artifact(result.path, metadata)

            try:
                send_telemetry(
                    "invoke.completed",
                    {"tool": "tts_chatterbox", "request_id": request_id, "artifact_uri": artifact_uri, "engine": metadata.get("engine")},
                )
            except Exception:
                pass
            try:
                telemetry.emit_event(
                    "invoke.completed",
                    {"tool": "tts_chatterbox", "request_id": request_id, "artifact_uri": artifact_uri, "engine": metadata.get("engine")},
                )
            except Exception:
                pass

            resp = TTSChatterboxResponse(status="success", artifact_uri=artifact_uri, request_id=request_id, metadata=metadata)
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()


def _watermarking_enabled() -> bool:
    return os.environ.get("CHATTERBOX_WATERMARKING", "1").strip().lower() in {"1", "true", "yes", "on"}


def _artifacts_dir() -> Path:
    base = Path(os.environ.get("ARTIFACTS_DIR", Path(os.getcwd()) / "artifacts"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _run_synthesis(req: TTSChatterboxRequest) -> chatterbox_adapter.SynthesisResult:
    text = (req.text or req.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' or 'prompt' is required")
    try:
        return chatterbox_adapter.synthesize_text(
            text=text,
            voice_id=req.voice_id,
            sample_rate=req.sample_rate,
            bit_depth=req.bit_depth,
            language=req.language,
            seed=req.seed,
            watermarking=_watermarking_enabled(),
            output_dir=_artifacts_dir(),
            metadata=req.metadata,
        )
    except gpu_utils.GpuBusyError as exc:
        LOG.warning("GPU busy for tts request", extra={"request_id": req.run_id})
        raise HTTPException(status_code=503, detail="gpu busy, retry later") from exc
    except Exception as exc:
        LOG.exception("tts synthesis failed", exc_info=exc)
        raise HTTPException(status_code=500, detail="synthesis failed") from exc


def _build_metadata(req: TTSChatterboxRequest, result: chatterbox_adapter.SynthesisResult, request_id: str) -> Dict[str, Any]:
    engine_meta = dict(result.metadata)
    metadata: Dict[str, Any] = {
        "request_id": request_id,
        "voice_id": req.voice_id,
        "language": req.language or "auto",
        "duration_s": round(result.duration_s, 3),
        "sample_rate": result.sample_rate,
        "bit_depth": result.bit_depth,
        "watermarked": result.watermarking,
        "engine": engine_meta.get("engine"),
        "engine_metadata": engine_meta,
        "local_path": str(result.path),
    }
    for field_name in ("plan_id", "step_id", "run_id"):
        value = getattr(req, field_name)
        if value:
            metadata[field_name] = value
    return metadata


def _publish_artifact(path: Path, metadata: Mapping[str, Any]) -> str:
    try:
        artifact = adk_helpers.publish_artifact(local_path=path, artifact_type="tts_audio", metadata=metadata)
    except Exception:
        artifact = {"uri": f"file://{path}", "metadata": metadata}
    uri = artifact.get("uri") or f"file://{path}"  # type: ignore[arg-type]
    if fixture_mode_enabled() and uri.startswith("artifact://"):
        return f"file://{path}"
    return uri
