from __future__ import annotations

"""FunctionTool entrypoint for WAV2Lip-driven lipsync renders.

Purpose & usage:
- Applies a spoken audio track to a hero frame or clip to produce a synced
    talking-head shot. Use whenever the production agent needs mouth movement for
    a shot flagged as `is_talking_closeup` or when QA demands replacement audio.

Request payload (`LipsyncWav2LipRequest`):
- `face` / `audio` (`LipsyncMediaPayload`): each accepts `uri`, filesystem
    `path`, or inline `data_b64` bytes describing the visual reference and voice.
- `plan_id`, `run_id`, `step_id`, `movie_title`, `metadata`, `out_basename`
    capture provenance for artifact publishing.
- `options` (`LipsyncWav2LipOptions`): fine-grained wav2lip knobs such as
    checkpoints, detector pads, resize factor, FPS, timeout, retry budget,
    repository/script paths, fixture flags, and fixture seeds.

Response dictionary (`LipsyncWav2LipResponse`):
- `status`: "success" or "error".
- `artifact_uri`: URI for the rendered mp4/wav pair.
- `request_id`: unique call identifier.
- `metadata`: structured adapter data (timings, fps, inference source, etc.).
- `logs`: stdout/stderr summary from the wav2lip subprocess for debugging.
"""

import asyncio
import base64
import binascii
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sparkle_motion import adk_factory, adk_helpers, observability, telemetry
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion.function_tools.lipsync_wav2lip import adapter
from sparkle_motion.function_tools.lipsync_wav2lip.adapter import LipsyncError
from sparkle_motion.function_tools.lipsync_wav2lip.models import (
    LipsyncMediaPayload,
    LipsyncWav2LipOptions,
    LipsyncWav2LipRequest,
    LipsyncWav2LipResponse,
)
from sparkle_motion.utils.env import fixture_mode_enabled

LOG = logging.getLogger("lipsync_wav2lip.entrypoint")
LOG.setLevel(logging.INFO)

# Keep legacy attribute names so shared entrypoint tests continue to reflect import contracts.
RequestModel = LipsyncWav2LipRequest
ResponseModel = LipsyncWav2LipResponse


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
            model_spec = os.environ.get("LIPSYNC_WAV2LIP_MODEL", "wav2lip-default")
            seed = int(os.environ.get("LIPSYNC_WAV2LIP_SEED")) if os.environ.get("LIPSYNC_WAV2LIP_SEED") else None
            app.state.agent = adk_factory.get_agent("lipsync_wav2lip", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="lipsync_wav2lip")
                telemetry.emit_event("agent.created", {"tool": "lipsync_wav2lip", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception:
            LOG.exception("failed to construct ADK agent for lipsync_wav2lip")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("lipsync_wav2lip ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "lipsync_wav2lip"})
        except Exception:
            pass
        try:
            telemetry.emit_event("tool.ready", {"tool": "lipsync_wav2lip"})
        except Exception:
            pass
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="lipsync_wav2lip Entrypoint (scaffold)", lifespan=lifespan)
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
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    @app.get("/health")
    def health() -> dict[str, str]:
        if getattr(app.state, "shutting_down", False):
            return {"status": "shutting_down"}
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"ready": bool(getattr(app.state, "ready", False)), "shutting_down": bool(getattr(app.state, "shutting_down", False))}

    @app.post("/invoke")
    def invoke(req: LipsyncWav2LipRequest) -> dict[str, Any]:
        _ensure_ready(app)
        request_id = uuid.uuid4().hex
        LOG.info("invoke.received", extra={"request_id": request_id})
        _emit_event("invoke.received", request_id)
        with app.state.lock:
            app.state.inflight += 1
        try:
            response = _handle_invoke(req, request_id=request_id)
            _emit_event("invoke.completed", request_id, extra={"artifact_uri": response["artifact_uri"]})
            return response
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()


def _handle_invoke(req: LipsyncWav2LipRequest, *, request_id: str) -> dict[str, Any]:
    adapter_opts = _build_adapter_opts(req)
    target_path = _target_path(req, request_id=request_id)
    with tempfile.TemporaryDirectory(prefix="lipsync-") as tmp_dir:
        scratch = Path(tmp_dir)
        face_path = _materialize_media(req.face, scratch=scratch, label="face")
        audio_path = _materialize_media(req.audio, scratch=scratch, label="audio")
        try:
            result = adapter.run_wav2lip(face_path, audio_path, target_path, opts=adapter_opts)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LipsyncError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    metadata = _build_metadata(req, result.metadata, request_id=request_id)
    artifact_ref = _publish_artifact(result.path, metadata=metadata, run_id=req.run_id)
    artifact_uri = artifact_ref["uri"]
    if fixture_mode_enabled() and artifact_uri.startswith("artifact://"):
        artifact_uri = f"file://{result.path.resolve()}"

    response = LipsyncWav2LipResponse(
        status="success",
        artifact_uri=artifact_uri,
        request_id=request_id,
        metadata=metadata,
        logs=dict(result.logs),
    )
    return response.model_dump()


def _ensure_ready(app: FastAPI) -> None:
    if getattr(app.state, "shutting_down", False):
        raise HTTPException(status_code=503, detail="shutting down")
    ready = getattr(app.state, "ready", False)
    if ready:
        return
    if fixture_mode_enabled() or os.environ.get("DETERMINISTIC", "0") == "1":
        app.state.ready = True
        app.state.shutting_down = False
        return
    try:
        delay = float(os.environ.get("MODEL_LOAD_DELAY", "0"))
    except (TypeError, ValueError):
        delay = 0.0
    if delay == 0.0:
        app.state.ready = True
        return
    raise HTTPException(status_code=503, detail="tool not ready")


def _emit_event(name: str, request_id: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"tool": "lipsync_wav2lip", "request_id": request_id}
    if extra:
        payload.update(extra)
    try:
        send_telemetry(name, payload)
    except Exception:
        pass
    try:
        telemetry.emit_event(name, payload)
    except Exception:
        pass


def _materialize_media(payload: LipsyncMediaPayload, *, scratch: Path, label: str) -> Path:
    if payload.data_base64:
        data = _decode_b64(payload.data_base64, label)
        dest = scratch / f"{label}{_guess_extension(payload)}"
        dest.write_bytes(data)
        return dest
    source = payload.uri or payload.path
    assert source is not None
    parsed = urlparse(source)
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme else source).expanduser()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"{label} not found: {path}")
        return path
    raise HTTPException(status_code=400, detail=f"{label} unsupported uri scheme {parsed.scheme}")


def _guess_extension(payload: LipsyncMediaPayload) -> str:
    if payload.uri:
        return Path(urlparse(payload.uri).path).suffix or ".bin"
    if payload.path:
        return Path(payload.path).suffix or ".bin"
    return ".bin"


def _decode_b64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{label} data_b64 invalid") from exc


def _target_path(req: LipsyncWav2LipRequest, *, request_id: str) -> Path:
    base = Path(os.environ.get("ARTIFACTS_DIR", Path.cwd() / "artifacts"))
    dest = base / "lipsync_wav2lip"
    dest.mkdir(parents=True, exist_ok=True)
    slug_source = req.out_basename or req.step_id or req.plan_id or "lipsync"
    slug = _slugify(slug_source or "lipsync")
    return (dest / f"{slug}-{request_id[:8]}.mp4").resolve()


def _slugify(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", cleaned)
    return cleaned[:80] or "artifact"


def _build_adapter_opts(req: LipsyncWav2LipRequest) -> Dict[str, Any]:
    if req.options is None:
        return {}
    opts = req.options.model_dump(exclude_none=True)
    return opts


def _build_metadata(req: LipsyncWav2LipRequest, adapter_meta: Dict[str, Any], *, request_id: str) -> Dict[str, Any]:
    metadata = dict(req.metadata or {})
    metadata.update(adapter_meta)
    metadata.setdefault("plan_id", req.plan_id)
    metadata.setdefault("run_id", req.run_id)
    metadata.setdefault("step_id", req.step_id)
    metadata.setdefault("movie_title", req.movie_title)
    metadata.setdefault("request_id", request_id)
    metadata.setdefault("tool", "lipsync_wav2lip")
    return metadata


def _publish_artifact(path: Path, *, metadata: Dict[str, Any], run_id: Optional[str]) -> adk_helpers.ArtifactRef:
    try:
        return adk_helpers.publish_artifact(
            local_path=path,
            artifact_type="lipsync_clip",
            media_type="video/mp4",
            metadata=metadata,
            run_id=run_id,
        )
    except adk_helpers.ArtifactPublishError as exc:
        LOG.error("artifact publish failed", extra={"path": str(path), "error": str(exc)})
        raise HTTPException(status_code=502, detail="artifact publish failed") from exc
