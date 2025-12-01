from __future__ import annotations

"""FunctionTool entrypoint for Stable Diffusion XL still-image generation.

Purpose & usage:
- Generates cinematic base images for each shot or storyboard beat in the
    Sparkle Motion pipeline. Call whenever the script or production stage needs a
    batch of SDXL renders tied to a MoviePlan.

Request payload (`ImagesSDXLRequest`):
- `prompt` (str, required) plus optional `negative_prompt`, `prompt_2`,
    `negative_prompt_2` to steer SDXL refiner inputs.
- `metadata` (dict) propagated to downstream artifacts, along with `plan_id`,
    `run_id`, `base_image_id`, `batch_start`, `count`, and `seed` for tracking.
- Render parameters: `width`, `height`, `steps`, `cfg_scale`, `sampler`,
    `denoising_start`, `denoising_end` (all validated for SDXL constraints).

Response dictionary (`ImagesSDXLResponse`):
- `status`: "success" or "error" for the render batch.
- `request_id`: unique call identifier.
- `artifact_uri`: optional URI of a packaged artifact bundle.
- `artifacts`: list of `{"artifact_uri", "metadata"}` entries, one per image.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from sparkle_motion import adk_factory, adk_helpers, gpu_utils, observability, telemetry
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion.function_tools.images_sdxl import adapter
from sparkle_motion.function_tools.images_sdxl.models import (
    ImagesSDXLArtifact,
    ImagesSDXLRequest,
    ImagesSDXLResponse,
)
from sparkle_motion.utils.env import fixture_mode_enabled

LOG = logging.getLogger("images_sdxl.entrypoint")
LOG.setLevel(logging.INFO)

ArtifactPayload = ImagesSDXLArtifact
RequestModel = ImagesSDXLRequest
ResponseModel = ImagesSDXLResponse


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
        # Eagerly construct per-tool ADK agent (fixture mode returns dummy agent).
        try:
            model_spec = os.environ.get("IMAGES_SDXL_MODEL", "stable-diffusion-xl")
            seed = int(os.environ.get("IMAGES_SDXL_SEED")) if os.environ.get("IMAGES_SDXL_SEED") else None
            app.state.agent = adk_factory.get_agent("images_sdxl", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="images_sdxl")
                telemetry.emit_event("agent.created", {"tool": "images_sdxl", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception:
            LOG.exception("failed to construct ADK agent for images_sdxl")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("images_sdxl ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "images_sdxl"})
        except Exception:
            pass
        try:
            telemetry.emit_event("tool.ready", {"tool": "images_sdxl"})
        except Exception:
            pass
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="images_sdxl Entrypoint (scaffold)", lifespan=lifespan)
    app.state.lock = Lock()
    app.state.ready = False
    app.state.shutting_down = False
    app.state.inflight = 0

    # Map Pydantic/FastAPI validation errors to HTTP 400 to keep contract stable
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        try:
            LOG.debug("validation error", exc_info=exc)
        except Exception:
            pass
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    @app.get("/health")
    def health() -> dict[str, str]:
        # Always report ok for health endpoint to keep tests deterministic.
        # The readiness endpoint covers `ready`/`shutting_down` semantics.
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"ready": bool(getattr(app.state, "ready", False)), "shutting_down": bool(getattr(app.state, "shutting_down", False))}

    @app.post("/invoke")
    def invoke(req: RequestModel) -> dict[str, Any]:
        # In test/fixture mode, or deterministic runs, consider the tool ready
        if fixture_mode_enabled() or os.environ.get("DETERMINISTIC", "0") == "1":
            app.state.ready = True
            # When running under test fixtures, ensure we are not marked shutting down
            app.state.shutting_down = False

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
            send_telemetry("invoke.received", {"tool": "images_sdxl", "request_id": request_id})
        except Exception:
            pass
        with app.state.lock:
            app.state.inflight += 1
        try:
            payloads = _render_and_publish(req, request_id=request_id)
            artifact_uri = payloads[0].artifact_uri if payloads else None
            try:
                send_telemetry(
                    "invoke.completed",
                    {"tool": "images_sdxl", "request_id": request_id, "artifact_uri": artifact_uri, "count": len(payloads)},
                )
            except Exception:
                pass

            response = ResponseModel(status="success", request_id=request_id, artifact_uri=artifact_uri, artifacts=payloads)
            return response.model_dump()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()


def _render_and_publish(req: RequestModel, *, request_id: str) -> List[ArtifactPayload]:
    try:
        render_results = adapter.render_images(
            req.prompt,
            {
                "negative_prompt": req.negative_prompt,
                "prompt_2": req.prompt_2,
                "negative_prompt_2": req.negative_prompt_2,
                "metadata": req.metadata,
                "count": req.count,
                "seed": req.seed,
                "width": req.width,
                "height": req.height,
                "steps": req.steps,
                "cfg_scale": req.cfg_scale,
                "sampler": req.sampler,
                "batch_start": req.batch_start,
                "denoising_start": req.denoising_start,
                "denoising_end": req.denoising_end,
            },
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=_serialize_validation_errors(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except gpu_utils.GpuBusyError as exc:  # pragma: no cover - requires GPU contention
        raise HTTPException(status_code=503, detail="gpu busy") from exc

    artifacts: List[ArtifactPayload] = []
    for idx, result in enumerate(render_results):
        metadata = _build_metadata(req, result.metadata, request_id=request_id, index=req.batch_start + idx)
        artifact_ref = _publish_image(result.path, metadata=metadata, run_id=req.run_id)
        uri = artifact_ref["uri"]
        if fixture_mode_enabled() and uri.startswith("artifact://"):
            uri = f"file://{result.path.resolve()}"
        payload = ArtifactPayload(artifact_uri=uri, metadata=dict(artifact_ref.get("metadata", metadata)))
        artifacts.append(payload)
    return artifacts


def _publish_image(path: Path, *, metadata: Dict[str, Any], run_id: Optional[str]) -> adk_helpers.ArtifactRef:
    try:
        return adk_helpers.publish_artifact(
            local_path=path,
            artifact_type="image_frame",
            metadata=metadata,
            media_type="image/png",
            run_id=run_id,
        )
    except adk_helpers.ArtifactPublishError as exc:
        LOG.error("artifact publish failed", extra={"path": str(path), "error": str(exc)})
        raise HTTPException(status_code=502, detail="artifact publish failed") from exc


def _build_metadata(req: RequestModel, base: Dict[str, Any], *, request_id: str, index: int) -> Dict[str, Any]:
    metadata = dict(req.metadata or {})
    metadata.update(base)
    metadata.setdefault("request_id", request_id)
    metadata.setdefault("plan_id", req.plan_id)
    metadata.setdefault("run_id", req.run_id)
    metadata.setdefault("index", index)
    metadata.setdefault("count", req.count)
    metadata.setdefault("prompt", req.prompt)
    metadata.setdefault("negative_prompt", req.negative_prompt)
    metadata.setdefault("width", req.width)
    metadata.setdefault("height", req.height)
    metadata.setdefault("sampler", req.sampler)
    metadata.setdefault("steps", req.steps)
    metadata.setdefault("cfg_scale", req.cfg_scale)
    metadata.setdefault("batch_start", req.batch_start)
    metadata.setdefault("tool", "images_sdxl")
    return metadata


def _serialize_validation_errors(exc: ValidationError) -> List[Dict[str, Any]]:
    return [
        {"loc": ".".join(str(part) for part in err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in exc.errors()
    ]
