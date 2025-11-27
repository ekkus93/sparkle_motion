from __future__ import annotations
from typing import Any, Literal
import re
from pathlib import Path
import os
import json
import logging
import uuid
import time
import asyncio
from threading import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion import adk_factory, gpu_utils

LOG = logging.getLogger("videos_wan.entrypoint")
LOG.setLevel(logging.INFO)


class RequestModel(BaseModel):
    # TODO: adjust fields for real schema
    prompt: str


class ResponseModel(BaseModel):
    status: Literal["success", "error"]
    artifact_uri: str | None
    request_id: str


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
            app.state.agent = adk_factory.get_agent("videos_wan", model_spec=model_spec, mode="per-tool")
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
        with app.state.lock:
            app.state.inflight += 1
        try:
            # minimal validation
            if not getattr(req, "prompt", None):
                raise HTTPException(status_code=400, detail="prompt required")

            deterministic = os.environ.get("DETERMINISTIC", "1") == "1"
            artifacts_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.getcwd(), "artifacts"))
            os.makedirs(artifacts_dir, exist_ok=True)

            def _slugify(s: str) -> str:
                s = s.strip()
                s = re.sub(r"\s+", "_", s)
                s = re.sub(r"[^A-Za-z0-9._-]", "", s)
                return s[:80] or "artifact"

            safe_name = _slugify(getattr(req, "prompt", "artifact"))
            filename = f"{safe_name}.json" if deterministic else f"{safe_name}_{os.getpid()}_{request_id}.json"
            local_path = os.path.join(artifacts_dir, filename)
            try:
                payload = req.model_dump_json() if hasattr(req, "model_dump_json") else req.json()
            except Exception:
                try:
                    payload = json.dumps(req.dict(), ensure_ascii=False)
                except Exception as e:
                    LOG.exception("failed to serialize request", extra={"request_id": request_id, "error": str(e)})
                    raise HTTPException(status_code=500, detail="failed to serialize request")

            try:
                with open(local_path, "w", encoding="utf-8") as fh:
                    fh.write(payload)
            except Exception as e:
                LOG.exception("failed to write artifact", extra={"request_id": request_id, "path": local_path, "error": str(e)})
                raise HTTPException(status_code=500, detail="failed to persist artifact")

            # Use gpu_utils.model_context to ensure any heavy model resources
            # are loaded and deterministically released. The loader attempts to
            # call an agent-provided load method if available, otherwise the
            # agent itself is used as the handle. In fixture mode load/teardown
            # are no-ops so unit tests remain fast.
            def _loader():
                agent = getattr(app.state, "agent", None)
                if agent is None:
                    return None
                load_fn = getattr(agent, "load_model", None)
                if callable(load_fn):
                    return load_fn()
                return agent

            artifact_uri = None
            artifact_name = Path(local_path).name
            try:
                with gpu_utils.model_context(_loader, name="videos_wan") as model_handle:
                    # model_handle may be agent or a model object; prefer
                    # calling .info() when present so warmup happens.
                    try:
                        if model_handle is not None and hasattr(model_handle, "info"):
                            _ = model_handle.info()
                    except Exception:
                        pass

                    # publish using the shared ADK helpers if available, with safe fallbacks
                    try:
                        import sparkle_motion.adk_helpers as ah
                    except Exception:
                        ah = None

                    if ah is not None:
                        try:
                            try:
                                sdk_res = ah.probe_sdk()
                            except SystemExit:
                                sdk_res = None

                            if sdk_res:
                                adk_module, client = sdk_res
                                try:
                                    artifact_uri = ah.publish_with_sdk(adk_module, client, local_path, artifact_name, dry_run=False, project=None)
                                except Exception:
                                    artifact_uri = None

                            if not artifact_uri:
                                try:
                                    artifact_uri = ah.publish_with_cli(local_path, artifact_name, project=None, dry_run=False)
                                except Exception:
                                    artifact_uri = None
                        except Exception:
                            artifact_uri = None

            except Exception as e:
                LOG.exception("model load/invoke failed: %s", e)
                raise HTTPException(status_code=500, detail="model load failed")

            # In fixture/test mode prefer a local file:// URI so tests can assert on files
            if artifact_uri and os.environ.get("ADK_USE_FIXTURE", "0") == "1" and artifact_uri.startswith("artifact://"):
                artifact_uri = f"file://{os.path.abspath(local_path)}"

            if not artifact_uri:
                artifact_uri = f"file://{os.path.abspath(local_path)}"

            try:
                send_telemetry("invoke.completed", {"tool": "videos_wan", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                pass

            resp = ResponseModel(status="success", artifact_uri=artifact_uri, request_id=request_id)
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()
