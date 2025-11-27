from __future__ import annotations
from typing import Any
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
from pydantic import BaseModel
from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion import adk_factory, observability, telemetry

LOG = logging.getLogger("qa_qwen2vl.entrypoint")
LOG.setLevel(logging.INFO)


class RequestModel(BaseModel):
    # TODO: adjust fields for real schema
    prompt: str


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
            model_spec = os.environ.get("QA_QWEN2VL_MODEL", "qwen-2vl-default")
            seed = int(os.environ.get("QA_QWEN2VL_SEED")) if os.environ.get("QA_QWEN2VL_SEED") else None
            app.state.agent = adk_factory.get_agent("qa_qwen2vl", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="qa_qwen2vl")
                telemetry.emit_event("agent.created", {"tool": "qa_qwen2vl", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception:
            LOG.exception("failed to construct ADK agent for qa_qwen2vl")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("qa_qwen2vl ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "qa_qwen2vl"})
        except Exception:
            pass
        try:
            telemetry.emit_event("tool.ready", {"tool": "qa_qwen2vl"})
        except Exception:
            pass
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="qa_qwen2vl Entrypoint (scaffold)", lifespan=lifespan)
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
            send_telemetry("invoke.received", {"tool": "qa_qwen2vl", "request_id": request_id})
        except Exception:
            pass
        try:
            telemetry.emit_event("invoke.received", {"tool": "qa_qwen2vl", "request_id": request_id})
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
            safe_name = (getattr(req, "prompt", "artifact")[:80]).replace(" ", "_")
            filename = f"{safe_name}.json" if deterministic else f"{safe_name}_{os.getpid()}_{request_id}.json"
            local_path = os.path.join(artifacts_dir, filename)
            try:
                with open(local_path, "w", encoding="utf-8") as fh:
                    fh.write(req.model_dump_json() if hasattr(req, "model_dump_json") else req.json())
            except Exception as e:
                LOG.exception("failed to write artifact", extra={"request_id": request_id, "path": local_path, "error": str(e)})
                raise HTTPException(status_code=500, detail="failed to persist artifact")

            # publish via ADK helpers (SDK -> CLI) with fallback to file://
            artifact_uri = None
            artifact_name = Path(local_path).name
            try:
                import sparkle_motion.adk_helpers as ah
            except Exception:
                ah = None

            if ah is not None:
                try:
                    sdk_res = ah.probe_sdk()
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

            # In fixture/test mode prefer a local file:// URI so tests can assert on files
            if artifact_uri and os.environ.get("ADK_USE_FIXTURE", "0") == "1" and artifact_uri.startswith("artifact://"):
                artifact_uri = f"file://{os.path.abspath(local_path)}"

            if not artifact_uri:
                artifact_uri = f"file://{os.path.abspath(local_path)}"

            try:
                send_telemetry("invoke.completed", {"tool": "qa_qwen2vl", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                pass
            try:
                telemetry.emit_event("invoke.completed", {"tool": "qa_qwen2vl", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                pass

            return {"status": "success", "artifact_uri": artifact_uri, "request_id": request_id}
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()
