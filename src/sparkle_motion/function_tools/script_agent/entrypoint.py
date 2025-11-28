from __future__ import annotations

from typing import Any, Literal, Optional
import re
import os
import json
import logging
import uuid
import time
import asyncio
from threading import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion import adk_helpers
from sparkle_motion import adk_factory, observability, telemetry

LOG = logging.getLogger("script_agent.entrypoint")
LOG.setLevel(logging.INFO)


class RequestModel(BaseModel):
    """Request schema for ScriptAgent with conservative validation.

    At least one of `prompt`, `title`, or `shots` must be provided.
    """
    title: Optional[str] = None
    shots: Optional[list[dict]] = None
    prompt: Optional[str] = None

    @model_validator(mode="after")
    def at_least_one(self):
        if not (self.prompt or self.title or self.shots):
            raise ValueError("empty request: provide prompt, title, or shots")
        return self


class ResponseModel(BaseModel):
    status: Literal["success", "error"]
    artifact_uri: Optional[str] = None
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

        # Eagerly construct per-tool agent (fixture mode returns dummy agent)
        try:
            model_spec = os.environ.get("SCRIPT_AGENT_MODEL", "script-agent-default")
            seed = int(os.environ.get("SCRIPT_AGENT_SEED")) if os.environ.get("SCRIPT_AGENT_SEED") else None
            app.state.agent = adk_factory.get_agent("script_agent", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="script_agent")
                telemetry.emit_event("agent.created", {"tool": "script_agent", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception:
            LOG.exception("failed to construct ADK agent for script_agent")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("script_agent ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "script_agent"})
        except Exception:
            LOG.exception("telemetry send failed on ready")
        try:
            telemetry.emit_event("tool.ready", {"tool": "script_agent"})
        except Exception:
            pass
        try:
            yield
        finally:
            # graceful shutdown: mark and wait for inflight ops
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 5.0:
                await asyncio.sleep(0.05)
            try:
                send_telemetry("tool.shutdown", {"tool": "script_agent"})
            except Exception:
                LOG.exception("telemetry send failed on shutdown")

    app = FastAPI(title="script_agent Entrypoint", lifespan=lifespan)
    app.state.lock = Lock()
    # During fixture/test runs we prefer deterministic startup.
    # If `ADK_USE_FIXTURE=1` we mark ready immediately so tests don't
    # race on the lifespan startup timing.
    app.state.ready = os.environ.get("ADK_USE_FIXTURE", "1") != "0"
    app.state.shutting_down = False
    app.state.inflight = 0

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
            raise HTTPException(status_code=503, detail="tool not ready")
        if getattr(app.state, "shutting_down", False):
            raise HTTPException(status_code=503, detail="shutting down")

        request_id = uuid.uuid4().hex
        LOG.info("invoke.received", extra={"request_id": request_id})
        try:
            send_telemetry("invoke.received", {"tool": "script_agent", "request_id": request_id})
        except Exception:
            LOG.exception("telemetry send failed on invoke.received")
        try:
            telemetry.emit_event("invoke.received", {"tool": "script_agent", "request_id": request_id})
        except Exception:
            pass
        with app.state.lock:
            app.state.inflight += 1
        try:
            # prepare artifact persistence
            deterministic = os.environ.get("DETERMINISTIC", "1") == "1"
            artifacts_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.getcwd(), "artifacts"))
            os.makedirs(artifacts_dir, exist_ok=True)

            # sanitize filename
            content_for_name = req.prompt or req.title or "artifact"
            def _slugify(s: str) -> str:
                s = s.strip()
                s = re.sub(r"\s+", "_", s)
                s = re.sub(r"[^A-Za-z0-9._-]", "", s)
                return s[:80] or "artifact"

            safe_name = _slugify(str(content_for_name))
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

            # publish via ADK helpers façade (SDK-first, CLI fallback).
            artifact_ref = adk_helpers.publish_artifact(
                local_path=local_path,
                artifact_type="script_agent_movie_plan",
                metadata={"request_id": request_id, "tool": "script_agent"},
            )
            artifact_uri = artifact_ref["uri"]
            try:
                send_telemetry("invoke.completed", {"tool": "script_agent", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                LOG.exception("telemetry send failed on invoke.completed")
            try:
                telemetry.emit_event("invoke.completed", {"tool": "script_agent", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                pass

            resp = ResponseModel(status="success", artifact_uri=artifact_uri, request_id=request_id)
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()
