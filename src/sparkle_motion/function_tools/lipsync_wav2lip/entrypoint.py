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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LOG = logging.getLogger("lipsync_wav2lip.entrypoint")
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
        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("lipsync_wav2lip ready")
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

            try:
                from sparkle_motion.function_tools.script_agent.entrypoint import publish_artifact
                try:
                    artifact_uri = publish_artifact(local_path)
                except Exception:
                    artifact_uri = f"file://{os.path.abspath(local_path)}"
            except Exception:
                artifact_uri = f"file://{os.path.abspath(local_path)}"

            resp = ResponseModel(status="success", artifact_uri=artifact_uri, request_id=request_id)
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()
