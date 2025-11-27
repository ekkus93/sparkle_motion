from __future__ import annotations
from typing import Any, Literal
import re
import subprocess
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
from sparkle_motion.function_tools.entrypoint_common import send_telemetry

LOG = logging.getLogger("script_agent.entrypoint")
LOG.setLevel(logging.INFO)


def publish_artifact(local_path: str) -> str:
    """Publish an artifact using the available integration path.

    Behavior:
    - If `ADK_USE_FIXTURE` != "0" (default), return a `file://` URI for local tests.
    - Otherwise, attempt to call `adk artifacts publish --file <local_path>` and parse
      an `artifact://...` URI from stdout/stderr. If the CLI is missing or parsing
      fails, fall back to `file://`.
    """
    # If tests or local runs prefer fixture mode, keep local file URI
    if os.environ.get("ADK_USE_FIXTURE", "1") != "0":
        return f"file://{os.path.abspath(local_path)}"

    # attempt to use 'adk' CLI as a best-effort integration path
    try:
        proc = subprocess.run(["adk", "artifacts", "publish", "--file", local_path], capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        import re as _re
        m = _re.search(r"(artifact://[\w\-\./]+)", out)
        if m:
            return m.group(1)
        if proc.returncode == 0:
            # no artifact URI found, but command succeeded — return file fallback
            return f"file://{os.path.abspath(local_path)}"
    except FileNotFoundError:
        LOG.debug("adk CLI not found; falling back to file URI")
    except Exception:
        LOG.exception("adk publish attempt failed")

    return f"file://{os.path.abspath(local_path)}"


class RequestModel(BaseModel):
    """Request schema for ScriptAgent.

    This is intentionally conservative for scaffolding but validates the
    common fields used by smoke tests. Per-tool entrypoints should tighten
    this model to match the tool's contract.
    """
    title: str | None = None
    shots: list[dict] | None = None
    prompt: str | None = None


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
        LOG.info("script_agent ready")
        try:
            send_telemetry("tool.ready", {"tool": "script_agent"})
        except Exception:
            pass
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="script_agent Entrypoint (scaffold)", lifespan=lifespan)
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
    def invoke(req: RequestModel) -> ResponseModel:
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
            send_telemetry("invoke.received", {"tool": "script_agent", "request_id": request_id})
        except Exception:
            pass
        with app.state.lock:
            app.state.inflight += 1
        try:
            # minimal validation: accept prompt OR title OR non-empty shots
            has_prompt = bool(getattr(req, "prompt", None))
            has_title = bool(getattr(req, "title", None))
            has_shots = bool(getattr(req, "shots", None))
            if not (has_prompt or has_title or has_shots):
                raise HTTPException(status_code=400, detail="empty request: provide prompt, title, or shots")

            # prepare artifact persistence
            deterministic = os.environ.get("DETERMINISTIC", "1") == "1"
            artifacts_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.getcwd(), "artifacts"))
            os.makedirs(artifacts_dir, exist_ok=True)

            # sanitize filename
            content_for_name = getattr(req, "prompt", None) or getattr(req, "title", None) or "artifact"
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
                # last-resort: serialize minimal dict
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

            # publish: use module-level helper (fixture-mode by default)
            artifact_uri = publish_artifact(local_path)
            try:
                send_telemetry("invoke.completed", {"tool": "script_agent", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                pass

            resp = ResponseModel(status="success", artifact_uri=artifact_uri, request_id=request_id)
            # Return JSON-serializable dict to avoid FastAPI response/model mismatches
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()
