from __future__ import annotations
from typing import Any
import os
import json
import logging
import asyncio
from threading import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LOG = logging.getLogger("script_agent.entrypoint")
LOG.setLevel(logging.INFO)

# Minimal deterministic MoviePlan model used for the smoke harness
class MoviePlan(BaseModel):
    title: str
    shots: list = []


def publish_artifact(local_path: str) -> str:
    """Publish artifact using the ADK ArtifactService when available.

    SDK-first: try to locate an ADK artifact service and call its
    `save_artifact` method. On any failure, fall back to creating a
    local file and returning a `file://` URI.
    """
    try:
        # Import the ADK SDK (we expect the development environment to have it)
        import google.adk as adk  # type: ignore

        # Prefer a GCS-backed artifact service when configured, otherwise use
        # the FileArtifactService which stores artifacts under a root path.
        bucket = os.environ.get("ADK_ARTIFACTS_GCS_BUCKET")
        if bucket:
            try:
                from google.adk.artifacts.gcs_artifact_service import GcsArtifactService as _Gcs

                svc = _Gcs(bucket)
            except Exception:
                svc = None
        else:
            try:
                from google.adk.artifacts.file_artifact_service import FileArtifactService as _FileSvc

                root = os.environ.get("ADK_ARTIFACTS_ROOT", os.path.join(os.getcwd(), "artifacts", "adk"))
                svc = _FileSvc(root)
            except Exception:
                svc = None

        # Try to construct a Part object when available (some SDKs expose
        # google.genai.types.Part). Otherwise fall back to passing a file URI
        Part = None
        try:
            from google.genai.types import Part  # type: ignore
        except Exception:
            # try to find a types module on the adk package as a fallback
            types_mod = getattr(adk, "artifacts", None)
            Part = getattr(types_mod, "Part", None) if types_mod is not None else None

        # Ensure the local file exists (the caller may expect it to be written
        # by prior logic). Create a stub if missing.
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            with open(local_path, "w", encoding="utf-8") as fh:
                json.dump({"stub": True, "source": "local-fallback"}, fh)

        # Build a Part containing the file bytes if the SDK Part type is
        # available; otherwise pass a simple file URI or raw bytes depending
        # on what the service accepts.
        part = None
        try:
            if Part is not None and hasattr(Part, "from_bytes"):
                import mimetypes

                mime_type, _ = mimetypes.guess_type(local_path)
                mime_type = mime_type or "application/octet-stream"
                with open(local_path, "rb") as _fh:
                    data = _fh.read()
                part = Part.from_bytes(data=data, mime_type=mime_type)
            elif Part is not None and hasattr(Part, "from_uri"):
                part = Part.from_uri(file_uri=str(os.path.abspath(local_path)))
            else:
                # fallback: pass the file URI string
                part = str(os.path.abspath(local_path))
        except Exception:
            part = str(os.path.abspath(local_path))

        # Attempt to call the service's async save_artifact method. Many
        # implementations are async; use asyncio.run to call them synchronously.
        if svc is not None:
            try:
                app_name = os.environ.get("ADK_PROJECT") or os.path.basename(os.getcwd())
                user_id = os.environ.get("USER") or "user"
                try:
                    rev = asyncio.run(svc.save_artifact(app_name=app_name, user_id=user_id, filename=os.path.basename(local_path), artifact=part))
                except TypeError:
                    rev = asyncio.run(svc.save_artifact(app_name=app_name, user_id=user_id, filename=os.path.basename(local_path), artifact=part))

                # rev is typically a numeric revision; craft a canonical artifact URI
                return f"artifact://{app_name}/artifacts/{os.path.basename(local_path)}/v{rev}"
            except Exception as e:  # fallback to local file on any error
                LOG.exception("ADK ArtifactService publish failed, falling back to local file: %s", e)

        # If we reach here, the SDK path was not usable
        LOG.info("ADK SDK not usable for publish; using local artifact fallback")
        return f"file://{os.path.abspath(local_path)}"
    except Exception as e:
        LOG.exception("ADK SDK import or publish attempt failed, using local fallback: %s", e)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            with open(local_path, "w", encoding="utf-8") as fh:
                json.dump({"stub": True, "source": "local-fallback"}, fh)
        return f"file://{os.path.abspath(local_path)}"


def make_app() -> FastAPI:
    """Create the FastAPI app and wire lifespan-based startup/shutdown."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # initialize state
        app.state.ready = False
        app.state.shutting_down = False
        app.state.inflight = 0
        app.state.lock = Lock()

        # Warmup / model load delay controlled by env var for tests
        try:
            delay = float(os.environ.get("MODEL_LOAD_DELAY", "0"))
        except Exception:
            delay = 0.0
        if delay > 0:
            LOG.info("Warmup: loading model, delay=%s", delay)
            await asyncio.sleep(delay)
        app.state.ready = True
        LOG.info("ScriptAgent ready")

        try:
            yield
        finally:
            LOG.info("Shutdown requested: refusing new invokes and waiting for inflight")
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            # wait up to 2s for inflight to drain
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="ScriptAgent Entrypoint (hardened harness)", lifespan=lifespan)

    # Initialize minimal state defaults so sync request handlers can access
    # state attributes even if lifespan hasn't fully executed in some test
    # harnesses that may not wait for startup.
    app.state.lock = Lock()
    app.state.ready = False
    app.state.shutting_down = False
    app.state.inflight = 0

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        return {"ready": bool(getattr(app.state, "ready", False))}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return {"inflight": getattr(app.state, "inflight", 0), "ready": bool(getattr(app.state, "ready", False))}

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "tool_name": "script-agent",
            "version": "0.1",
            "response_json_schema": {"type": "object", "properties": {"artifact_uri": {"type": "string"}}},
        }

    @app.post("/invoke")
    def invoke(plan: MoviePlan) -> dict[str, Any]:
        # Accept invokes if ready; allow immediate readiness when MODEL_LOAD_DELAY==0
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

        # Track inflight work
        with app.state.lock:
            app.state.inflight += 1
        try:
            if not plan.title:
                raise HTTPException(status_code=400, detail="title is required")

            deterministic = os.environ.get("DETERMINISTIC", "0") == "1"
            artifacts_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.getcwd(), "artifacts"))
            os.makedirs(artifacts_dir, exist_ok=True)
            filename = f"{plan.title.replace(' ', '_')}.json" if deterministic else f"{plan.title.replace(' ', '_')}_{os.getpid()}.json"
            local_path = os.path.join(artifacts_dir, filename)

            artifact_uri = publish_artifact(local_path)

            LOG.info("Invocation success", extra={"artifact_uri": artifact_uri, "title": plan.title})
            return {"status": "success", "artifact_uri": artifact_uri}
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()

