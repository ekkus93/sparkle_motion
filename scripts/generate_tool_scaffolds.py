#!/usr/bin/env python3
"""Generate FunctionTool entrypoint scaffolds and matching test skeletons.

Usage:
  python scripts/generate_tool_scaffolds.py [--dry-run] [--force] [--tools a,b]

This script is conservative by default: it will only print what it would do
(`--dry-run`). To actually write files, pass `--force` to overwrite existing
scaffolds. By default it inspects `configs/tool_registry.yaml` and
`function_tools/` to discover declared tool IDs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

try:
    import yaml
except Exception:
    yaml = None


IMAGES_TEMPLATE = '''from __future__ import annotations
from typing import Any
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

LOG = logging.getLogger("{tool}.entrypoint")
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
        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("{tool} ready")
        try:
            yield
        finally:
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 2.0:
                await asyncio.sleep(0.05)

    app = FastAPI(title="{tool} Entrypoint (scaffold)", lifespan=lifespan)
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
            safe_name = (getattr(req, "prompt", "artifact")[:80]).replace(" ", "_")
            filename = f"{safe_name}.json" if deterministic else f"{safe_name}_{os.getpid()}_{request_id}.json"
            local_path = os.path.join(artifacts_dir, filename)
            try:
                with open(local_path, "w", encoding="utf-8") as fh:
                    fh.write(req.model_dump_json() if hasattr(req, "model_dump_json") else req.json())
            except Exception as e:
                LOG.exception("failed to write artifact", extra={"request_id": request_id, "path": local_path, "error": str(e)})
                raise HTTPException(status_code=500, detail="failed to persist artifact")

            # best-effort publish: reuse script_agent.publish_artifact if available
            try:
                from sparkle_motion.function_tools.script_agent.entrypoint import publish_artifact
                try:
                    artifact_uri = publish_artifact(local_path)
                except Exception:
                    artifact_uri = f"file://{os.path.abspath(local_path)}"
            except Exception:
                artifact_uri = f"file://{os.path.abspath(local_path)}"

            return {"status": "success", "artifact_uri": artifact_uri, "request_id": request_id}
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()
'''

TEST_TEMPLATE = '''from __future__ import annotations
from fastapi.testclient import TestClient
import os

from sparkle_motion.function_tools.{tool}.entrypoint import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_invoke_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    client = TestClient(app)
    payload = {"prompt": "test prompt"}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert (
        data["artifact_uri"].startswith("file://")
        or data["artifact_uri"].startswith("artifact://")
        or data["artifact_uri"].startswith("artifact+fs://")
    )

'''


def discover_tools(config_path: Path) -> List[str]:
    tools = []
    if config_path.exists() and yaml is not None:
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            tmap = data.get("tools") or {}
            tools.extend(list(tmap.keys()))
        except Exception:
            pass

    # also include any dirs under function_tools
    ft_root = Path(__file__).resolve().parents[1] / "function_tools"
    if ft_root.exists():
        for p in ft_root.iterdir():
            if p.is_dir():
                name = p.name
                if name not in tools:
                    tools.append(name)

    return sorted(set(tools))


def write_if_needed(path: Path, content: str, dry_run: bool, force: bool) -> bool:
    if path.exists() and not force:
        return False
    if dry_run:
        print(f"[dry-run] would write {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")
    return True


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Show what would be created")
    p.add_argument("--force", action="store_true", help="Overwrite existing scaffolds")
    p.add_argument("--tools", type=str, help="Comma-separated list of tools to generate (defaults to discovery)")
    p.add_argument("--config", type=Path, default=Path("configs/tool_registry.yaml"), help="Path to tool_registry.yaml")
    args = p.parse_args(argv)

    if args.tools:
        tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    else:
        tools = discover_tools(args.config)

    created = []
    for t in tools:
        module_dir = Path(__file__).resolve().parents[1] / "src" / "sparkle_motion" / "function_tools" / t
        entrypoint_path = module_dir / "entrypoint.py"
        test_path = Path(__file__).resolve().parents[1] / "tests" / "test_function_tools" / f"test_{t}_entrypoint.py"

        # Use simple replace for the {tool} placeholder to avoid having to
        # escape all braces in the template (the template intentionally
        # contains many `{}` tokens that must remain for generated code).
        entry_content = IMAGES_TEMPLATE.replace("{tool}", t)
        test_content = TEST_TEMPLATE.replace("{tool}", t)

        wrote_entry = write_if_needed(entrypoint_path, entry_content, args.dry_run, args.force)
        wrote_test = write_if_needed(test_path, test_content, args.dry_run, args.force)
        if wrote_entry or wrote_test:
            created.append((t, wrote_entry, wrote_test))

    print("Summary:")
    for t, e, te in created:
        print(f"- {t}: entrypoint_written={e}, test_written={te}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
