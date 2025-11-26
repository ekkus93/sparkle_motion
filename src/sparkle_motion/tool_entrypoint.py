from __future__ import annotations

from typing import Any, Awaitable, Callable
import inspect

from fastapi import FastAPI, Request


def create_app(tool_id: str, invoke_fn: Callable[[dict], Any]) -> FastAPI:
    """Create a lightweight FastAPI app for a FunctionTool.

    invoke_fn may be sync or async and must accept a dict request body and
    return a JSON-serializable response.
    """
    app = FastAPI(title=f"FunctionTool: {tool_id}")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "tool": tool_id}

    @app.post("/invoke")
    async def invoke(req: Request):
        body = await req.json()
        if inspect.iscoroutinefunction(invoke_fn):
            result = await invoke_fn(body)
        else:
            # sync call allowed
            result = invoke_fn(body)
        return {"tool": tool_id, "result": result}

    return app


def mount_handlers(app: FastAPI, *, tool_id: str, invoke_fn: Callable[[dict], Any]):
    """Backward-compatible alias for create_app semantics when embedding."""
    # attach as attribute for tests/tools to inspect
    app.state.tool_id = tool_id
    app.state.invoke_fn = invoke_fn
    return app
