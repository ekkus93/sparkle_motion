"""Compatibility entrypoint module for tools that expect
`sparkle_motion.function_tools.entrypoint_common.entrypoint`.

This helper provides a minimal FastAPI `app` and `make_app()` so the smoke
test harness can import it without errors. It intentionally implements a
lightweight health/ready/invoke surface that returns deterministic responses.
"""
from __future__ import annotations

from . import send_telemetry
from typing import Any, Dict

from fastapi import FastAPI

__all__ = ["send_telemetry", "app", "make_app"]


def make_app() -> FastAPI:
	app = FastAPI()

	@app.get("/health")
	def health() -> Dict[str, Any]:
		return {"status": "ok"}

	@app.get("/ready")
	def ready() -> Dict[str, Any]:
		# In test mode this helper is always ready
		return {"ready": True}

	from starlette.responses import JSONResponse

	@app.post("/invoke")
	def invoke(payload: Dict[str, Any]):
		# Not a real tool — return a 503 with JSON body to indicate not-implemented
		return JSONResponse(status_code=503, content={"status": "not-implemented", "artifact_uri": ""})

	return app


# Provide a module-level app for convenience
app = make_app()
