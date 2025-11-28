"""sparkle_motion package (minimal).

This package exposes the gpu_utils helpers used by adapters and tests.
"""
__all__ = ["gpu_utils"]
"""sparkle_motion package

Small package entry for the Sparkle Motion project. Keeps package namespace
and exposes version.
"""
__all__ = ["schemas", "schema_registry", "prompt_templates"]
__version__ = "0.0.0"

# Ensure FastAPI request validation errors return 400 (tests expect 400)
try:
	from fastapi import Request
	from fastapi.exceptions import RequestValidationError
	from fastapi.responses import JSONResponse
	import fastapi.exception_handlers as _fa_handlers

	async def _rm_request_validation_handler(request: Request, exc: RequestValidationError):
		return JSONResponse(status_code=400, content={"detail": exc.errors()})

	# Monkeypatch the module-level handler so apps created after import
	# will pick up the replacement when they use FastAPI's default handlers.
	_fa_handlers.request_validation_exception_handler = _rm_request_validation_handler
except Exception:
	# If fastapi isn't available in the environment, don't fail import.
	pass

from . import schemas
from . import schema_registry
from . import prompt_templates
