"""FastAPI application for the filesystem ArtifactService shim."""

from __future__ import annotations

import base64
import binascii
import sqlite3
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile
from pydantic import ValidationError

from .config import FilesystemArtifactsConfig
from .models import (
    ArtifactJsonUpload,
    ArtifactListResponse,
    ArtifactManifest,
    ArtifactRecord,
)
from .storage import ArtifactStorageError, FilesystemArtifactStore, PayloadTooLargeError

_MAX_LIST_LIMIT = 200
_DEFAULT_LIST_LIMIT = 50


def create_app(
    config: FilesystemArtifactsConfig | None = None,
    *,
    store: FilesystemArtifactStore | None = None,
) -> FastAPI:
    """Create a FastAPI app that exposes the filesystem artifact APIs."""

    cfg = config or FilesystemArtifactsConfig.from_env()
    artifact_store = store or FilesystemArtifactStore(cfg)
    app = FastAPI(title="Filesystem ArtifactService Shim")

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        content = exc.detail if isinstance(exc.detail, dict) else {
            "error": {
                "code": "fs_backend_failure",
                "message": str(exc.detail),
                "details": {},
            }
        }
        return JSONResponse(status_code=exc.status_code, content=content)

    async def _require_auth(request: Request) -> None:
        if cfg.allow_insecure:
            return
        header = request.headers.get("authorization") or ""
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise _http_error(status.HTTP_401_UNAUTHORIZED, "invalid_token", "Missing bearer token")
        if cfg.token is None or token.strip() != cfg.token:
            raise _http_error(status.HTTP_401_UNAUTHORIZED, "invalid_token", "Invalid bearer token")

    @app.post("/artifacts", response_model=ArtifactRecord, status_code=status.HTTP_201_CREATED)
    async def upload_artifact(request: Request, _: None = Depends(_require_auth)) -> ArtifactRecord:
        manifest, payload, filename_hint = await _parse_upload_request(request)
        try:
            return artifact_store.save_artifact(
                manifest=manifest,
                payload=payload,
                filename_hint=filename_hint,
            )
        except PayloadTooLargeError as exc:
            raise _http_error(status.HTTP_413_CONTENT_TOO_LARGE, "payload_too_large", str(exc)) from exc
        except ArtifactStorageError as exc:
            raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", str(exc)) from exc
        except sqlite3.Error as exc:
            raise _http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "fs_backend_failure",
                "Failed to persist artifact metadata",
            ) from exc
        except OSError as exc:
            raise _http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "fs_backend_failure",
                "Filesystem backend reported an I/O error",
            ) from exc

    @app.get("/artifacts/{artifact_id:path}", response_model=ArtifactRecord)
    async def get_artifact(
        artifact_id: str,
        include_payload: bool = False,
        _: None = Depends(_require_auth),
    ) -> ArtifactRecord | FileResponse:
        record = artifact_store.get_artifact(artifact_id)
        if record is None:
            raise _http_error(status.HTTP_404_NOT_FOUND, "artifact_missing", "Artifact not found")
        if include_payload:
            payload_path = Path(record.storage.absolute_path)
            return FileResponse(
                path=payload_path,
                media_type=record.mime_type,
                filename=payload_path.name,
            )
        return record

    @app.get("/artifacts", response_model=ArtifactListResponse)
    async def list_artifacts(
        run_id: str,
        stage: str | None = None,
        artifact_type: str | None = None,
        limit: int = _DEFAULT_LIST_LIMIT,
        order: Literal["asc", "desc"] = "desc",
        page_token: str | None = None,
        _: None = Depends(_require_auth),
    ) -> ArtifactListResponse:
        if order not in ("asc", "desc"):
            raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", "order must be 'asc' or 'desc'")
        normalized_limit = max(1, min(limit, _MAX_LIST_LIMIT))
        page_marker = _parse_page_token(page_token) if page_token else None
        items, next_token = artifact_store.list_artifacts(
            run_id=run_id,
            stage=stage,
            artifact_type=artifact_type,
            limit=normalized_limit,
            order=order,
            page_marker=page_marker,
        )
        return ArtifactListResponse(items=items, next_page_token=next_token)

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def _parse_upload_request(request: Request) -> tuple[ArtifactManifest, bytes | None, str | None]:
    content_type = (request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        manifest = _parse_manifest_field(form.get("manifest"))
        file_field = form.get("file")
        payload: bytes | None = None
        filename_hint: str | None = None
        if isinstance(file_field, UploadFile):
            payload = await file_field.read()
            filename_hint = file_field.filename
        elif isinstance(file_field, bytes):
            payload = file_field
        elif isinstance(file_field, str):
            payload = file_field.encode("utf-8")
        elif file_field not in (None, ""):
            raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", "file field must be an uploaded file")
        return manifest, payload, filename_hint

    if content_type.startswith("application/json"):
        try:
            raw = await request.json()
        except ValueError as exc:
            raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", "Invalid JSON body") from exc
        try:
            envelope = ArtifactJsonUpload.model_validate(raw)
        except ValidationError as exc:
            raise _validation_error(exc)
        payload = _decode_base64(envelope.payload_b64) if envelope.payload_b64 else None
        return envelope.manifest, payload, envelope.filename_hint

    raise _http_error(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "bad_content_type",
        "Only multipart/form-data or application/json bodies are supported",
    )


def _parse_page_token(token: str) -> tuple[int, str]:
    try:
        created_at_str, artifact_id = token.split(":", 1)
        return int(created_at_str), artifact_id
    except ValueError as exc:
        raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", "Malformed page_token") from exc


def _parse_manifest_field(raw_manifest: Any) -> ArtifactManifest:
    if raw_manifest is None:
        raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", "manifest field is required")
    text = raw_manifest
    if isinstance(raw_manifest, bytes):
        text = raw_manifest.decode("utf-8")
    if not isinstance(text, str):
        raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", "manifest field must be JSON text")
    try:
        return ArtifactManifest.model_validate_json(text)
    except ValidationError as exc:
        raise _validation_error(exc)


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _http_error(status.HTTP_400_BAD_REQUEST, "invalid_request", "payload_b64 is not valid base64") from exc


def _validation_error(exc: ValidationError) -> HTTPException:
    raise _http_error(
        status.HTTP_400_BAD_REQUEST,
        "invalid_request",
        "Request validation failed",
        details={"errors": exc.errors()},
    )


def _http_error(status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            }
        },
    )
