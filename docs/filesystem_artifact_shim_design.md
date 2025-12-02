# Filesystem ArtifactService Shim — Endpoint & Error Design

## Context

The filesystem shim lets Sparkle Motion run real production workflows without
Google Cloud services while emitting artifacts that are contract-compatible with
ADK. It must:

- Emit manifests, URIs, and metadata indistinguishable from ADK ArtifactService
  so `/status`, `/artifacts`, and notebooks remain backend-agnostic.
- Persist payloads under a deterministic local root and track metadata inside a
  SQLite index referenced by production dashboards.
- Stay single-user and single-job, prioritizing simplicity over multitenancy.

Configuration knobs:

- `ARTIFACTS_BACKEND` — `"adk"` (default) or `"filesystem"` to select this shim.
- `ARTIFACTS_FS_ROOT` — directory root for payloads (default `./artifacts_fs`).
- `ARTIFACTS_FS_INDEX` — SQLite file (default `./artifacts_fs/index.db`).
- `ARTIFACTS_FS_BASE_URL` — base URL when the shim runs as HTTP service
  (default `http://127.0.0.1:7077`).
- `ARTIFACTS_FS_TOKEN` — shared bearer token required on every HTTP request.

## API surface

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/artifacts` | Uploads payload + manifest and returns canonical artifact metadata. |
| `GET` | `/artifacts/{artifact_id}` | Retrieves manifest + metadata for a single artifact. |
| `GET` | `/artifacts` | Lists manifests filtered by `run_id`, `stage`, `artifact_type`, etc. |
| `GET` | `/healthz` | Optional readiness probe (no auth required). |

### `POST /artifacts`

- **Auth**: `Authorization: Bearer ${ARTIFACTS_FS_TOKEN}`.
- **Content types**:
  - `multipart/form-data` for binary payloads (part `file` + part `manifest`).
  - `application/json` when only metadata is uploaded (e.g., manifest-only
    artifacts); payload content may be embedded as base64 in `body.bytes`.
- **Required fields** (body JSON or multipart `manifest` part):
  - `run_id`: string (validated against `^[a-zA-Z0-9._-]+$`).
  - `stage`: string (`plan_intake`, `dialogue_audio`, etc.).
  - `artifact_type`: string (`movie_plan`, `video_final`, etc.).
  - `mime_type`: string (IANA media type).
  - `metadata`: object with arbitrary JSON metadata; must include
    `schema_uri`, `checksum.sha256`, `size_bytes`, and QA linkage when
    applicable.
- **Optional fields**:
  - `local_path_hint`: path used by the caller before upload (recorded for
    traceability only).
  - `qa`: structured QA payload (decision, report URIs, issues).
- **Response** (`201 Created`):
  ```json
  {
    "artifact_id": "run_123/stage/dialogue_audio/c2f9...",
    "artifact_uri": "artifact+fs://run_123/c2f9...",
    "manifest": { ... },
    "storage": {
      "backend": "filesystem",
      "relative_path": "run_123/dialogue_audio/c2f9.../artifact.bin"
    }
  }
  ```

### `GET /artifacts/{artifact_id}`

- Looks up the artifact row inside `ARTIFACTS_FS_INDEX`.
- Responds with `404` when the ID is missing or tombstoned.
- Payload:
  ```json
  {
    "artifact_id": "run_123/...",
    "artifact_uri": "artifact+fs://run_123/...",
    "run_id": "run_123",
    "stage": "dialogue_audio",
    "artifact_type": "tts_timeline",
    "mime_type": "audio/wav",
    "metadata": { ... },
    "paths": {
      "relative_path": "run_123/dialogue_audio/.../tts_timeline.wav",
      "absolute_path": "/content/sparkle_motion/artifacts_fs/..."
    }
  }
  ```
- Supports `?include_payload=true` to stream the stored file when the caller
  needs bytes (used only for troubleshooting; dashboards stick to metadata).

### `GET /artifacts`

- Query parameters:
  - `run_id` (required).
  - Optional `stage`, `artifact_type`, `limit` (default 50), `page_token`.
  - `order=asc|desc` (default `desc`, newest first).
- Response contains `items` array (same schema as single lookup) and optional
  `next_page_token`.
- Implementation reads from SQLite index with ORDER + LIMIT, ensuring listings
  remain fast even when the filesystem tree grows large.

### `GET /healthz`

- Returns `200 OK` with `{ "status": "ok" }`.
- No auth; used by notebooks to wait for the shim to start.

## Authentication & authorization

- Single shared secret transported as `Authorization: Bearer <token>`.
- Token value sourced from `ARTIFACTS_FS_TOKEN` env var; shim refuses to start
  without it unless `ARTIFACTS_FS_ALLOW_INSECURE=1` is set for local debugging.
- All write operations log the caller (from `X-Caller` header or `run_id`) via
  `adk_helpers.write_memory_event()`.

## Storage layout & indexing

### Deterministic directory schema

Payloads live under `${ARTIFACTS_FS_ROOT}/${run_id}/${stage}/${artifact_id}/` and
always contain at least `artifact.bin` (opaque bytes) and `manifest.json`
(ADK-compatible metadata). `artifact_id` is generated as
`${run_id}/${stage}/${artifact_type}/${uuid4_hex}` to guarantee uniqueness while
remaining debuggable. Stages map 1:1 with production_agent stages (e.g.,
`plan_intake`, `dialogue_audio`, `shot_001`, `qa_publish`).

Rules:
- Directory names must be URL-safe (`[a-zA-Z0-9._-]`).
- `manifest.json` is the single source of truth for metadata; callers never
  edit payloads without updating the manifest.
- Binary payload file names mirror their MIME types when obvious
  (`artifact.bin` for arbitrary bytes, `artifact.png`, `artifact.wav`, etc.) to
  ease manual inspection. Manifests include the canonical filename so callers
  can reconstruct paths if naming conventions change later.

```
${ARTIFACTS_FS_ROOT}/
  run_<id>/
    <stage>/
      <artifact_id>/
        artifact.bin   # opaque payload
        manifest.json  # ADK-compatible manifest
```

SQLite schema (mirrors `docs/ARCHITECTURE.md` guidance):

```sql
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  checksum_sha256 TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  qa_decision TEXT,
  metadata TEXT
);
CREATE INDEX IF NOT EXISTS ix_artifacts_run_stage ON artifacts(run_id, stage);
CREATE INDEX IF NOT EXISTS ix_artifacts_type ON artifacts(artifact_type);
```

## Error semantics

| Status | Error code | When it fires | Notes |
| --- | --- | --- | --- |
| `400 Bad Request` | `invalid_request` | Missing required fields, invalid JSON, stage/type not allowed. | Response includes `fields` array describing validation failures. |
| `401 Unauthorized` | `invalid_token` | Missing/incorrect bearer token. | All non-health endpoints require auth. |
| `404 Not Found` | `artifact_missing` | Unknown artifact ID or run/stage combination. | Listing with no results returns `200` + empty list instead. |
| `409 Conflict` | `artifact_exists` | Re-upload attempts with the same `artifact_id` unless `overwrite=true`. | Includes existing manifest in response payload. |
| `413 Payload Too Large` | `payload_too_large` | Uploaded file exceeds `ARTIFACTS_FS_MAX_BYTES`. | Limit defaults to 12 GiB; configurable via env. |
| `415 Unsupported Media Type` | `bad_content_type` | Missing `multipart/form-data` for payload uploads. | Encourages clients to send proper MIME hints. |
| `429 Too Many Requests` | `fs_backend_backpressure` | Shim detects disk quota exhaustion or throttling. | Client should retry with backoff. |
| `500 Internal Server Error` | `fs_backend_failure` | Unhandled exceptions (I/O errors, SQLite corruption). | Response hides local paths but logs them internally. |

All error responses share the shape:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "stage is required",
    "details": {"fields": ["stage"]}
  }
}
```

## Compatibility requirements

- URI scheme: `artifact+fs://<run_id>/<artifact_id>` to distinguish from ADK
  while keeping the `artifact://` contract intact for existing dashboards when
  the helper resolves URIs.
- Manifest JSON mirrors ADK’s keys: `artifact_type`, `artifact_uri`,
  `local_path`, `download_url` (file:// path), `schema_uri`, `checksum`,
  `size`, `qa` metadata, and `tags`.
- `/status` and `/artifacts` continue to read manifests via helper APIs; they
  should not know whether data lives in ADK or the filesystem.
- Publish telemetry via `adk_helpers.write_memory_event()` whenever artifacts
  are saved, include `storage_backend="filesystem"` so operators see the
  backend choice in dashboards/logs.

## Out-of-scope / follow-ups

- Multi-tenant IAM, signed download URLs, lifecycle policies, and cross-host
  synchronization remain explicitly out of scope.
- Garbage collection tooling is tracked as a separate P0 task: implement CLI
  utilities that prune by age/size and surface dry-run summaries.
- Drive sync automation (e.g., rsync from local disk to Google Drive) stays as
  manual operator responsibility for now but should leverage the same manifest
  metadata when built.
