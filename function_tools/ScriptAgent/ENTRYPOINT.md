# ScriptAgent Entrypoint Checklist

This checklist documents the required behaviors and hooks for a FunctionTool
entrypoint implementing the ScriptAgent stage. Use as a developer checklist
and as acceptance criteria for per-tool smoke tests.

Required endpoints and contracts
- **GET /health**: returns 200 and a JSON health object (e.g. `{ "status": "ok" }`).
- **GET /ready** (optional): readiness probe distinguishing warmup vs liveness.
- **POST /invoke**: primary invoke contract accepting JSON that validates
  against the canonical `MoviePlan` schema or a small wrapper that includes
  metadata + plan. Must return a JSON payload that includes an `artifact_uri`
  (ADK artifact:// or `file://` local fallback) and `status`.

Input validation
- Validate incoming JSON against the canonical schema (SDK/JSON Schema or
  Pydantic model). Reject with 400 and a clear error payload on validation
  errors.
- Sanitize and enforce limits on fields that affect resource usage (e.g.,
  `max_frames`, `resolution`, `model` choices).

Observability & telemetry
- Emit structured logs (JSON-friendly) with stage name, run_id, and trace id.
- Expose metrics endpoints or integrate with SDK telemetry hooks (e.g.,
  ADK telemetry client) for invocation counts, latency, and error rates.

ADK integration
- When producing artifacts, prefer publishing via ADK `ArtifactService` and
  returning canonical `artifact://` URIs.
- Provide a `--local-fallback` or `LOCAL_MODE` env var path where local
  `file://` artifacts are written when ADK credentials are unavailable.

Behavioral requirements
- Idempotent tool registration metadata: entrypoint must support returning
  tool metadata (name, version, response_json_schema) for registrars.
- Graceful shutdown: handle SIGTERM/SIGINT, finish in-flight work or mark
  run state so WorkflowAgent can resume.
- Health checks: start in a warming state if model loading required, only
  return ready after successful warmup.

Security & configuration
- Read secrets from environment only (no hard-coded credentials).
- Fail fast with useful error messages if required env vars are missing.

Testing & smoke harness
- Provide an in-process TestClient-compatible FastAPI `app` object so tests
  can exercise endpoints without a network server.
- Provide a small deterministic mode (e.g., `DETERMINISTIC=1`) to make
  artifact outputs stable for unit tests.

Acceptance
- Unit tests exercise `/health` and `/invoke` paths and assert the response
  shape includes `artifact_uri` and `status: success` for valid inputs.
