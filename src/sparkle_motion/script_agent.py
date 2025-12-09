"""ScriptAgent plan generation helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import ValidationError

from . import adk_factory, adk_helpers, observability, schema_registry, telemetry
from .schemas import MoviePlan

_LOG = logging.getLogger("sparkle_motion.script_agent")
_ARTIFACT_TYPE = "script_agent_movie_plan"
_DEFAULT_MODEL = "script-agent-default"


class ScriptAgentError(RuntimeError):
    """Base error for ScriptAgent helpers."""


class AgentInvocationError(ScriptAgentError):
    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        detail = f"{message}: {cause}" if cause else message
        super().__init__(detail)
        if cause is not None:
            self.__cause__ = cause


class PlanParseError(ScriptAgentError):
    def __init__(self, message: str, *, raw_output: str | None = None, cause: Exception | None = None) -> None:
        detail = message if raw_output is None else f"{message}: {raw_output[:200]}"
        super().__init__(detail)
        self.raw_output = raw_output
        if cause is not None:
            self.__cause__ = cause


class PlanSchemaError(ScriptAgentError):
    def __init__(self, message: str, *, errors: Sequence[Mapping[str, Any]] | None = None, cause: Exception | None = None) -> None:
        detail = message
        if errors:
            detail = f"{message}: {errors}"
        super().__init__(detail)
        self.errors = list(errors or [])
        if cause is not None:
            self.__cause__ = cause


class PlanResourceError(ScriptAgentError):
    pass



def generate_plan(
    prompt: str,
    *,
    model_spec: str | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> MoviePlan:
    """Generate a MoviePlan by invoking the ADK ScriptAgent."""

    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")

    resolved_model = model_spec or os.environ.get("SCRIPT_AGENT_MODEL") or _DEFAULT_MODEL
    resolved_seed = seed if seed is not None else _maybe_int(os.environ.get("SCRIPT_AGENT_SEED"))
    resolved_run_id = run_id or os.environ.get("SPARKLE_RUN_ID") or observability.get_session_id()

    agent = adk_factory.get_agent("script_agent", model_spec=resolved_model, seed=resolved_seed)

    raw_result = _call_agent(agent, prompt)
    payload, raw_text = _coerce_plan_payload(raw_result)

    plan = _validate_movie_plan(payload)
    _enforce_resource_limits(plan)

    schema_artifact = schema_registry.movie_plan_schema()
    artifact_ref = _persist_plan_artifact(
        prompt=prompt,
        raw_output=raw_text,
        plan=plan,
        schema_uri=schema_artifact.uri,
        model_spec=resolved_model,
        seed=resolved_seed,
    )

    _record_plan_event(
        plan=plan,
        prompt=prompt,
        artifact_uri=artifact_ref["uri"],
        model_spec=resolved_model,
        schema_uri=schema_artifact.uri,
        run_id=resolved_run_id,
    )

    telemetry.emit_event(
        "script_agent.generate_plan.completed",
        {
            "model_spec": resolved_model,
            "schema_uri": schema_artifact.uri,
            "artifact_uri": artifact_ref["uri"],
            "shot_count": len(plan.shots),
        },
    )

    return plan


def _call_agent(agent: Any, prompt: str) -> Any:
    last_error: Exception | None = None
    callables = [
        getattr(agent, "generate_plan", None),
        getattr(agent, "generate", None),
        getattr(agent, "run", None),
        getattr(agent, "__call__", None),
    ]

    for fn in [c for c in callables if callable(c)]:
        for args, kwargs in _call_variants(prompt):
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:  # pragma: no cover - defensive
                raise AgentInvocationError("ScriptAgent invocation failed", cause=exc) from exc

    raise AgentInvocationError("ScriptAgent does not expose a callable interface", cause=last_error)


def _call_variants(prompt: str) -> Iterable[tuple[tuple[Any, ...], Mapping[str, Any]]]:
    payload = {"prompt": prompt}
    return (
        ((prompt,), {}),
        ((payload,), {}),
        ((), payload),
        ((), {"input": prompt}),
        ((), {"data": payload}),
    )


def _coerce_plan_payload(result: Any) -> tuple[dict[str, Any], str]:
    if isinstance(result, MoviePlan):
        data = result.model_dump()
        return data, json.dumps(data, ensure_ascii=False)

    if hasattr(result, "model_dump") and callable(result.model_dump):
        candidate = result.model_dump()
        if isinstance(candidate, Mapping):
            return dict(candidate), json.dumps(candidate, ensure_ascii=False)

    if hasattr(result, "dict") and callable(result.dict):  # pydantic v1 fallback
        candidate = result.dict()
        if isinstance(candidate, Mapping):
            return dict(candidate), json.dumps(candidate, ensure_ascii=False)

    if isinstance(result, Mapping):
        data = dict(result)
        return data, json.dumps(data, ensure_ascii=False)

    if isinstance(result, (list, tuple)) and result:
        first = next((item for item in result if item is not None), None)
        if isinstance(first, Mapping):
            data = dict(first)
            return data, json.dumps(data, ensure_ascii=False)

    text = _extract_text(result)
    json_payload = _parse_json_payload(text)
    if not isinstance(json_payload, Mapping):
        raise PlanSchemaError("MoviePlan payload must be an object")
    return dict(json_payload), text


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="ignore")
    for attr in ("text", "content"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
    return json.dumps(result, default=str, ensure_ascii=False)


def _parse_json_payload(text: str) -> Any:
    trimmed = text.strip()
    if not trimmed:
        raise PlanParseError("Agent returned empty response", raw_output=text)

    candidates = _json_candidates(trimmed)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise PlanParseError("Agent output is not valid JSON", raw_output=trimmed, cause=last_error)


def _json_candidates(text: str) -> Iterable[str]:
    yield text
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        yield text[start : end + 1]


def _validate_movie_plan(payload: Mapping[str, Any]) -> MoviePlan:
    try:
        if hasattr(MoviePlan, "model_validate"):
            return MoviePlan.model_validate(payload)  # type: ignore[attr-defined]
        return MoviePlan.parse_obj(payload)  # type: ignore[attr-defined]
    except ValidationError as exc:
        raise PlanSchemaError("MoviePlan validation failed", errors=exc.errors(), cause=exc) from exc


def _enforce_resource_limits(plan: MoviePlan) -> None:
    shot_count = len(plan.shots)
    if shot_count == 0:
        raise PlanResourceError("MoviePlan must contain at least one shot")

    max_shots = _maybe_int(os.environ.get("SCRIPT_AGENT_MAX_SHOTS")) or 20
    if shot_count > max_shots:
        raise PlanResourceError(f"MoviePlan exceeds shot limit: {shot_count} > {max_shots}")

    max_total_duration = _maybe_int(os.environ.get("SCRIPT_AGENT_MAX_DURATION_SEC")) or 600
    total_duration = sum(shot.duration_sec for shot in plan.shots)
    if total_duration > max_total_duration:
        raise PlanResourceError(f"MoviePlan exceeds duration cap: {total_duration} > {max_total_duration}")


def _persist_plan_artifact(
    *,
    prompt: str,
    raw_output: str,
    plan: MoviePlan,
    schema_uri: str,
    model_spec: str,
    seed: int | None,
) -> adk_helpers.ArtifactRef:
    payload = {
        "prompt": prompt,
        "raw_output": raw_output,
        "validated_plan": _plan_to_dict(plan),
        "schema_uri": schema_uri,
        "model_spec": model_spec,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    if seed is not None:
        payload["seed"] = seed

    tmp_path = _write_temp_payload(payload)
    metadata = {
        "tool": "script_agent",
        "schema_uri": schema_uri,
        "model_spec": model_spec,
        "seed": seed,
        "shot_count": len(plan.shots),
    }
    try:
        return adk_helpers.publish_artifact(
            local_path=tmp_path,
            artifact_type=_ARTIFACT_TYPE,
            metadata=metadata,
            media_type="application/json",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_temp_payload(payload: Mapping[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        return Path(handle.name)


def _plan_to_dict(plan: MoviePlan) -> dict[str, Any]:
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    return plan.dict()  # type: ignore[attr-defined]


def _record_plan_event(
    *,
    plan: MoviePlan,
    prompt: str,
    artifact_uri: str,
    model_spec: str,
    schema_uri: str,
    run_id: str,
) -> None:
    total_duration = sum(shot.duration_sec for shot in plan.shots)
    payload = {
        "artifact_uri": artifact_uri,
        "model_spec": model_spec,
        "schema_uri": schema_uri,
        "shot_count": len(plan.shots),
        "total_duration_sec": total_duration,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
    }
    try:
        adk_helpers.write_memory_event(
            run_id=run_id,
            event_type="script_agent.generate_plan",
            payload=payload,
        )
    except adk_helpers.MemoryWriteError as exc:  # type: ignore[attr-defined]
        _LOG.debug("memory event skipped: %s", exc)


def _maybe_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


__all__ = [
    "generate_plan",
    "ScriptAgentError",
    "AgentInvocationError",
    "PlanParseError",
    "PlanSchemaError",
    "PlanResourceError",
]
