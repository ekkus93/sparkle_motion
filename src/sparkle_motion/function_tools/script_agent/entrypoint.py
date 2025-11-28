from __future__ import annotations

from typing import Any, Literal, Optional
import re
import os
import json
import logging
import uuid
import time
import asyncio
from datetime import datetime, timezone
from threading import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion import adk_helpers
from sparkle_motion import adk_factory, observability, telemetry, script_agent, schema_registry
from sparkle_motion.schemas import MoviePlan

LOG = logging.getLogger("script_agent.entrypoint")
LOG.setLevel(logging.INFO)


class RequestModel(BaseModel):
    """Request schema for ScriptAgent with conservative validation.

    At least one of `prompt`, `title`, or `shots` must be provided.
    """
    title: Optional[str] = None
    shots: Optional[list[dict]] = None
    prompt: Optional[str] = None

    @model_validator(mode="after")
    def at_least_one(self):
        if not (self.prompt or self.title or self.shots):
            raise ValueError("empty request: provide prompt, title, or shots")
        return self


class ResponseModel(BaseModel):
    status: Literal["success", "error"]
    artifact_uri: Optional[str] = None
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

        # Eagerly construct per-tool agent (fixture mode returns dummy agent)
        try:
            model_spec = os.environ.get("SCRIPT_AGENT_MODEL", "script-agent-default")
            seed = int(os.environ.get("SCRIPT_AGENT_SEED")) if os.environ.get("SCRIPT_AGENT_SEED") else None
            app.state.agent = adk_factory.get_agent("script_agent", model_spec=model_spec, mode="per-tool", seed=seed)
            try:
                observability.record_seed(seed, tool_name="script_agent")
                telemetry.emit_event("agent.created", {"tool": "script_agent", "model_spec": model_spec, "seed": seed})
            except Exception:
                pass
        except Exception:
            LOG.exception("failed to construct ADK agent for script_agent")
            raise

        app.state._start_time = time.time()
        app.state.ready = True
        LOG.info("script_agent ready (agent attached)")
        try:
            send_telemetry("tool.ready", {"tool": "script_agent"})
        except Exception:
            LOG.exception("telemetry send failed on ready")
        try:
            telemetry.emit_event("tool.ready", {"tool": "script_agent"})
        except Exception:
            pass
        try:
            yield
        finally:
            # graceful shutdown: mark and wait for inflight ops
            app.state.shutting_down = True
            start = asyncio.get_event_loop().time()
            while app.state.inflight > 0 and (asyncio.get_event_loop().time() - start) < 5.0:
                await asyncio.sleep(0.05)
            try:
                send_telemetry("tool.shutdown", {"tool": "script_agent"})
            except Exception:
                LOG.exception("telemetry send failed on shutdown")

    app = FastAPI(title="script_agent Entrypoint", lifespan=lifespan)
    app.state.lock = Lock()
    # During fixture/test runs we prefer deterministic startup.
    # If `ADK_USE_FIXTURE=1` we mark ready immediately so tests don't
    # race on the lifespan startup timing.
    app.state.ready = os.environ.get("ADK_USE_FIXTURE", "1") != "0"
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
            raise HTTPException(status_code=503, detail="tool not ready")
        if getattr(app.state, "shutting_down", False):
            raise HTTPException(status_code=503, detail="shutting down")

        request_id = uuid.uuid4().hex
        LOG.info("invoke.received", extra={"request_id": request_id})
        try:
            send_telemetry("invoke.received", {"tool": "script_agent", "request_id": request_id})
        except Exception:
            LOG.exception("telemetry send failed on invoke.received")
        try:
            telemetry.emit_event("invoke.received", {"tool": "script_agent", "request_id": request_id})
        except Exception:
            pass
        with app.state.lock:
            app.state.inflight += 1
        try:
            plan = _generate_movie_plan(req, request_id)

            # prepare artifact persistence
            deterministic = os.environ.get("DETERMINISTIC", "1") == "1"
            artifacts_dir = os.environ.get("ARTIFACTS_DIR", os.path.join(os.getcwd(), "artifacts"))
            os.makedirs(artifacts_dir, exist_ok=True)

            # sanitize filename using either prompt or generated title
            content_for_name = plan.title or req.prompt or req.title or "artifact"

            def _slugify(s: str) -> str:
                s = s.strip()
                s = re.sub(r"\s+", "_", s)
                s = re.sub(r"[^A-Za-z0-9._-]", "", s)
                return s[:80] or "artifact"

            safe_name = _slugify(str(content_for_name))
            filename = f"{safe_name}.json" if deterministic else f"{safe_name}_{os.getpid()}_{request_id}.json"
            local_path = os.path.join(artifacts_dir, filename)

            payload = _build_artifact_payload(req, plan, request_id)

            try:
                with open(local_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
            except Exception as e:
                LOG.exception("failed to write artifact", extra={"request_id": request_id, "path": local_path, "error": str(e)})
                raise HTTPException(status_code=500, detail="failed to persist artifact")

            # publish via ADK helpers façade (SDK-first, CLI fallback).
            artifact_ref = adk_helpers.publish_artifact(
                local_path=local_path,
                artifact_type="script_agent_movie_plan",
                metadata={"request_id": request_id, "tool": "script_agent", "shot_count": len(plan.shots)},
            )
            artifact_uri = artifact_ref["uri"]
            try:
                send_telemetry("invoke.completed", {"tool": "script_agent", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                LOG.exception("telemetry send failed on invoke.completed")
            try:
                telemetry.emit_event("invoke.completed", {"tool": "script_agent", "request_id": request_id, "artifact_uri": artifact_uri})
            except Exception:
                pass

            resp = ResponseModel(status="success", artifact_uri=artifact_uri, request_id=request_id)
            return resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        finally:
            with app.state.lock:
                app.state.inflight = max(0, app.state.inflight - 1)

    return app


app = make_app()


def _build_artifact_payload(req: RequestModel, plan: MoviePlan, request_id: str) -> dict[str, Any]:
    request_payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    plan_payload = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
    payload: dict[str, Any] = {
        "request": request_payload,
        "validated_plan": plan_payload,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "tool": "script_agent",
            "request_id": request_id,
            "shot_count": len(plan.shots),
        },
    }
    try:
        payload["schema_uri"] = schema_registry.movie_plan_schema().uri
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOG.debug("schema lookup failed: %s", exc)
    return payload


def _generate_movie_plan(req: RequestModel, request_id: str) -> MoviePlan:
    if req.shots:
        try:
            return _plan_from_request_shots(req)
        except Exception as exc:
            LOG.warning("failed to coerce request shots; falling back to generation: %s", exc)

    prompt = _derive_prompt(req)
    try:
        plan = script_agent.generate_plan(prompt, run_id=request_id)
        plan.metadata.setdefault("source", "script_agent.generate_plan")
        if req.title:
            plan.metadata.setdefault("requested_title", req.title)
        return plan
    except Exception as exc:
        LOG.warning("script_agent.generate_plan failed; synthesizing inline: %s", exc)

    fallback_payload = _synthetic_plan_payload(prompt, req.title, request_id)
    return MoviePlan.model_validate(fallback_payload)


def _plan_from_request_shots(req: RequestModel) -> MoviePlan:
    shots: list[dict[str, Any]] = []
    for idx, raw in enumerate(req.shots or []):
        shots.append(_normalize_shot(idx, raw or {}, req))
    if not shots:
        raise ValueError("no shots provided")
    payload = {
        "title": req.title or _title_from_prompt(req.prompt, fallback="ScriptAgent Plan"),
        "shots": shots,
        "metadata": {"source": "script_agent.entrypoint.pass_through"},
    }
    return MoviePlan.model_validate(payload)


def _normalize_shot(idx: int, raw: dict[str, Any], req: RequestModel) -> dict[str, Any]:
    base_desc = _first_non_empty(
        raw.get("visual_description"),
        raw.get("description"),
        raw.get("desc"),
        f"Shot {idx + 1} for {req.title or _title_from_prompt(req.prompt)}",
    )
    start_prompt = _first_non_empty(
        raw.get("start_frame_prompt"),
        raw.get("start_prompt"),
        raw.get("prompt"),
        f"{base_desc} establishing frame",
    )
    end_prompt = _first_non_empty(
        raw.get("end_frame_prompt"),
        raw.get("end_prompt"),
        f"{base_desc} closing frame",
    )
    duration = _coerce_duration(raw.get("duration_sec") or raw.get("duration"), default=4.0 + idx)
    dialogue = _normalize_dialogue(raw.get("dialogue"), idx)
    return {
        "id": str(raw.get("id") or f"shot_{idx + 1:03d}"),
        "duration_sec": duration,
        "visual_description": base_desc,
        "start_frame_prompt": start_prompt,
        "end_frame_prompt": end_prompt,
        "motion_prompt": raw.get("motion_prompt") or "Deliberate cinematic move",
        "is_talking_closeup": bool(raw.get("is_talking_closeup") or raw.get("talking_closeup") or False),
        "dialogue": dialogue,
        "setting": raw.get("setting") or (req.title or "soundstage"),
    }


def _coerce_duration(value: Any, *, default: float) -> float:
    if value is None:
        return max(default, 1.0)
    if isinstance(value, (int, float)):
        return max(float(value), 1.0)
    cleaned = "".join(ch for ch in str(value) if ch.isdigit() or ch == ".")
    try:
        parsed = float(cleaned) if cleaned else default
    except ValueError:
        parsed = default
    return max(parsed, 1.0)


def _normalize_dialogue(raw_dialogue: Any, shot_idx: int) -> list[dict[str, Any]]:
    if not raw_dialogue:
        return []
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_dialogue):
        if entry is None:
            continue
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if isinstance(entry, str):
            normalized.append({"character_id": f"char_{shot_idx + 1:02d}", "text": entry})
            continue
        if isinstance(entry, dict):
            text = _first_non_empty(entry.get("text"), entry.get("line"), entry.get("dialogue"))
            if not text:
                continue
            char_id = _first_non_empty(entry.get("character_id"), entry.get("character"), entry.get("speaker"), f"char_{shot_idx + 1:02d}")
            payload: dict[str, Any] = {"character_id": str(char_id), "text": str(text)}
            if entry.get("start_time_sec") is not None:
                try:
                    payload["start_time_sec"] = float(entry["start_time_sec"])
                except (TypeError, ValueError):
                    pass
            normalized.append(payload)
    return normalized


def _derive_prompt(req: RequestModel) -> str:
    if req.prompt and req.prompt.strip():
        return req.prompt.strip()
    if req.title:
        return f"Write a cinematic short film titled '{req.title}'."
    if req.shots:
        desc = _first_non_empty(*(shot.get("visual_description") for shot in req.shots if isinstance(shot, dict)))
        if desc:
            return f"Create a plan around: {desc}"
    return "Write a concise, imaginative short film plan with vibrant visuals."


def _synthetic_plan_payload(prompt: str, title_hint: Optional[str], request_id: str) -> dict[str, Any]:
    base_title = title_hint or _title_from_prompt(prompt, fallback="Generated Short Film")
    descriptor = prompt or "cinematic short"
    shots: list[dict[str, Any]] = []
    motifs = (
        "establishing vista",
        "character turning point",
        "closing tableau",
    )
    for idx, motif in enumerate(motifs[:2]):
        shots.append(
            {
                "id": f"shot_{idx + 1:03d}",
                "duration_sec": 4.0 + idx,
                "visual_description": f"{motif.title()} inspired by {descriptor}",
                "start_frame_prompt": f"{descriptor} — {motif} opening",
                "end_frame_prompt": f"{descriptor} — {motif} closing",
                "motion_prompt": "Subtle dolly with volumetric light",
                "is_talking_closeup": idx == 1,
                "dialogue": [
                    {
                        "character_id": "narrator",
                        "text": f"Moment {idx + 1}: narrate the {motif} beat.",
                    }
                ],
            }
        )
    return {
        "title": base_title,
        "shots": shots,
        "metadata": {
            "source": "script_agent.entrypoint.synthetic",
            "request_id": request_id,
            "prompt_excerpt": descriptor[:120],
        },
    }


def _title_from_prompt(prompt: Optional[str], fallback: str = "Untitled Short") -> str:
    if not prompt:
        return fallback
    words = [w.strip(".,:;!?") for w in prompt.split() if w.strip()]
    if not words:
        return fallback
    snippet = " ".join(words[:6])
    return snippet.title() or fallback


def _first_non_empty(*candidates: Any) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return ""
