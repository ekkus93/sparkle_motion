from __future__ import annotations

import re
import os
import json
import logging
import uuid
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from contextlib import asynccontextmanager
from typing import Any, Optional, Iterable, Tuple, Dict

from fastapi import FastAPI, HTTPException

from sparkle_motion.function_tools.entrypoint_common import send_telemetry
from sparkle_motion import adk_helpers
from sparkle_motion import adk_factory, observability, telemetry, script_agent, schema_registry
from sparkle_motion.utils.env import fixture_mode_enabled
from sparkle_motion.schemas import MoviePlan
from sparkle_motion.function_tools.script_agent.models import ScriptAgentRequest, ScriptAgentResponse

LOG = logging.getLogger("script_agent.entrypoint")
LOG.setLevel(logging.INFO)


RequestModel = ScriptAgentRequest
ResponseModel = ScriptAgentResponse


def _fixture_ready_override() -> bool:
    return fixture_mode_enabled() or os.environ.get("DETERMINISTIC", "0") == "1"


def make_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.shutting_down = False
        app.state.inflight = 0
        app.state.lock = Lock()
        if not fixture_mode_enabled():
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
        ready_flag = bool(getattr(app.state, "ready", False))
        shutting_down = bool(getattr(app.state, "shutting_down", False))
        if _fixture_ready_override():
            ready_flag = True
            shutting_down = False
        return {"ready": ready_flag, "shutting_down": shutting_down}

    @app.post("/invoke")
    def invoke(req: RequestModel) -> dict[str, Any]:
        if _fixture_ready_override():
            app.state.ready = True
            app.state.shutting_down = False

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
            plan = _canonicalize_plan_base_images(plan)

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

            try:
                schema_uri = schema_registry.movie_plan_schema().uri
            except Exception:
                schema_uri = None

            resp = ResponseModel(status="success", artifact_uri=artifact_uri, request_id=request_id, schema_uri=schema_uri)
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


def _canonicalize_plan_base_images(plan: MoviePlan) -> MoviePlan:
    """Return a copy of ``plan`` with sequential base image ids and prompts."""

    shots = plan.shots
    if not shots:
        return plan

    plan_dict = plan.model_dump()
    base_images = plan_dict.get("base_images") or []
    base_lookup: Dict[str, Dict[str, Any]] = {}
    for entry in base_images:
        if isinstance(entry, dict) and entry.get("id"):
            base_lookup[str(entry["id"])] = entry

    new_base_images: list[dict[str, Any]] = []
    updated_shots: list[dict[str, Any]] = plan_dict.get("shots", [])
    if not isinstance(updated_shots, list):
        return plan

    for idx, shot in enumerate(updated_shots):
        if not isinstance(shot, dict):
            continue
        original_start_id = shot.get("start_base_image_id")
        original_end_id = shot.get("end_base_image_id")
        shot_desc = shot.get("visual_description") or f"Shot {idx + 1}"

        start_prompt = _lookup_base_prompt(base_lookup, original_start_id, fallback=f"{shot_desc}: establishing")
        end_prompt = _lookup_base_prompt(base_lookup, original_end_id, fallback=f"{shot_desc}: closing")

        if idx == 0:
            new_base_images.append(_canonical_base_image(0, start_prompt, source_entry=base_lookup.get(str(original_start_id))))
        shot["start_base_image_id"] = _frame_id(idx)

        terminal_index = idx + 1
        shot["end_base_image_id"] = _frame_id(terminal_index)
        new_base_images.append(_canonical_base_image(terminal_index, end_prompt, source_entry=base_lookup.get(str(original_end_id))))

    plan_dict["base_images"] = new_base_images
    plan_dict["shots"] = updated_shots
    return MoviePlan.model_validate(plan_dict)


def _lookup_base_prompt(base_lookup: Dict[str, Dict[str, Any]], base_id: Optional[str], *, fallback: str) -> str:
    if base_id and base_id in base_lookup:
        prompt = base_lookup[base_id].get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        meta = base_lookup[base_id].get("metadata")
        if isinstance(meta, dict):
            meta_prompt = meta.get("prompt")
            if isinstance(meta_prompt, str) and meta_prompt.strip():
                return meta_prompt.strip()
    return fallback


def _canonical_base_image(index: int, prompt: str, *, source_entry: Optional[Dict[str, Any]]) -> dict[str, Any]:
    base = _base_image(index, prompt)
    metadata = dict(base.get("metadata") or {})
    metadata.setdefault("ordinal", index)
    if source_entry:
        metadata.setdefault("source_base_image_id", source_entry.get("id"))
        if source_entry.get("metadata"):
            metadata.setdefault("source_metadata", source_entry.get("metadata"))
    base["metadata"] = metadata
    return base


def _plan_from_request_shots(req: RequestModel) -> MoviePlan:
    raw_shots = [dict(raw or {}) for raw in (req.shots or [])]
    if not raw_shots:
        raise ValueError("no shots provided")
    descriptor = _derive_prompt(req)
    shots, base_images, timeline = _build_plan_components(raw_shots, descriptor=descriptor, req=req)
    payload = {
        "title": req.title or _title_from_prompt(req.prompt, fallback="ScriptAgent Plan"),
        "shots": shots,
        "base_images": base_images,
        "dialogue_timeline": timeline,
        "render_profile": _default_render_profile(),
        "metadata": {"source": "script_agent.entrypoint.pass_through"},
    }
    return MoviePlan.model_validate(payload)


@dataclass
class _NormalizedShot:
    data: dict[str, Any]
    start_prompt: str
    end_prompt: str
    dialogue: list[dict[str, Any]]


def _normalize_shot(idx: int, raw: dict[str, Any], req: Optional[RequestModel], descriptor: str) -> _NormalizedShot:
    title_hint = req.title if req else None
    base_desc = _first_non_empty(
        raw.get("visual_description"),
        raw.get("description"),
        raw.get("desc"),
        f"Shot {idx + 1} for {title_hint or _title_from_prompt(descriptor)}",
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
    shot = {
        "id": str(raw.get("id") or f"shot_{idx + 1:03d}"),
        "duration_sec": duration,
        "visual_description": base_desc,
        "motion_prompt": raw.get("motion_prompt") or "Deliberate cinematic move",
        "is_talking_closeup": bool(raw.get("is_talking_closeup") or raw.get("talking_closeup") or False),
        "setting": raw.get("setting") or (title_hint or "soundstage"),
    }
    return _NormalizedShot(shot, start_prompt, end_prompt, dialogue)


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
            if entry.get("duration_sec") is not None:
                try:
                    payload["duration_sec"] = max(float(entry["duration_sec"]), 0.05)
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
    motifs = (
        "establishing vista",
        "character turning point",
        "closing tableau",
    )
    raw_shots: list[dict[str, Any]] = []
    for idx, motif in enumerate(motifs[:2]):
        raw_shots.append(
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

    shots, base_images, timeline = _build_plan_components(raw_shots, descriptor=descriptor)
    return {
        "title": base_title,
        "shots": shots,
        "base_images": base_images,
        "dialogue_timeline": timeline,
        "render_profile": _default_render_profile(),
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


def _build_plan_components(
    raw_shots: Iterable[dict[str, Any]],
    *,
    descriptor: str,
    req: Optional[RequestModel] = None,
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[_NormalizedShot] = [
        _normalize_shot(idx, raw or {}, req, descriptor)
        for idx, raw in enumerate(raw_shots)
    ]
    if not normalized:
        raise ValueError("no shots provided")

    base_images: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    cursor = 0.0
    shots: list[dict[str, Any]] = []
    for idx, payload in enumerate(normalized):
        shot = payload.data
        if idx == 0:
            base_images.append(_base_image(idx, payload.start_prompt or shot["visual_description"]))
        start_id = base_images[idx]["id"]
        end_id = _frame_id(idx + 1)
        base_images.append(_base_image(idx + 1, payload.end_prompt or shot["visual_description"]))
        shot["start_base_image_id"] = start_id
        shot["end_base_image_id"] = end_id
        shots.append(shot)
        entries, cursor = _timeline_entries_from_dialogue(payload.dialogue, cursor, shot["duration_sec"])
        timeline.extend(entries)
    return shots, base_images, timeline


def _frame_id(index: int) -> str:
    return f"frame_{index:03d}"


def _base_image(index: int, prompt: str) -> dict[str, Any]:
    safe_prompt = prompt or f"Cinematic frame {index:03d}"
    return {"id": _frame_id(index), "prompt": safe_prompt, "metadata": {"ordinal": index}}


def _timeline_entries_from_dialogue(
    dialogue: list[dict[str, Any]],
    start_time: float,
    duration: float,
) -> Tuple[list[dict[str, Any]], float]:
    entries: list[dict[str, Any]] = []
    shot_start = start_time
    shot_end = shot_start + duration
    cursor = shot_start
    if dialogue:
        for idx, line in enumerate(dialogue):
            remaining_lines = len(dialogue) - idx
            remaining_time = max(shot_end - cursor, 0.0)
            if remaining_lines <= 0:
                break
            chunk = remaining_time / remaining_lines if remaining_lines else 0.0
            chunk = max(chunk, 0.01)
            requested_duration = line.get("duration_sec")
            if requested_duration is not None:
                try:
                    chunk = max(min(float(requested_duration), remaining_time), 0.01)
                except (TypeError, ValueError):
                    chunk = max(chunk, 0.01)
            text = line.get("text", "").strip()
            if not text:
                cursor += chunk
                continue
            entries.append(
                {
                    "type": "dialogue",
                    "character_id": line.get("character_id") or f"char_{idx + 1:02d}",
                    "text": text,
                    "start_time_sec": cursor,
                    "duration_sec": chunk,
                }
            )
            cursor += chunk
    if cursor < shot_end:
        entries.append(_silence_entry(cursor, shot_end - cursor))
        cursor = shot_end
    if not entries:
        entries.append(_silence_entry(shot_start, duration))
        cursor = shot_end
    return entries, cursor


def _silence_entry(start: float, duration: float) -> dict[str, Any]:
    return {
        "type": "silence",
        "start_time_sec": start,
        "duration_sec": max(duration, 0.01),
    }


def _default_render_profile() -> dict[str, Any]:
    model_id = os.environ.get("SCRIPT_AGENT_VIDEO_MODEL", "wan-2.1")
    max_fps = os.environ.get("SCRIPT_AGENT_VIDEO_MAX_FPS")
    profile: dict[str, Any] = {"video": {"model_id": model_id}}
    if max_fps:
        try:
            profile["video"]["max_fps"] = float(max_fps)
        except ValueError:
            pass
    profile.setdefault("metadata", {"source": "script_agent.entrypoint"})
    return profile
