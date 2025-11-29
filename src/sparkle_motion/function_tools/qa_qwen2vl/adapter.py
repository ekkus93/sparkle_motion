from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from sparkle_motion import adk_helpers, gpu_utils, telemetry
from sparkle_motion.schemas import QAReport, QAReportPerShot

LOG = logging.getLogger("qa_qwen2vl.adapter")
LOG.setLevel(logging.INFO)

TRUTHY = {"1", "true", "yes", "on"}
BANNED_KEYWORDS = {"weapon", "blood", "violent", "nudity", "gore", "nsfw", "hate"}
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[3].parent
DEFAULT_POLICY_PATH = REPO_ROOT / "configs" / "qa_policy.yaml"
QWEN_RESPONSE_INSTRUCTION = (
    "You are a safety QA reviewer for animation workflows. Compare each frame to the provided prompt "
    "and respond ONLY with strict JSON using this schema: {\"prompt_match\": float 0-1, \"finger_issues\": bool, "
    "\"finger_issue_ratio\": float 0-1, \"safety_violation\": bool, \"missing_audio_detected\": bool, "
    "\"artifact_notes\": array of short strings, \"issues\": array of short strings}."
    " Do not include prose, markdown fences, or explanations outside the JSON object."
)


@dataclass(frozen=True)
class PolicyThresholds:
    prompt_match_min: float = 0.78
    max_finger_issue_ratio: float = 0.10
    allow_missing_audio: bool = False
    rerun_on_artifact_notes: bool = True
    min_audio_peak_db: float = -25.0


@dataclass(frozen=True)
class FrameAnalysis:
    index: int
    frame_id: str
    prompt: str
    prompt_match: float
    finger_issues: bool
    finger_issue_ratio: float
    safety_violation: bool
    missing_audio_detected: bool
    artifact_notes: list[str]


@dataclass(frozen=True)
class QAInspectionResult:
    report: QAReport
    artifact_path: Path
    artifact_uri: str
    metadata: dict[str, Any]
    decision: str
    human_task_id: str | None = None


class QAWiringError(RuntimeError):
    """Raised when real Qwen inference cannot be executed."""


@dataclass(frozen=True)
class EngineDecision:
    use_real: bool
    reason: str


def inspect_frames(
    frames: Sequence[bytes],
    prompts: Sequence[str],
    *,
    opts: Mapping[str, Any] | None = None,
) -> QAInspectionResult:
    if not frames:
        raise ValueError("frames must be non-empty")
    options = dict(opts or {})
    frame_ids = _normalize_frame_ids(options.pop("frame_ids", None), len(frames))
    options_snapshot = _snapshot_options(options)
    prompts_list = _normalize_prompts(prompts, len(frames), options.get("prompt"))
    policy = _load_policy(options.get("policy_path"))
    analyses, engine, engine_details, fallback_reason = _collect_analyses(
        frames,
        prompts_list,
        frame_ids,
        options,
    )
    report = _build_report(analyses, policy, options)
    metadata = _build_metadata(
        analyses,
        report,
        policy,
        options,
        engine,
        engine_details,
        fallback_reason,
        prompts_list,
        frame_ids,
        options_snapshot,
    )
    artifact_path = _persist_report(report, metadata)
    artifact = adk_helpers.publish_artifact(
        local_path=artifact_path,
        artifact_type="qa_report",
        media_type="application/json",
        metadata=metadata,
        run_id=options.get("run_id"),
    )
    artifact_uri = artifact["uri"]
    metadata["artifact_uri"] = artifact_uri
    _write_report_payload(artifact_path, report, metadata)
    _log_memory_event(report, metadata, artifact_uri, options)
    human_task_id = None
    if report.decision == "escalate":
        human_task_id = _request_human_review(report, artifact_uri, options)
    telemetry.emit_event(
        "qa_qwen2vl.inspect_frames.completed",
        {
            "decision": report.decision,
            "frames": len(analyses),
            "artifact_uri": artifact_uri,
            "fixture": metadata.get("engine") == "qwen_fixture",
            "fallback_reason": fallback_reason,
        },
    )
    return QAInspectionResult(
        report=report,
        artifact_path=artifact_path,
        artifact_uri=artifact_uri,
        metadata=metadata,
        decision=report.decision or "pending",
        human_task_id=human_task_id,
    )


def _normalize_prompts(prompts: Sequence[str], frame_count: int, default_prompt: Optional[str]) -> list[str]:
    if not prompts and not default_prompt:
        raise ValueError("prompts must be provided")
    if prompts and len(prompts) not in {1, frame_count}:
        raise ValueError("prompts length must be 1 or match frames length")
    if not prompts:
        prompts = [default_prompt or ""]
    if len(prompts) == 1 and frame_count > 1:
        prompts = prompts * frame_count
    return [p or "unspecified scene" for p in prompts]


def _normalize_frame_ids(frame_ids: Optional[Sequence[str]], frame_count: int) -> list[str]:
    if not frame_ids:
        return [f"frame_{idx:04d}" for idx in range(frame_count)]
    if len(frame_ids) != frame_count:
        raise ValueError("frame_ids length must match frames length")
    normalized: list[str] = []
    for idx, frame_id in enumerate(frame_ids):
        trimmed = (frame_id or "").strip()
        normalized.append(trimmed or f"frame_{idx:04d}")
    return normalized


def _collect_analyses(
    frames: Sequence[bytes],
    prompts: Sequence[str],
    frame_ids: Sequence[str],
    options: Mapping[str, Any],
) -> tuple[list[FrameAnalysis], str, dict[str, Any], Optional[str]]:
    decision = _decide_engine(options)
    fallback_reason: Optional[str] = None
    engine_details: dict[str, Any] = {}
    if decision.use_real:
        try:
            responses, engine_details = _run_qwen(frames, prompts, options, frame_ids)
            analyses = [
                _analysis_from_text(idx, frame_id, prompt, resp)
                for idx, (frame_id, prompt, resp) in enumerate(zip(frame_ids, prompts, responses))
            ]
            engine_details.setdefault("engine_reason", decision.reason)
            return analyses, "qwen2vl", engine_details, None
        except Exception as exc:  # pragma: no cover - real path optional
            fallback_reason = f"real_engine_error:{exc.__class__.__name__}"
            LOG.warning("Real Qwen inference failed; falling back to fixture", exc_info=True)
            telemetry.emit_event(
                "qa_qwen2vl.real_engine_failed",
                {"reason": fallback_reason, "frames": len(frames)},
            )

    seed = int(options.get("fixture_seed") or 0)
    rng = random.Random(seed)
    analyses: list[FrameAnalysis] = []
    for idx, (frame_id, prompt, data) in enumerate(zip(frame_ids, prompts, frames)):
        analyses.append(_fixture_analysis(idx, frame_id, prompt, data, rng))
    engine_details = {"fixture_seed": seed, "engine_reason": decision.reason}
    if fallback_reason is None:
        fallback_reason = decision.reason
    return analyses, "qwen_fixture", engine_details, fallback_reason


def _decide_engine(options: Mapping[str, Any]) -> EngineDecision:
    force_fixture = options.get("fixture_only")
    if isinstance(force_fixture, str) and force_fixture.strip().lower() in TRUTHY:
        return EngineDecision(False, "fixture_only_option")
    if force_fixture is True:
        return EngineDecision(False, "fixture_only_option")

    env = os.environ
    if env.get("ADK_USE_FIXTURE", "0").lower() in TRUTHY:
        return EngineDecision(False, "adk_fixture_env")
    if env.get("QA_QWEN2VL_FIXTURE_ONLY", "0").lower() in TRUTHY:
        return EngineDecision(False, "env_fixture_only")

    force_real = options.get("force_real_engine")
    if force_real is None:
        force_real = env.get("QA_QWEN2VL_FORCE_REAL")
    if _coerce_bool(force_real):
        return EngineDecision(True, "force_real_flag")

    if env.get("SMOKE_QA", "0").lower() in TRUTHY:
        return EngineDecision(True, "smoke_qa")

    return EngineDecision(False, "real_engine_disabled")


def _snapshot_options(options: Mapping[str, Any]) -> dict[str, Any]:
    whitelist = {
        "model_id",
        "max_new_tokens",
        "policy_path",
        "fixture_seed",
        "fixture_only",
        "force_real_engine",
        "dtype",
        "attention",
        "min_pixels",
        "max_pixels",
        "cache_ttl_s",
    }
    snapshot: dict[str, Any] = {}
    for key in whitelist:
        if key in options and options[key] is not None:
            snapshot[key] = options[key]
    return snapshot


def _safe_frame_filename(frame_id: str, idx: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", frame_id).strip("._")
    if not slug:
        slug = f"frame_{idx:04d}"
    return slug[:48]


def _fixture_analysis(idx: int, frame_id: str, prompt: str, data: bytes, rng: random.Random) -> FrameAnalysis:
    digest = hashlib.sha256(data + prompt.encode("utf-8") + idx.to_bytes(2, "big")).hexdigest()
    prompt_match = ((int(digest[:4], 16) % 1000) / 1000.0)
    finger_ratio = ((int(digest[4:8], 16) % 100) / 100.0)
    safety_violation = any(keyword in prompt.lower() for keyword in BANNED_KEYWORDS)
    safety_violation = safety_violation or ((int(digest[8:10], 16) % 37) == 0)
    artifact_notes: list[str] = []
    if (int(digest[10:12], 16) % 5) == 0:
        artifact_notes.append("noticeable artifact near subject boundary")
    missing_audio = False
    finger_issues = finger_ratio > 0.5
    return FrameAnalysis(
        index=idx,
        frame_id=frame_id,
        prompt=prompt,
        prompt_match=round(prompt_match, 4),
        finger_issues=finger_issues,
        finger_issue_ratio=round(min(finger_ratio, 1.0), 3),
        safety_violation=safety_violation,
        missing_audio_detected=missing_audio,
        artifact_notes=artifact_notes,
    )


def _analysis_from_text(idx: int, frame_id: str, prompt: str, response: str) -> FrameAnalysis:
    structured = _parse_structured_response(response)
    if structured:
        return _analysis_from_structured(idx, frame_id, prompt, structured)
    text = response.lower()
    prompt_match = 0.9 if "matches" in text else 0.7
    if "deviates" in text or "mismatch" in text:
        prompt_match = 0.4
    finger_issues = "finger" in text or "hand" in text and "issue" in text
    finger_ratio = 0.2 if finger_issues else 0.05
    safety_violation = any(keyword in text for keyword in BANNED_KEYWORDS)
    artifact_notes: list[str] = []
    if "artifact" in text or "glitch" in text:
        artifact_notes.append("model reported visual artifact")
    missing_audio = "audio" in text and "missing" in text
    return FrameAnalysis(
        index=idx,
        frame_id=frame_id,
        prompt=prompt,
        prompt_match=round(prompt_match, 4),
        finger_issues=finger_issues,
        finger_issue_ratio=round(min(finger_ratio, 1.0), 3),
        safety_violation=safety_violation,
        missing_audio_detected=missing_audio,
        artifact_notes=artifact_notes,
    )


def _analysis_from_structured(idx: int, frame_id: str, prompt: str, payload: Mapping[str, Any]) -> FrameAnalysis:
    prompt_match = _clamp_float(payload.get("prompt_match") or payload.get("prompt_match_score"), default=0.75)
    finger_ratio = _clamp_float(payload.get("finger_issue_ratio"), default=0.1)
    finger_issues = _coerce_bool(payload.get("finger_issues"))
    safety_violation = _coerce_bool(payload.get("safety_violation"))
    missing_audio = _coerce_bool(payload.get("missing_audio_detected") or payload.get("missing_audio"))
    artifact_notes = _string_list(payload.get("artifact_notes"))
    if not artifact_notes:
        artifact_notes = _string_list(payload.get("issues"))
    if not artifact_notes and _coerce_bool(payload.get("artifact_flag")):
        artifact_notes = ["model flagged a potential artifact"]
    return FrameAnalysis(
        index=idx,
        frame_id=frame_id,
        prompt=prompt,
        prompt_match=prompt_match,
        finger_issues=finger_issues or finger_ratio > 0.5,
        finger_issue_ratio=finger_ratio,
        safety_violation=safety_violation,
        missing_audio_detected=missing_audio,
        artifact_notes=artifact_notes,
    )


def _parse_structured_response(response: str) -> Mapping[str, Any] | None:
    if not response:
        return None
    text = response.strip()
    if not text:
        return None
    if "```" in text:
        segments = [seg.strip() for seg in re.split(r"```(?:json)?", text) if seg.strip()]
        if segments:
            text = segments[-1]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, Mapping) else None


def _clamp_float(value: Any, *, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return round(default, 4)
    return round(max(lo, min(hi, as_float)), 4)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    result.append(trimmed)
        return result
    return []


def _run_qwen(
    frames: Sequence[bytes],
    prompts: Sequence[str],
    options: Mapping[str, Any],
    frame_ids: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:  # pragma: no cover - heavy path
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except Exception as exc:  # pragma: no cover - import guard
        raise QAWiringError("transformers[qwen2-vl] and qwen-vl-utils are required for real inference") from exc

    model_id = options.get("model_id") or os.environ.get("QA_QWEN2VL_MODEL", DEFAULT_MODEL_ID)
    dtype_name = options.get("dtype") or os.environ.get("QA_QWEN2VL_DTYPE", "auto")
    attn_impl = options.get("attention") or os.environ.get("QA_QWEN2VL_ATTN", "flash_attention_2")
    max_new_tokens = int(options.get("max_new_tokens") or os.environ.get("QA_QWEN2VL_MAX_TOKENS", 160))
    min_pixels = int(options.get("min_pixels") or os.environ.get("QA_QWEN2VL_PIXELS_MIN", 256))
    max_pixels = int(options.get("max_pixels") or os.environ.get("QA_QWEN2VL_PIXELS_MAX", 1024))

    def _loader():
        torch_dtype = getattr(torch, dtype_name, torch.bfloat16 if dtype_name == "auto" else torch.float16)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype if dtype_name != "auto" else "auto",
            device_map="auto",
            attn_implementation=attn_impl,
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return _QwenRuntime(model=model, processor=processor)

    cache_ttl = float(options.get("cache_ttl_s") or os.environ.get("QA_QWEN2VL_CACHE_TTL", 900))
    responses: list[str] = []
    engine_details = {
        "model_id": model_id,
        "dtype": dtype_name,
        "attention": attn_impl,
        "max_new_tokens": max_new_tokens,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "cache_ttl_s": cache_ttl,
    }

    with gpu_utils.model_context(
        f"qa_qwen2vl::{model_id}",
        loader=_loader,
        weights=model_id,
        keep_warm=True,
        warm_ttl_s=cache_ttl,
    ) as ctx:
        runtime: _QwenRuntime = ctx.pipeline
        with tempfile.TemporaryDirectory(prefix="qa_qwen2vl_") as tmp:
            for idx, (frame_bytes, prompt, frame_id) in enumerate(zip(frames, prompts, frame_ids)):
                safe_name = _safe_frame_filename(frame_id, idx)
                frame_path = Path(tmp) / f"{safe_name}.png"
                frame_path.write_bytes(frame_bytes)
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_path.as_uri()},
                            {
                                "type": "text",
                                "text": (
                                    f"{QWEN_RESPONSE_INSTRUCTION}\nPrompt: {prompt}\n"
                                    "Return JSON immediately with no additional narrative."
                                ),
                            },
                        ],
                    }
                ]
                text = runtime.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(message)
                inputs = runtime.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                )
                inputs = {k: v.to(runtime.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                output = runtime.model.generate(**inputs, max_new_tokens=max_new_tokens)
                trimmed = output[:, inputs["input_ids"].shape[-1] :]
                decoded = runtime.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                responses.append(decoded[0])
    return responses, engine_details


@dataclass
class _QwenRuntime:
    model: Any
    processor: Any

    def close(self) -> None:  # pragma: no cover - cleanup
        try:
            if hasattr(self.model, "cpu"):
                self.model.cpu()
        except Exception:
            pass


def _build_report(analyses: Sequence[FrameAnalysis], policy: PolicyThresholds, options: Mapping[str, Any]) -> QAReport:
    per_shot: list[QAReportPerShot] = []
    issues = 0
    aggregate_prompt_match = 0.0
    decision = "approve"
    for analysis in analyses:
        aggregate_prompt_match += analysis.prompt_match
        shot = QAReportPerShot(
            shot_id=analysis.frame_id,
            prompt_match=analysis.prompt_match,
            finger_issues=analysis.finger_issues,
            finger_issue_ratio=analysis.finger_issue_ratio,
            artifact_notes=list(analysis.artifact_notes),
            missing_audio_detected=analysis.missing_audio_detected,
            safety_violation=analysis.safety_violation,
        )
        per_shot.append(shot)
        if analysis.safety_violation:
            decision = "escalate"
            issues += 1
        elif analysis.prompt_match < policy.prompt_match_min or analysis.artifact_notes:
            issues += 1
            if decision != "escalate":
                decision = "regenerate"
    aggregate_prompt_match = round(aggregate_prompt_match / max(len(analyses), 1), 4)
    summary = _summarize(decision, issues, len(analyses))
    return QAReport(
        movie_title=options.get("movie_title"),
        per_shot=per_shot,
        summary=summary,
        decision=decision,
        issues_found=issues,
        aggregate_prompt_match=aggregate_prompt_match,
    )


def _summarize(decision: str, issues: int, total: int) -> str:
    if issues == 0:
        return f"All {total} frames approved"
    if decision == "escalate":
        return f"Escalation required on {issues} of {total} frames"
    return f"Regenerate suggested for {issues} of {total} frames"


def _build_metadata(
    analyses: Sequence[FrameAnalysis],
    report: QAReport,
    policy: PolicyThresholds,
    options: Mapping[str, Any],
    engine: str,
    engine_details: Mapping[str, Any],
    fallback_reason: Optional[str],
    prompts: Sequence[str],
    frame_ids: Sequence[str],
    options_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_list = list(prompts)
    metadata: dict[str, Any] = {
        "engine": engine,
        "engine_details": dict(engine_details),
        "plan_id": options.get("plan_id"),
        "run_id": options.get("run_id"),
        "step_id": options.get("step_id"),
        "frames": len(analyses),
        "decision": report.decision,
        "issues_found": report.issues_found,
        "options_snapshot": dict(options_snapshot),
        "frame_ids": list(frame_ids),
        "prompt_preview": prompt_list[: min(len(prompt_list), 3)],
        "artifact_uri": None,
        "policy": {
            "prompt_match_min": policy.prompt_match_min,
            "max_finger_issue_ratio": policy.max_finger_issue_ratio,
            "allow_missing_audio": policy.allow_missing_audio,
            "min_audio_peak_db": policy.min_audio_peak_db,
        },
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason
    frame_meta: list[dict[str, Any]] = []
    for analysis in analyses:
        frame_meta.append(
            {
                "frame_id": analysis.frame_id,
                "prompt": analysis.prompt,
                "prompt_match": analysis.prompt_match,
                "finger_issue_ratio": analysis.finger_issue_ratio,
                "artifact_notes": analysis.artifact_notes,
                "safety_violation": analysis.safety_violation,
            }
        )
    metadata["frames_detail"] = frame_meta
    extra_meta = options.get("metadata")
    if isinstance(extra_meta, Mapping):
        metadata.update({k: v for k, v in extra_meta.items() if k not in metadata})
    return metadata


def _persist_report(report: QAReport, metadata: Mapping[str, Any]) -> Path:
    target_dir = _artifact_dir()
    filename = f"qa_report_{int(time.time())}_{hashlib.sha1(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest()[:8]}.json"
    path = target_dir / filename
    _write_report_payload(path, report, metadata)
    return path


def _write_report_payload(path: Path, report: QAReport, metadata: Mapping[str, Any]) -> None:
    payload = {
        "report": report.model_dump(mode="json"),
        "metadata": metadata,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _artifact_dir() -> Path:
    base = Path(os.environ.get("ARTIFACTS_DIR", Path.cwd() / "artifacts"))
    target = base / "qa_qwen2vl"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _log_memory_event(report: QAReport, metadata: Mapping[str, Any], artifact_uri: str, options: Mapping[str, Any]) -> None:
    payload = {
        "decision": report.decision,
        "issues_found": report.issues_found,
        "artifact_uri": artifact_uri,
        "metadata": metadata,
    }
    try:
        adk_helpers.write_memory_event(run_id=options.get("run_id"), event_type="qa_qwen2vl.report", payload=payload)
    except adk_helpers.MemoryWriteError:
        LOG.debug("Memory event write failed", exc_info=True)


def _request_human_review(report: QAReport, artifact_uri: str, options: Mapping[str, Any]) -> str:
    reason = report.summary or "QA escalation requested"
    run_id = options.get("run_id")
    metadata = {
        "step_id": options.get("step_id"),
        "plan_id": options.get("plan_id"),
        "decision": report.decision,
    }
    return adk_helpers.request_human_input(run_id=run_id, reason=reason, artifact_uri=artifact_uri, metadata=metadata)


def _load_policy(path_override: Any) -> PolicyThresholds:
    path = Path(path_override) if path_override else DEFAULT_POLICY_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"QA policy file not found: {path}") from exc
    thresholds = data.get("thresholds", {}) if isinstance(data, Mapping) else {}
    return PolicyThresholds(
        prompt_match_min=float(thresholds.get("prompt_match_min", 0.78)),
        max_finger_issue_ratio=float(thresholds.get("max_finger_issue_ratio", 0.10)),
        allow_missing_audio=bool(thresholds.get("allow_missing_audio", False)),
        rerun_on_artifact_notes=bool(thresholds.get("rerun_on_artifact_notes", True)),
        min_audio_peak_db=float(thresholds.get("min_audio_peak_db", -25.0)),
    )
