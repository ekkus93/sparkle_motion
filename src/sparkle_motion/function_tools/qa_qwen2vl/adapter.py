from __future__ import annotations

import hashlib
import json
import logging
import os
import random
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


def inspect_frames(
    frames: Sequence[bytes],
    prompts: Sequence[str],
    *,
    opts: Mapping[str, Any] | None = None,
) -> QAInspectionResult:
    if not frames:
        raise ValueError("frames must be non-empty")
    options = dict(opts or {})
    prompts_list = _normalize_prompts(prompts, len(frames), options.get("prompt"))
    policy = _load_policy(options.get("policy_path"))
    analyses, engine = _analyze_frames(frames, prompts_list, policy, options)
    report = _build_report(analyses, policy, options)
    metadata = _build_metadata(analyses, report, policy, options, engine)
    artifact_path = _persist_report(report, metadata)
    artifact = adk_helpers.publish_artifact(
        local_path=artifact_path,
        artifact_type="qa_report",
        media_type="application/json",
        metadata=metadata,
        run_id=options.get("run_id"),
    )
    artifact_uri = artifact["uri"]
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


def _analyze_frames(
    frames: Sequence[bytes],
    prompts: Sequence[str],
    policy: PolicyThresholds,
    options: Mapping[str, Any],
) -> tuple[list[FrameAnalysis], str]:
    use_real = _should_use_real_engine(options.get("fixture_only"))
    if use_real:
        try:
            responses = _run_qwen(frames, prompts, options)
            analyses = [_analysis_from_text(idx, prompt, resp) for idx, (prompt, resp) in enumerate(zip(prompts, responses))]
            return analyses, "qwen2vl"
        except Exception as exc:  # pragma: no cover - real path optional
            LOG.warning("Real Qwen inference failed; falling back to fixture: %s", exc)
    seed = int(options.get("fixture_seed") or 0)
    rng = random.Random(seed)
    analyses: list[FrameAnalysis] = []
    for idx, (prompt, data) in enumerate(zip(prompts, frames)):
        analyses.append(_fixture_analysis(idx, prompt, data, rng))
    return analyses, "qwen_fixture"


def _should_use_real_engine(force_fixture: Any) -> bool:
    if isinstance(force_fixture, str):
        if force_fixture.strip().lower() in TRUTHY:
            return False
    elif force_fixture is True:
        return False
    env = os.environ
    if env.get("ADK_USE_FIXTURE", "0").lower() in TRUTHY:
        return False
    if env.get("QA_QWEN2VL_FIXTURE_ONLY", "0").lower() in TRUTHY:
        return False
    flags = ("SMOKE_QA", "SMOKE_ADAPTERS", "SMOKE_ADK")
    return any(env.get(flag, "0").lower() in TRUTHY for flag in flags)


def _fixture_analysis(idx: int, prompt: str, data: bytes, rng: random.Random) -> FrameAnalysis:
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
    frame_id = f"frame_{idx:04d}"
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


def _analysis_from_text(idx: int, prompt: str, response: str) -> FrameAnalysis:
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
        frame_id=f"frame_{idx:04d}",
        prompt=prompt,
        prompt_match=round(prompt_match, 4),
        finger_issues=finger_issues,
        finger_issue_ratio=round(min(finger_ratio, 1.0), 3),
        safety_violation=safety_violation,
        missing_audio_detected=missing_audio,
        artifact_notes=artifact_notes,
    )


def _run_qwen(frames: Sequence[bytes], prompts: Sequence[str], options: Mapping[str, Any]) -> list[str]:  # pragma: no cover - heavy path
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except Exception as exc:  # pragma: no cover - import guard
        raise QAWiringError("transformers[qwen2-vl] and qwen-vl-utils are required for real inference") from exc

    model_id = options.get("model_id") or os.environ.get("QA_QWEN2VL_MODEL", DEFAULT_MODEL_ID)
    dtype_name = os.environ.get("QA_QWEN2VL_DTYPE", "auto")
    attn_impl = os.environ.get("QA_QWEN2VL_ATTN", "flash_attention_2")
    max_new_tokens = int(options.get("max_new_tokens") or os.environ.get("QA_QWEN2VL_MAX_TOKENS", 160))
    min_pixels = int(os.environ.get("QA_QWEN2VL_PIXELS_MIN", 256))
    max_pixels = int(os.environ.get("QA_QWEN2VL_PIXELS_MAX", 1024))

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

    cache_ttl = float(os.environ.get("QA_QWEN2VL_CACHE_TTL", 900))
    responses: list[str] = []

    with gpu_utils.model_context(
        f"qa_qwen2vl::{model_id}",
        loader=_loader,
        weights=model_id,
        keep_warm=True,
        warm_ttl_s=cache_ttl,
    ) as ctx:
        runtime: _QwenRuntime = ctx.pipeline
        with tempfile.TemporaryDirectory(prefix="qa_qwen2vl_") as tmp:
            for idx, (frame_bytes, prompt) in enumerate(zip(frames, prompts)):
                frame_path = Path(tmp) / f"frame_{idx:04d}.png"
                frame_path.write_bytes(frame_bytes)
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_path.as_uri()},
                            {
                                "type": "text",
                                "text": (
                                    "You are a safety QA reviewer. Given the provided prompt and frame, "
                                    "answer whether the frame matches the prompt, any safety issues, and"
                                    " mention artifacts if present. Prompt: "
                                    f"{prompt}"
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
    return responses


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
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "engine": engine,
        "plan_id": options.get("plan_id"),
        "run_id": options.get("run_id"),
        "step_id": options.get("step_id"),
        "frames": len(analyses),
        "decision": report.decision,
        "issues_found": report.issues_found,
        "policy": {
            "prompt_match_min": policy.prompt_match_min,
            "max_finger_issue_ratio": policy.max_finger_issue_ratio,
            "allow_missing_audio": policy.allow_missing_audio,
        },
    }
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
    payload = {
        "report": report.model_dump(mode="json"),
        "metadata": metadata,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


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
