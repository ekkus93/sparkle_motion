"""QA adapter stub that emits structured QA reports.

The production adapter should call a VLM (e.g., Qwen2-VL-7B) to score frames and
audio. Until then we create deterministic results based on available assets so
the orchestrator can exercise the QA gating workflow.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List


def _ensure_shots(asset_refs: Dict[str, Any]) -> Dict[str, Any]:
    shots = asset_refs.get("shots") or {}
    if not isinstance(shots, dict):
        raise ValueError("asset_refs['shots'] must be a dict")
    return shots


def _score_shot(shot_id: str, shot_refs: Dict[str, Any]) -> Dict[str, Any]:
    has_final = bool(shot_refs.get("final_video_clip"))
    has_audio = bool(shot_refs.get("dialogue_audio"))
    prompt_match = 0.92 if has_final else 0.55
    missing_audio = not has_audio and bool(shot_refs.get("dialogue_required", True))
    artifact_notes: List[str] = []
    if not has_final:
        artifact_notes.append("final_video_missing")
    if missing_audio:
        artifact_notes.append("audio_missing")
    return {
        "shot_id": shot_id,
        "prompt_match": round(prompt_match, 3),
        "finger_issues": False,
        "artifact_notes": artifact_notes,
        "missing_audio_detected": missing_audio,
        "safety_violation": False,
    }


def _decision(per_shot: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_scores = [entry["prompt_match"] for entry in per_shot]
    avg_prompt = statistics.mean(prompt_scores) if prompt_scores else 0.0
    has_blocker = any(entry["artifact_notes"] for entry in per_shot)
    decision = "approve"
    reasons: List[str] = []
    if has_blocker or avg_prompt < 0.75:
        decision = "regenerate"
        reasons.append("prompt_match_low" if avg_prompt < 0.75 else "artifact_detected")
    return {
        "decision": decision,
        "issues_found": sum(len(entry["artifact_notes"]) for entry in per_shot),
        "average_prompt_match": round(avg_prompt, 3),
        "reasons": reasons,
    }


def run_qa(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Build a QA report derived from the current asset references."""

    shots_refs = _ensure_shots(asset_refs)
    per_shot_results: List[Dict[str, Any]] = []
    for shot_id, shot_refs in shots_refs.items():
        if not isinstance(shot_refs, dict):
            continue
        per_shot_results.append(_score_shot(shot_id, shot_refs))

    summary = _decision(per_shot_results)
    return {
        "movie_title": movie_plan.get("title", "Untitled"),
        "decision": summary["decision"],
        "issues_found": summary["issues_found"],
        "average_prompt_match": summary["average_prompt_match"],
        "reasons": summary["reasons"],
        "per_shot": per_shot_results,
        "qa_version": "stub-1",
    }
