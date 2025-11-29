from __future__ import annotations
from typing import Any, Dict, List, Mapping

from sparkle_motion.function_tools.qa_qwen2vl import adapter


def inspect_frames(frames: List[bytes], prompts: List[str], opts: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Compatibility wrapper that delegates to the shared adapter implementation."""

    result = adapter.inspect_frames(frames, prompts, opts=opts)
    payload: Dict[str, Any] = {
        "status": result.decision,
        "decision": result.decision,
        "artifact_uri": result.artifact_uri,
        "report": result.report.model_dump(mode="json"),
        "metadata": dict(result.metadata),
    }
    if result.human_task_id:
        payload["human_task_id"] = result.human_task_id
    return payload


__all__ = ["inspect_frames"]
