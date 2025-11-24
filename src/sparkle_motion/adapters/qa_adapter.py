"""QA adapter stub (Qwen2-VL or similar).

Expose `run_qa(movie_plan, asset_refs, run_dir)` which samples frames and
produces a QAReport (dict). This stub raises MissingDependencyError with guidance.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from .common import MissingDependencyError


def run_qa(movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Run QA checks via a vision-language model and return a QAReport dict.

    Typical implementation uses a VQA-capable model like Qwen2-VL-7B-Instruct
    or another vision-language model.
    """
    raise MissingDependencyError(
        "QA adapter not implemented.\n"
        "Implement src.sparkle_motion.adapters.qa_adapter.run_qa using a VLM (e.g., Qwen2-VL).\n"
        "For now the orchestrator will fall back to a lightweight simulation if this raises."
    )
