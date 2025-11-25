from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .services import MemoryService


DecisionStatus = Literal["pending", "approved", "revise"]


@dataclass
class HumanReviewDecision:
    stage: str
    status: DecisionStatus
    notes: Optional[str] = None


class HumanReviewCoordinator:
    """Coordinates human review hooks for script and asset stages.

    Decisions are provided via small JSON files under
    `runs/<run_id>/human_review/<stage>.json`, for example:

    ```json
    {"decision": "revise", "notes": "Character motivations unclear"}
    ```
    """

    def __init__(self, memory_service: MemoryService) -> None:
        self.memory_service = memory_service
        self.run_dir = memory_service.session.run_dir

    def _decision_path(self, stage: str) -> Path:
        return self.run_dir / "human_review" / f"{stage}.json"

    def _read_decision(self, stage: str) -> HumanReviewDecision:
        path = self._decision_path(stage)
        if not path.exists():
            return HumanReviewDecision(stage=stage, status="pending")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {"decision": "pending"}
        decision = data.get("decision", "pending")
        notes = data.get("notes")
        if decision not in ("pending", "approved", "revise"):
            decision = "pending"
        if decision != "pending":
            self.memory_service.record_human_feedback(stage=stage, decision=decision, notes=notes)
        return HumanReviewDecision(stage=stage, status=decision, notes=notes)

    def require_script_signoff(self) -> HumanReviewDecision:
        """Check for script approval before moving to image generation."""
        decision = self._read_decision("script")
        if decision.status == "pending":
            self.memory_service.record_event(
                "human_review_pending",
                {"stage": "script", "message": "Waiting for script approval"},
            )
        return decision

    def require_images_signoff(self) -> HumanReviewDecision:
        """Check for approval of generated base images before proceeding."""
        decision = self._read_decision("images")
        if decision.status == "pending":
            self.memory_service.record_event(
                "human_review_pending",
                {"stage": "images", "message": "Waiting for base image approval"},
            )
        return decision