from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import jsonschema
import yaml

from .schemas import QAReport


class QAPolicyError(RuntimeError):
    """Raised when a QA policy or report cannot be parsed or validated."""


@dataclass(frozen=True)
class QAGatingDecision:
    action: Literal["approve", "regenerate", "escalate"]
    reasons: List[str]
    metrics: Dict[str, Any]
    regenerate_stages: List[str] = field(default_factory=list)


def load_policy(policy_path: Path, *, schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate the QA policy YAML file."""

    try:
        data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - yaml exceptions vary
        raise QAPolicyError(f"Failed to read QA policy at {policy_path}: {exc}") from exc

    if schema_path and schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            raise QAPolicyError(f"QA policy does not match schema {schema_path}: {exc.message}") from exc
        except Exception as exc:
            raise QAPolicyError(f"Failed to validate QA policy schema {schema_path}: {exc}") from exc

    return data


def load_report(report_path: Path) -> QAReport:
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        return QAReport.model_validate(raw)
    except Exception as exc:  # pragma: no cover - Pydantic already tested elsewhere
        raise QAPolicyError(f"Failed to parse QA report at {report_path}: {exc}") from exc


def evaluate_report(
    *,
    report_path: Path,
    policy_path: Path,
    policy_schema_path: Optional[Path] = None,
) -> QAGatingDecision:
    """Convenience helper that loads inputs and produces a gating decision."""

    policy = load_policy(policy_path, schema_path=policy_schema_path)
    report = load_report(report_path)
    evaluator = QAPolicyEvaluator(policy, policy_path=policy_path)
    return evaluator.evaluate(report)


class QAPolicyEvaluator:
    _PREDICATE_RE = re.compile(r"^(?P<lhs>[A-Za-z_]+)\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<rhs>.+)$")
    _THRESHOLD_RE = re.compile(r"threshold\((?P<name>[A-Za-z0-9_]+)\)")

    def __init__(self, policy: Dict[str, Any], *, policy_path: Optional[Path] = None) -> None:
        self.policy = policy
        self.policy_path = policy_path
        self.thresholds = policy.get("thresholds", {})
        self.actions = policy.get("actions", {})
        self.regenerate_stages = policy.get("regenerate_stages", [])

    def evaluate(self, report: QAReport) -> QAGatingDecision:
        metrics = self._aggregate_metrics(report)
        decision, reasons = self._evaluate_actions(metrics)
        return QAGatingDecision(
            action=decision,
            reasons=reasons,
            metrics=metrics,
            regenerate_stages=list(self.regenerate_stages),
        )

    def _aggregate_metrics(self, report: QAReport) -> Dict[str, Any]:
        shots = report.per_shot
        total_shots = max(len(shots), 1)
        prompt_scores = [shot.prompt_match for shot in shots if shot.prompt_match is not None]
        artifact_notes: List[str] = []
        finger_issue_count = 0
        missing_audio = False
        safety_violation = False

        for shot in shots:
            artifact_notes.extend(shot.artifact_notes)
            if shot.finger_issues:
                finger_issue_count += 1
            missing_audio = missing_audio or shot.missing_audio_detected
            safety_violation = safety_violation or shot.safety_violation

        prompt_match = min(prompt_scores) if prompt_scores else 1.0
        finger_ratio = finger_issue_count / total_shots

        metrics: Dict[str, Any] = {
            "prompt_match": prompt_match,
            "finger_issues": finger_ratio,
            "artifact_notes": artifact_notes,
            "missing_audio_detected": missing_audio,
            "safety_violation": safety_violation,
            "issues_found": len(artifact_notes) + finger_issue_count,
            "shots_total": total_shots,
            "rerun_on_artifact_notes": self.thresholds.get("rerun_on_artifact_notes", False),
        }

        if report.aggregate_prompt_match is not None:
            metrics["aggregate_prompt_match"] = report.aggregate_prompt_match

        return metrics

    def _evaluate_actions(self, metrics: Dict[str, Any]) -> Tuple[Literal["approve", "regenerate", "escalate"], List[str]]:
        ordered_actions: Iterable[Tuple[str, List[str]]] = (
            ("approve", self.actions.get("approve_if", [])),
            ("regenerate", self.actions.get("regenerate_if", [])),
            ("escalate", self.actions.get("escalate_if", [])),
        )

        for action_name, predicates in ordered_actions:
            if not predicates:
                continue
            capture_success = action_name != "approve"
            passed, reasons = self._predicates_satisfied(predicates, metrics, capture_success=capture_success)
            if passed:
                return action_name, reasons if action_name != "approve" else []

        return "regenerate", ["No QA policy predicates matched; defaulting to regenerate"]

    def _predicates_satisfied(
        self,
        predicates: List[str],
        metrics: Dict[str, Any],
        *,
        capture_success: bool = False,
    ) -> Tuple[bool, List[str]]:
        messages: List[str] = []
        for expr in predicates:
            ok, message, lhs_value, rhs_value = self._evaluate_predicate(expr, metrics)
            if not ok:
                messages.append(message or f"Predicate failed: {expr}")
                return False, messages
            if capture_success:
                messages.append(f"{expr} (lhs={lhs_value!r}, rhs={rhs_value!r})")
        return True, messages

    def _evaluate_predicate(self, expr: str, metrics: Dict[str, Any]) -> Tuple[bool, str, Any, Any]:
        match = self._PREDICATE_RE.match(expr.strip())
        if not match:
            return False, f"Unsupported predicate syntax: {expr}", None, None

        lhs = match.group("lhs")
        op = match.group("op")
        rhs_raw = match.group("rhs").strip()
        lhs_value = metrics.get(lhs)

        if lhs_value is None:
            return False, f"Metric '{lhs}' unavailable for predicate '{expr}'", None, None

        rhs_value: Any
        threshold_match = self._THRESHOLD_RE.match(rhs_raw)
        if threshold_match:
            thresh_name = threshold_match.group("name")
            if thresh_name not in self.thresholds:
                return False, f"Unknown threshold '{thresh_name}' referenced in '{expr}'", lhs_value, None
            rhs_value = self.thresholds[thresh_name]
        elif rhs_raw.lower() in {"true", "false"}:
            rhs_value = rhs_raw.lower() == "true"
        elif rhs_raw == "[]":
            rhs_value = []
        else:
            try:
                rhs_value = float(rhs_raw)
            except ValueError:
                rhs_value = rhs_raw.strip('"')

        comparator = {
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
        }.get(op)

        if comparator is None:
            return False, f"Unsupported operator '{op}' in predicate '{expr}'", lhs_value, rhs_value

        try:
            result = comparator(lhs_value, rhs_value)
        except TypeError as exc:  # pragma: no cover - depends on predicate
            return False, f"Type mismatch evaluating '{expr}': {exc}", lhs_value, rhs_value

        if result:
            return True, "", lhs_value, rhs_value

        return False, f"Predicate '{expr}' failed (lhs={lhs_value!r}, rhs={rhs_value!r})", lhs_value, rhs_value


__all__ = ["QAGatingDecision", "QAPolicyError", "evaluate_report", "load_policy", "load_report", "QAPolicyEvaluator"]