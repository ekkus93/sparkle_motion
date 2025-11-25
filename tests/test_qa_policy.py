from __future__ import annotations

import json
from pathlib import Path

from sparkle_motion import qa_policy


def _write_policy(tmp_path: Path, settings: str) -> Path:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(settings, encoding="utf-8")
    return policy_path


def _write_schema(tmp_path: Path) -> Path:
    schema_path = tmp_path / "policy.schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["version"],
            }
        ),
        encoding="utf-8",
    )
    return schema_path


def _write_report(tmp_path: Path, payload: dict) -> Path:
    report_path = tmp_path / "qa_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def test_policy_approve_when_thresholds_met(tmp_path: Path) -> None:
    policy = _write_policy(
        tmp_path,
        """
version: 1
thresholds:
  prompt_match_min: 0.7
  max_finger_issue_ratio: 0.2
  allow_missing_audio: false
  rerun_on_artifact_notes: true
  min_audio_peak_db: -25
actions:
  approve_if:
    - prompt_match >= threshold(prompt_match_min)
    - finger_issues <= threshold(max_finger_issue_ratio)
  regenerate_if:
    - prompt_match < threshold(prompt_match_min)
  escalate_if:
    - missing_audio_detected == true
regenerate_stages:
  - images
""",
    )
    schema = _write_schema(tmp_path)
    report = _write_report(
        tmp_path,
        {
            "movie_title": "Test",
            "per_shot": [
                {"shot_id": "s1", "prompt_match": 0.9, "finger_issues": False, "artifact_notes": []}
            ],
        },
    )

    decision = qa_policy.evaluate_report(report_path=report, policy_path=policy, policy_schema_path=schema)

    assert decision.action == "approve"
    assert decision.reasons == []


def test_policy_regenerates_on_artifact_notes(tmp_path: Path) -> None:
    policy = _write_policy(
        tmp_path,
        """
version: 1
thresholds:
  prompt_match_min: 0.8
  max_finger_issue_ratio: 0.1
  allow_missing_audio: false
  rerun_on_artifact_notes: true
  min_audio_peak_db: -25
actions:
  approve_if:
    - artifact_notes == []
  regenerate_if:
    - artifact_notes != []
    - rerun_on_artifact_notes == true
  escalate_if: []
regenerate_stages:
  - images
  - videos
""",
    )
    report = _write_report(
        tmp_path,
        {
            "movie_title": "Test",
            "per_shot": [
                {
                    "shot_id": "s1",
                    "prompt_match": 0.95,
                    "finger_issues": False,
                    "artifact_notes": ["hand artifact"],
                }
            ],
        },
    )

    decision = qa_policy.evaluate_report(report_path=report, policy_path=policy)

    assert decision.action == "regenerate"
    assert decision.regenerate_stages == ["images", "videos"]
    assert any("artifact_notes" in reason for reason in decision.reasons)


def test_policy_escalates_on_missing_audio(tmp_path: Path) -> None:
    policy = _write_policy(
        tmp_path,
        """
version: 1
thresholds:
  prompt_match_min: 0.5
  max_finger_issue_ratio: 1.0
  allow_missing_audio: false
  rerun_on_artifact_notes: false
  min_audio_peak_db: -25
actions:
  approve_if:
    - missing_audio_detected == false
  regenerate_if:
    - artifact_notes != []
  escalate_if:
    - missing_audio_detected == true
regenerate_stages:
  - tts
""",
    )
    report = _write_report(
        tmp_path,
        {
            "movie_title": "Test",
            "per_shot": [
                {
                    "shot_id": "s1",
                    "prompt_match": 0.6,
                    "finger_issues": False,
                    "artifact_notes": [],
                    "missing_audio_detected": True,
                }
            ],
        },
    )

    decision = qa_policy.evaluate_report(report_path=report, policy_path=policy)

    assert decision.action == "escalate"
    assert any("missing_audio" in reason for reason in decision.reasons)