from __future__ import annotations

import json
from pathlib import Path

from sparkle_motion.orchestrator import Runner


def _policy_text() -> str:
    return """
version: 1
thresholds:
  prompt_match_min: 0.8
  max_finger_issue_ratio: 0.1
  allow_missing_audio: false
  rerun_on_artifact_notes: true
  min_audio_peak_db: -30
actions:
  approve_if:
    - artifact_notes == []
  regenerate_if:
    - artifact_notes != []
    - rerun_on_artifact_notes == true
  escalate_if:
    - missing_audio_detected == true
regenerate_stages:
  - images
  - videos
"""


def test_runner_records_qa_gating(tmp_path: Path) -> None:
    policy_path = tmp_path / "qa.yaml"
    policy_path.write_text(_policy_text(), encoding="utf-8")

    def stub_qa(movie_plan, asset_refs, run_dir):
        qa_report = run_dir / "qa_report.json"
        qa_payload = {
            "movie_title": movie_plan.get("title"),
            "per_shot": [
                {
                    "shot_id": "s1",
                    "prompt_match": 0.95,
                    "finger_issues": False,
                    "artifact_notes": ["edge artifact"],
                }
            ],
        }
        qa_report.write_text(json.dumps(qa_payload), encoding="utf-8")
        return asset_refs, {"qa_report": str(qa_report)}

    runner = Runner(runs_root=str(tmp_path), qa_policy_path=policy_path)
    runner.stages = [("qa", stub_qa)]

    movie_plan = {"title": "QA", "shots": [{"id": "s1", "duration_sec": 1.0, "visual_description": "x"}]}
    run_id = "qa_run"
    runner.run(movie_plan=movie_plan, run_id=run_id, resume=False)

    run_dir = tmp_path / run_id
    actions_path = run_dir / "qa_actions.json"
    assert actions_path.exists()
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    assert actions["decision"] == "regenerate"
    assert actions["regenerate_stages"] == ["images", "videos"]

    memory_log = json.loads((run_dir / "memory_log.json").read_text(encoding="utf-8"))
    gating_events = [entry for entry in memory_log if entry["event_type"] == "qa_gating_decision"]
    assert gating_events, "Expected qa_gating_decision to be recorded"


def test_runner_auto_regenerates_requested_stages(tmp_path: Path) -> None:
  policy_path = tmp_path / "qa.yaml"
  policy_path.write_text(_policy_text(), encoding="utf-8")

  call_counts = {"images": 0, "qa": 0}

  def stub_images(movie_plan, asset_refs, run_dir):
    call_counts["images"] += 1
    return asset_refs

  def stub_qa(movie_plan, asset_refs, run_dir):
    call_counts["qa"] += 1
    qa_report = run_dir / "qa_report.json"
    artifact_notes = [] if call_counts["qa"] > 1 else ["redo"]
    qa_payload = {
      "movie_title": movie_plan.get("title"),
      "per_shot": [
        {
          "shot_id": "s1",
          "prompt_match": 0.9,
          "finger_issues": False,
          "artifact_notes": artifact_notes,
        }
      ],
    }
    qa_report.write_text(json.dumps(qa_payload), encoding="utf-8")
    return asset_refs, {"qa_report": str(qa_report)}

  runner = Runner(
    runs_root=str(tmp_path),
    qa_policy_path=policy_path,
    auto_regenerate_on_qa_fail=True,
  )
  runner.stages = [
    ("script", runner.stage_script),
    ("images", stub_images),
    ("qa", stub_qa),
  ]

  movie_plan = {"title": "Auto QA", "shots": [{"id": "s1", "duration_sec": 1.0}]}
  runner.run(movie_plan=movie_plan, run_id="qa_auto", resume=False)

  assert call_counts["images"] == 2, "images stage should rerun after QA regenerate request"
  assert call_counts["qa"] == 2, "QA should run again after auto-regeneration"

  memory_log = json.loads((tmp_path / "qa_auto" / "memory_log.json").read_text(encoding="utf-8"))
  auto_events = [entry for entry in memory_log if entry["event_type"] == "qa_auto_regenerate_triggered"]
  assert auto_events, "Expected qa_auto_regenerate_triggered event to be recorded"