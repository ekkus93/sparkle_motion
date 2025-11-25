from __future__ import annotations

import json
from pathlib import Path

from sparkle_motion.orchestrator import Runner


def _movie_plan() -> dict:
    return {
        "title": "Smoke Test",
        "shots": [
            {
                "id": "shot_001",
                "duration_sec": 2.0,
                "visual_description": "A hero stands on a rooftop.",
                "dialogue": [{"character_id": "hero", "text": "We did it."}],
            }
        ],
        "characters": [
            {"id": "hero", "name": "Hero", "description": "Stoic protagonist"},
        ],
    }


def test_runner_smoke(tmp_path: Path) -> None:
    runner = Runner(runs_root=str(tmp_path))
    run_id = "smoke"

    asset_refs = runner.run(movie_plan=_movie_plan(), run_id=run_id, resume=False)

    run_dir = tmp_path / run_id
    assert (run_dir / "movie_plan.json").exists()
    assert (run_dir / "asset_refs.json").exists()
    assert (run_dir / "run_events.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "qa_report.json").exists()

    shots = asset_refs.get("shots", {})
    assert "shot_001" in shots
    shot_refs = shots["shot_001"]
    assert shot_refs.get("start_frame")
    assert shot_refs.get("end_frame")
    assert shot_refs.get("final_video_clip")
    assert Path(shot_refs["final_video_clip"]).exists(), "final video clip should be written"

    run_events = json.loads((run_dir / "run_events.json").read_text(encoding="utf-8"))
    assert run_events["timeline"], "timeline should contain entries"