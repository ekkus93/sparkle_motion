import json
from pathlib import Path

from sparkle_motion.orchestrator import Runner


def test_runner_creates_checkpoints(tmp_path: Path):
    runs_root = tmp_path / "runs"
    runner = Runner(runs_root=str(runs_root))

    movie_plan = {"title": "t", "shots": [{"id": "s1", "duration_sec": 1.0, "visual_description": "x"}]}
    run_id = "test_run"
    asset_refs = runner.run(movie_plan=movie_plan, run_id=run_id, resume=False)

    run_dir = runs_root / run_id
    assert run_dir.exists()
    # checkpoint for at least the first stage should exist
    cp = run_dir / "checkpoints" / "script.json"
    assert cp.exists()
    with cp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    assert obj.get("status") == "success"
