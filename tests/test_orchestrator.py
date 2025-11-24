import json
from pathlib import Path

from sparkle_motion.orchestrator import Runner


def test_runner_creates_checkpoints(tmp_path: Path):
    runs_root = tmp_path / "runs"
    runner = Runner(runs_root=str(runs_root))

    movie_plan = {"title": "t", "shots": [{"id": "s1", "duration_sec": 1.0, "visual_description": "x"}]}
    run_id = "test_run"
    asset_refs = runner.run(movie_plan=movie_plan, run_id=run_id, resume=False)
    assert "shots" in asset_refs

    run_dir = runs_root / run_id
    assert run_dir.exists()
    # checkpoint for at least the first stage should exist
    cp = run_dir / "checkpoints" / "script.json"
    assert cp.exists()
    with cp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    assert obj.get("status") == "success"
    assert obj.get("attempt") == 1
    assert obj.get("metadata", {}).get("adapter")


def test_runner_checkpoint_attempt_counts(tmp_path: Path):
    runner = Runner(runs_root=str(tmp_path))
    attempts = {"count": 0}

    def flaky_stage(movie_plan, asset_refs, run_dir):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("boom")
        return asset_refs

    runner.stages = [("flaky", flaky_stage)]
    runner.run(movie_plan={"title": "test", "shots": []}, run_id="flaky", resume=False)
    cp = tmp_path / "flaky" / "checkpoints" / "flaky.json"
    data = json.loads(cp.read_text(encoding="utf-8"))
    assert data["status"] == "success"
    assert data["attempt"] == 2, "checkpoint should record final attempt count"
