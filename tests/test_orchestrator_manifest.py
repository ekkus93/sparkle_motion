from __future__ import annotations

from pathlib import Path

from sparkle_motion.orchestrator import Runner


def test_runner_creates_manifest(tmp_path: Path):
    runner = Runner(runs_root=str(tmp_path))
    example = {"title": "Test", "shots": [{"id": "s1"}]}
    run_id = "r_manifest"
    out = runner.run(movie_plan=example, run_id=run_id, resume=False)
    assert isinstance(out, dict)
    run_dir = tmp_path / run_id
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json should be created"
    text = manifest_path.read_text(encoding="utf-8")
    assert "run_id" in text
    assert "events" in text
    # confirm that a checkpoint for at least the first stage exists
    cp = run_dir / "checkpoints" / "script.json"
    assert cp.exists()
