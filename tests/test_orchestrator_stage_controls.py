from __future__ import annotations

import time
from pathlib import Path

import pytest

from sparkle_motion.orchestrator import Runner


def _bootstrap_run(tmp_path: Path, run_id: str = "stage_controls") -> Path:
    runner = Runner(runs_root=str(tmp_path))
    movie_plan = {"title": "Stage Controls", "shots": [{"id": "s1"}]}
    runner.run(movie_plan=movie_plan, run_id=run_id, resume=False)
    return tmp_path / run_id


def test_start_stage_skips_prior_stages(tmp_path: Path) -> None:
    run_dir = _bootstrap_run(tmp_path, run_id="start_stage")
    runner = Runner(runs_root=str(tmp_path))

    script_cp = run_dir / "checkpoints" / "script.json"
    images_cp = run_dir / "checkpoints" / "images.json"
    before_script_mtime = script_cp.stat().st_mtime
    before_images_mtime = images_cp.stat().st_mtime

    time.sleep(0.02)
    runner.run(movie_plan=None, run_id="start_stage", resume=False, start_stage="images")

    assert script_cp.stat().st_mtime == pytest.approx(before_script_mtime)
    assert images_cp.stat().st_mtime > before_images_mtime


def test_only_stage_reruns_single_stage(tmp_path: Path) -> None:
    run_dir = _bootstrap_run(tmp_path, run_id="only_stage")
    runner = Runner(runs_root=str(tmp_path))

    qa_cp = run_dir / "checkpoints" / "qa.json"
    script_cp = run_dir / "checkpoints" / "script.json"
    before_qa_mtime = qa_cp.stat().st_mtime
    before_script_mtime = script_cp.stat().st_mtime

    time.sleep(0.02)
    runner.run(movie_plan=None, run_id="only_stage", resume=True, only_stage="qa")

    assert qa_cp.stat().st_mtime > before_qa_mtime
    assert script_cp.stat().st_mtime == pytest.approx(before_script_mtime)


def test_invalid_stage_requests_raise(tmp_path: Path) -> None:
    runner = Runner(runs_root=str(tmp_path))
    movie_plan = {"title": "Invalid Stage", "shots": []}
    runner.run(movie_plan=movie_plan, run_id="invalid", resume=False)

    with pytest.raises(ValueError):
        runner.run(movie_plan=None, run_id="invalid", resume=False, start_stage="does_not_exist")

    with pytest.raises(ValueError):
        runner.run(movie_plan=None, run_id="invalid", resume=False, only_stage="missing")

    with pytest.raises(ValueError):
        runner.run(movie_plan=None, run_id="invalid", resume=False, start_stage="script", only_stage="images")


def test_resume_from_stage_helper(tmp_path: Path) -> None:
    run_id = "resume_helper"
    run_dir = _bootstrap_run(tmp_path, run_id=run_id)
    runner = Runner(runs_root=str(tmp_path))

    script_cp = run_dir / "checkpoints" / "script.json"
    images_cp = run_dir / "checkpoints" / "images.json"
    before_script = script_cp.stat().st_mtime
    before_images = images_cp.stat().st_mtime

    time.sleep(0.02)
    runner.resume_from_stage(run_id=run_id, stage="images")

    assert script_cp.stat().st_mtime == pytest.approx(before_script)
    assert images_cp.stat().st_mtime > before_images


def test_retry_stage_helper(tmp_path: Path) -> None:
    run_id = "retry_helper"
    run_dir = _bootstrap_run(tmp_path, run_id=run_id)
    runner = Runner(runs_root=str(tmp_path))

    qa_cp = run_dir / "checkpoints" / "qa.json"
    videos_cp = run_dir / "checkpoints" / "videos.json"
    before_qa = qa_cp.stat().st_mtime
    before_videos = videos_cp.stat().st_mtime

    time.sleep(0.02)
    runner.retry_stage(run_id=run_id, stage="qa")

    assert videos_cp.stat().st_mtime == pytest.approx(before_videos)
    assert qa_cp.stat().st_mtime > before_qa
