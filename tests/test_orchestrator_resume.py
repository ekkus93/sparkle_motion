from __future__ import annotations

import json
import time
from pathlib import Path

from sparkle_motion.orchestrator import Runner


def test_runner_resume_skips_completed_stages(tmp_path: Path):
    runner = Runner(runs_root=str(tmp_path))
    example = {"title": "Resume Test", "shots": [{"id": "s1"}]}
    run_id = "resume_test"
    run_dir = tmp_path / run_id
    # pre-create run dir and checkpoints/manifest indicating first two stages succeeded
    cp_dir = run_dir / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)

    # write a preexisting checkpoint for 'script' and 'images' with a marker
    script_cp = cp_dir / "script.json"
    images_cp = cp_dir / "images.json"
    script_cp.write_text(json.dumps({"stage": "script", "status": "success", "marker": "preexisting"}), encoding="utf-8")
    images_cp.write_text(json.dumps({"stage": "images", "status": "success", "marker": "preexisting"}), encoding="utf-8")

    # also write a manifest indicating success for those stages
    manifest_path = run_dir / "manifest.json"
    # craft manifest json indicating script and images already succeeded
    manifest_json = {"run_id": run_id, "events": [
        {"run_id": run_id, "stage": "script", "status": "success", "timestamp": time.time(), "attempt": 1},
        {"run_id": run_id, "stage": "images", "status": "success", "timestamp": time.time(), "attempt": 1},
    ]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    # run with resume=True — the runner should skip script/images and not overwrite their checkpoints
    out = runner.run(movie_plan=example, run_id=run_id, resume=True)

    # ensure the preexisting checkpoint markers remain (i.e., they were not overwritten)
    assert json.loads(script_cp.read_text(encoding="utf-8")).get("marker") == "preexisting"
    assert json.loads(images_cp.read_text(encoding="utf-8")).get("marker") == "preexisting"

    # runner should have produced at least an asset_refs.json file
    asset_refs = (run_dir / "asset_refs.json").read_text(encoding="utf-8")
    assert "shots" in asset_refs
