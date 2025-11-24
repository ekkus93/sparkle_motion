from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sparkle_motion.run_manifest import RunManifest, StageEvent, retry
from sparkle_motion.orchestrator import Runner


def test_retry_decorator_records_attempts() -> None:
    # manifest in-memory (no need to persist to disk for this test)
    m = RunManifest(run_id="t_retry", path=None)

    # flaky function: fail twice then succeed
    state: dict[str, int] = {"calls": 0}

    def flaky(x: Any, *, manifest: RunManifest = None):
        state["calls"] += 1
        if state["calls"] < 3:
            raise RuntimeError(f"transient-{state['calls']}")
        return "ok"

    wrapped = retry(max_attempts=3, base_delay=0.01, max_delay=0.02, jitter=0.0, stage_name="flaky_test")(flaky)

    res = wrapped(None, manifest=m)
    assert res == "ok"

    # inspect manifest events in memory
    statuses = [e.get("status") for e in m.events]
    assert statuses[0] == "begin"
    # expect two fail events then success (order: begin, fail, fail, success)
    assert statuses[-1] == "success"
    assert statuses.count("fail") == 2


def test_manifest_load_save_integrity(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    m1 = RunManifest(run_id="m1", path=p)
    m1.add_event(StageEvent(run_id="m1", stage="s1", status="begin", timestamp=time.time(), attempt=0))
    m1.add_event(StageEvent(run_id="m1", stage="s1", status="fail", timestamp=time.time(), attempt=1, error="x"))
    m1.save()

    # load, append, save again
    m2 = RunManifest.load(p)
    # append success
    m2.add_event(StageEvent(run_id="m1", stage="s1", status="success", timestamp=time.time(), attempt=2))
    m2.save()

    m3 = RunManifest.load(p)
    assert len(m3.events) == 3
    assert m3.events[0]["status"] == "begin"
    assert m3.events[-1]["status"] == "success"


def test_runner_reruns_failed_stage(tmp_path: Path) -> None:
    runner = Runner(runs_root=str(tmp_path))
    example = {"title": "ReRun Test", "shots": [{"id": "s1"}]}
    run_id = "r_rerun"
    run_dir = tmp_path / run_id
    cp_dir = run_dir / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)

    # write a failed checkpoint for images
    images_cp = cp_dir / "images.json"
    images_cp.write_text(json.dumps({"stage": "images", "status": "failed"}), encoding="utf-8")

    # write manifest showing images failed
    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = {"run_id": run_id, "events": [
        {"run_id": run_id, "stage": "images", "status": "fail", "timestamp": time.time(), "attempt": 1}
    ]}
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    # run with resume=True; runner should re-run images and produce a success checkpoint
    out = runner.run(movie_plan=example, run_id=run_id, resume=True)

    # reload manifest and checkpoint to confirm success
    from sparkle_motion.run_manifest import RunManifest as RM

    m = RM.load(manifest_path)
    assert m.last_status_for_stage("images") == "success"

    cp = json.loads(images_cp.read_text(encoding="utf-8"))
    assert cp.get("status") == "success"


def test_manifest_save_is_atomic(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    m = RunManifest(run_id="atomic", path=p)
    m.add_event(StageEvent(run_id="atomic", stage="s", status="begin", timestamp=time.time(), attempt=0))
    # call save (will write via temp file then replace)
    m.save()

    # file must exist and contain expected JSON
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert '"run_id": "atomic"' in txt

    # There should be no temp manifest files left in the directory
    tmp_files = [x for x in p.parent.iterdir() if x.name.startswith(".manifest-")]
    assert not tmp_files, f"temp files left behind: {tmp_files}"
