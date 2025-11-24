from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sparkle_motion.run_manifest import RunManifest, StageEvent, retry


def test_manifest_writer(tmp_path: Path):
    run_id = "test1"
    p = tmp_path / "runs" / run_id / "manifest.json"
    m = RunManifest(run_id=run_id, path=p)
    m.add_event(StageEvent(run_id=run_id, stage="s1", status="begin", timestamp=0.0))
    m.add_event(StageEvent(run_id=run_id, stage="s1", status="success", timestamp=1.0, attempt=1))
    m.save()

    loaded = RunManifest.load(p)
    assert loaded.run_id == run_id
    assert isinstance(loaded.events, list)
    assert any(ev["status"] == "success" for ev in loaded.events)


def test_retry_decorator_records_events(tmp_path: Path):
    run_id = "retry1"
    p = tmp_path / "runs" / run_id / "manifest.json"
    m = RunManifest(run_id=run_id, path=p)

    # function that fails twice then succeeds
    state: Dict[str, int] = {"calls": 0}

    @retry(max_attempts=4, base_delay=0.01, jitter=0.0, stage_name="unstable")
    def flaky(*, manifest=None):
        state["calls"] += 1
        if state["calls"] < 3:
            raise RuntimeError("transient")
        return "ok"

    res = flaky(manifest=m)
    assert res == "ok"
    # ensure events were recorded (begin, fail x2, success)
    statuses = [e["status"] for e in m.events]
    assert statuses[0] == "begin"
    assert statuses.count("fail") >= 2
    assert statuses[-1] == "success"
    # save and inspect file
    m.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["run_id"] == run_id
    assert len(data["events"]) >= 4
