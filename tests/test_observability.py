from __future__ import annotations

from pathlib import Path

from sparkle_motion import observability


def test_write_run_events_log_creates_sorted_timeline(tmp_path: Path) -> None:
    stage_events = [
        {"stage": "images", "status": "success", "timestamp": 5.0, "attempt": 1, "metadata": {}},
        {"stage": "script", "status": "begin", "timestamp": 1.0, "attempt": 0},
    ]
    memory_events = [
        {"event_type": "stage_success", "timestamp": 5.5, "payload": {"stage": "images"}},
        {"event_type": "qa_decision", "timestamp": 10.0, "payload": {"decision": "approve"}},
    ]

    out_path = tmp_path / "run_events.json"
    data = observability.write_run_events_log(
        run_id="demo",
        output_path=out_path,
        stage_events=stage_events,
        memory_events=memory_events,
    )

    assert out_path.exists()
    timeline = data["timeline"]
    assert [entry["source"] for entry in timeline] == ["stage", "stage", "memory", "memory"]
    assert timeline[0]["payload"]["stage"] == "script"
    assert timeline[-1]["payload"]["event_type"] == "qa_decision"
    assert data["counts"] == {"stage_events": 2, "memory_events": 2}
