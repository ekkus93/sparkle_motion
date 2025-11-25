from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class TimelineEvent:
    """Unified event record combining stage manifest entries and memory events."""

    timestamp: float
    source: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "source": self.source, "payload": self.payload}


def _stage_event_to_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": event.get("stage"),
        "status": event.get("status"),
        "attempt": event.get("attempt"),
        "error": event.get("error"),
        "metadata": event.get("metadata", {}),
    }


def _memory_event_to_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_type": event.get("event_type"),
        "payload": event.get("payload", {}),
    }


def merged_timeline(
    *,
    stage_events: Iterable[Dict[str, Any]],
    memory_events: Iterable[Dict[str, Any]],
) -> List[TimelineEvent]:
    """Return a timestamp-sorted timeline of stage + memory events."""

    combined: List[TimelineEvent] = []
    for ev in stage_events:
        combined.append(
            TimelineEvent(
                timestamp=float(ev.get("timestamp", 0.0)),
                source="stage",
                payload=_stage_event_to_payload(ev),
            )
        )
    for ev in memory_events:
        combined.append(
            TimelineEvent(
                timestamp=float(ev.get("timestamp", 0.0)),
                source="memory",
                payload=_memory_event_to_payload(ev),
            )
        )
    combined.sort(key=lambda e: e.timestamp)
    return combined


def write_run_events_log(
    *,
    run_id: str,
    output_path: Path,
    stage_events: Iterable[Dict[str, Any]],
    memory_events: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Persist a merged run timeline to ``output_path`` and return the data structure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_list = list(stage_events)
    memory_list = list(memory_events)
    timeline = merged_timeline(stage_events=stage_list, memory_events=memory_list)
    data = {
        "run_id": run_id,
        "timeline": [event.to_dict() for event in timeline],
        "counts": {
            "stage_events": len(stage_list),
            "memory_events": len(memory_list),
        },
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


__all__ = ["TimelineEvent", "merged_timeline", "write_run_events_log"]
