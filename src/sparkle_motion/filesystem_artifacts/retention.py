"""Retention planning helpers for filesystem-backed artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import shutil
import sqlite3
import time
from typing import Literal, Optional, Sequence

from .config import FilesystemArtifactsConfig

LOG = logging.getLogger(__name__)

DeletionReason = Literal["max_age", "max_bytes", "min_free"]


@dataclass(frozen=True)
class ArtifactRow:
    """Subset of artifact metadata required for retention planning."""

    artifact_id: str
    run_id: str
    stage: str
    artifact_type: str
    relative_path: str
    size_bytes: int
    created_at: int

    def created_at_iso(self) -> str:
        return datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class RetentionOptions:
    """Retention constraints supplied by the operator."""

    max_bytes: Optional[int] = None
    max_age_seconds: Optional[int] = None
    min_free_bytes: Optional[int] = None


@dataclass(frozen=True)
class DeletionCandidate:
    """Artifact slated for removal along with the triggering reason."""

    artifact: ArtifactRow
    reason: DeletionReason


@dataclass(frozen=True)
class RetentionPlan:
    """Summary of a retention sweep."""

    candidates: tuple[DeletionCandidate, ...]
    initial_count: int
    initial_bytes: int

    @property
    def freed_bytes(self) -> int:
        return sum(candidate.artifact.size_bytes for candidate in self.candidates)

    @property
    def freed_count(self) -> int:
        return len(self.candidates)

    @property
    def remaining_bytes(self) -> int:
        return max(self.initial_bytes - self.freed_bytes, 0)

    def summary_by_reason(self) -> dict[DeletionReason, int]:
        summary: dict[DeletionReason, int] = {}
        for candidate in self.candidates:
            summary[candidate.reason] = summary.get(candidate.reason, 0) + 1
        return summary


def load_artifacts(config: FilesystemArtifactsConfig, *, runs: Optional[set[str]] = None) -> list[ArtifactRow]:
    """Load artifact rows from the SQLite index ordered by creation time."""

    clauses = []
    params: list[object] = []
    if runs:
        placeholders = ",".join("?" for _ in runs)
        clauses.append(f"run_id IN ({placeholders})")
        params.extend(sorted(runs))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT artifact_id, run_id, stage, artifact_type, relative_path, size_bytes, created_at "
        "FROM artifacts "
        f"{where_sql} "
        "ORDER BY created_at ASC, artifact_id ASC"
    )
    rows: list[ArtifactRow] = []
    with sqlite3.connect(str(config.index_path)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(sql, params):
            rows.append(
                ArtifactRow(
                    artifact_id=row["artifact_id"],
                    run_id=row["run_id"],
                    stage=row["stage"],
                    artifact_type=row["artifact_type"],
                    relative_path=row["relative_path"],
                    size_bytes=int(row["size_bytes"] or 0),
                    created_at=int(row["created_at"] or 0),
                )
            )
    return rows


def plan_retention(
    artifacts: Sequence[ArtifactRow],
    options: RetentionOptions,
    *,
    disk_free_bytes: int,
    now_ts: Optional[int] = None,
) -> RetentionPlan:
    """Plan which artifacts should be removed to satisfy retention constraints."""

    if options.max_bytes is not None and options.max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if options.max_age_seconds is not None and options.max_age_seconds < 0:
        raise ValueError("max_age_seconds must be non-negative")
    if options.min_free_bytes is not None and options.min_free_bytes < 0:
        raise ValueError("min_free_bytes must be non-negative")

    now = now_ts or int(time.time())
    ordered = list(artifacts)
    initial_bytes = sum(item.size_bytes for item in ordered)
    candidates: list[DeletionCandidate] = []
    marked: set[str] = set()

    def mark(row: ArtifactRow, reason: DeletionReason) -> None:
        if row.artifact_id in marked:
            return
        marked.add(row.artifact_id)
        candidates.append(DeletionCandidate(artifact=row, reason=reason))

    remaining = ordered
    if options.max_age_seconds is not None:
        cutoff = now - options.max_age_seconds
        aged_out = [row for row in remaining if row.created_at < cutoff]
        for row in aged_out:
            mark(row, "max_age")
        remaining = [row for row in remaining if row.artifact_id not in marked]

    if options.max_bytes is not None:
        allowed = options.max_bytes
        remaining_bytes = sum(row.size_bytes for row in remaining)
        excess = remaining_bytes - allowed
        if excess > 0:
            for row in remaining:
                mark(row, "max_bytes")
                excess -= row.size_bytes
                if excess <= 0:
                    break
            remaining = [row for row in remaining if row.artifact_id not in marked]

    if options.min_free_bytes is not None:
        shortage = options.min_free_bytes - disk_free_bytes - sum(
            candidate.artifact.size_bytes for candidate in candidates
        )
        if shortage > 0:
            for row in remaining:
                mark(row, "min_free")
                shortage -= row.size_bytes
                if shortage <= 0:
                    break

    return RetentionPlan(
        candidates=tuple(candidates),
        initial_count=len(ordered),
        initial_bytes=initial_bytes,
    )


def execute_plan(plan: RetentionPlan, config: FilesystemArtifactsConfig, *, dry_run: bool) -> None:
    """Apply a retention plan by removing files and deleting SQLite rows."""

    if dry_run or not plan.candidates:
        return

    with sqlite3.connect(str(config.index_path)) as conn:
        for candidate in plan.candidates:
            _delete_artifact(candidate, config, conn)
        conn.commit()


def _delete_artifact(candidate: DeletionCandidate, config: FilesystemArtifactsConfig, conn: sqlite3.Connection) -> None:
    payload_path = (config.root / candidate.artifact.relative_path).resolve()
    artifact_dir = payload_path.parent
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=False)
    else:
        LOG.warning("Artifact directory %s missing; continuing", artifact_dir)
    conn.execute("DELETE FROM artifacts WHERE artifact_id = ?", (candidate.artifact.artifact_id,))
    _record_memory_event(candidate)


def _record_memory_event(candidate: DeletionCandidate) -> None:
    payload = {
        "artifact_id": candidate.artifact.artifact_id,
        "run_id": candidate.artifact.run_id,
        "stage": candidate.artifact.stage,
        "reason": candidate.reason,
    }
    try:
        from sparkle_motion import adk_helpers as _adk_helpers
        from sparkle_motion.adk_helpers import MemoryWriteError as _MemoryWriteError
    except ImportError:  # pragma: no cover - available at runtime
        return

    try:
        _adk_helpers.write_memory_event(
            run_id=candidate.artifact.run_id,
            event_type="filesystem_artifact_pruned",
            payload=payload,
        )
    except _MemoryWriteError:
        LOG.debug("Memory write failed for %s", candidate.artifact.artifact_id)


__all__ = [
    "ArtifactRow",
    "RetentionOptions",
    "DeletionCandidate",
    "RetentionPlan",
    "load_artifacts",
    "plan_retention",
    "execute_plan",
]
