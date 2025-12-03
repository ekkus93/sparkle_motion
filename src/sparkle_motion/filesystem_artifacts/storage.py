from __future__ import annotations

"""Filesystem-backed artifact persistence helpers."""

import hashlib
import json
import mimetypes
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - imported only for typing
    from sparkle_motion.adk_helpers import MemoryWriteError

from .config import FilesystemArtifactsConfig
from .models import ArtifactManifest, ArtifactRecord, StoragePaths


class ArtifactStorageError(RuntimeError):
    """Base class for filesystem artifact persistence failures."""


class PayloadTooLargeError(ArtifactStorageError):
    """Raised when an upload payload exceeds the configured byte limit."""

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  relative_path TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  checksum_sha256 TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  metadata TEXT
);
CREATE INDEX IF NOT EXISTS ix_artifacts_run_stage ON artifacts(run_id, stage);
CREATE INDEX IF NOT EXISTS ix_artifacts_type ON artifacts(artifact_type);
"""

_PageMarker = Tuple[int, str]


@dataclass
class FilesystemArtifactStore:
    """Persists artifacts on disk and records metadata in SQLite."""

    config: FilesystemArtifactsConfig

    def __post_init__(self) -> None:
        self.config.root.mkdir(parents=True, exist_ok=True)
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SQLITE_DDL)
            conn.commit()

    def save_artifact(
        self,
        *,
        manifest: ArtifactManifest,
        payload: bytes | None,
        filename_hint: str | None,
    ) -> ArtifactRecord:
        payload_bytes = payload or b""
        if len(payload_bytes) > self.config.max_payload_bytes:
            raise PayloadTooLargeError("Payload size exceeds ARTIFACTS_FS_MAX_BYTES limit")

        artifact_slug = uuid.uuid4().hex
        artifact_id = f"{manifest.run_id}/{manifest.stage}/{manifest.artifact_type}/{artifact_slug}"
        artifact_dir = self.config.root / manifest.run_id / manifest.stage / artifact_slug
        artifact_dir.mkdir(parents=True, exist_ok=True)

        extension = _extension_for(manifest.mime_type, filename_hint)
        payload_path = artifact_dir / f"artifact{extension}"
        payload_path.write_bytes(payload_bytes)

        checksum = _hash_file(payload_path)
        size_bytes = payload_path.stat().st_size
        created_at = int(time.time())
        artifact_uri = f"artifact+fs://{artifact_id}"

        manifest_doc = {
            "artifact_id": artifact_id,
            "artifact_uri": artifact_uri,
            "artifact_type": manifest.artifact_type,
            "run_id": manifest.run_id,
            "stage": manifest.stage,
            "mime_type": manifest.mime_type,
            "size_bytes": size_bytes,
            "checksum": {"sha256": checksum},
            "local_path": str(payload_path),
            "download_url": payload_path.resolve().as_uri(),
            "metadata": dict(manifest.metadata),
            "local_path_hint": manifest.local_path_hint,
            "storage_backend": "filesystem",
            "created_at": created_at,
        }

        if manifest.tags is not None:
            manifest_doc["tags"] = manifest.tags

        manifest_path = artifact_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")

        relative_path = str(payload_path.relative_to(self.config.root))
        metadata_json = json.dumps(manifest.metadata, ensure_ascii=False) if manifest.metadata else None

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, run_id, stage, artifact_type, mime_type,
                    relative_path, manifest_json, size_bytes, checksum_sha256,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    manifest.run_id,
                    manifest.stage,
                    manifest.artifact_type,
                    manifest.mime_type,
                    relative_path,
                    json.dumps(manifest_doc, ensure_ascii=False),
                    size_bytes,
                    checksum,
                    created_at,
                    metadata_json,
                ),
            )
            conn.commit()

        record = self._row_to_record(
            {
                "artifact_id": artifact_id,
                "run_id": manifest.run_id,
                "stage": manifest.stage,
                "artifact_type": manifest.artifact_type,
                "mime_type": manifest.mime_type,
                "relative_path": relative_path,
                "manifest_json": json.dumps(manifest_doc, ensure_ascii=False),
                "size_bytes": size_bytes,
                "checksum_sha256": checksum,
                "created_at": created_at,
                "metadata": metadata_json,
            }
        )
        self._record_memory_event(record)
        return record

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return self._row_to_record(dict(row)) if row else None

    def list_artifacts(
        self,
        *,
        run_id: str,
        stage: str | None,
        artifact_type: str | None,
        limit: int,
        order: Literal["asc", "desc"],
        page_marker: _PageMarker | None,
    ) -> tuple[list[ArtifactRecord], Optional[str]]:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if artifact_type:
            clauses.append("artifact_type = ?")
            params.append(artifact_type)

        if page_marker:
            created_at, artifact_id = page_marker
            comparator = ">" if order == "asc" else "<"
            clauses.append(
                f"(created_at {comparator} ? OR (created_at = ? AND artifact_id {comparator} ?))"
            )
            params.extend([created_at, created_at, artifact_id])

        where_sql = " AND ".join(clauses)
        order_sql = "ORDER BY created_at ASC, artifact_id ASC" if order == "asc" else "ORDER BY created_at DESC, artifact_id DESC"
        sql = f"SELECT * FROM artifacts WHERE {where_sql} {order_sql} LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        items = [self._row_to_record(dict(row)) for row in rows]
        next_token = None
        if len(items) == limit and items:
            last = items[-1]
            next_token = f"{last.created_at}:{last.artifact_id}"
        return items, next_token

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.config.index_path), timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_record(self, row: dict[str, Any]) -> ArtifactRecord:
        manifest = json.loads(row["manifest_json"])
        relative_path = row["relative_path"]
        payload_path = (self.config.root / relative_path).resolve()
        manifest_path = payload_path.with_name("manifest.json")
        storage = StoragePaths(
            relative_path=relative_path,
            absolute_path=str(payload_path),
            manifest_path=str(manifest_path),
        )
        metadata = json.loads(row["metadata"]) if row.get("metadata") else manifest.get("metadata") or {}
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            artifact_uri=manifest.get(
                "artifact_uri",
                f"artifact+fs://{row['artifact_id']}",
            ),
            run_id=row["run_id"],
            stage=row["stage"],
            artifact_type=row["artifact_type"],
            mime_type=row["mime_type"],
            metadata=metadata,
            tags=manifest.get("tags"),
            created_at=row["created_at"],
            manifest=manifest,
            storage=storage,
        )

    def _record_memory_event(self, record: ArtifactRecord) -> None:
        payload = {
            "artifact_id": record.artifact_id,
            "artifact_uri": record.artifact_uri,
            "storage": record.storage.backend,
            "stage": record.stage,
        }
        try:
            from sparkle_motion import adk_helpers as _adk_helpers  # local import to avoid cycle
            from sparkle_motion.adk_helpers import MemoryWriteError as _MemoryWriteError
        except ImportError:  # pragma: no cover - occurs during partial initialization
            return

        try:
            _adk_helpers.write_memory_event(
                run_id=record.run_id,
                event_type="filesystem_artifact_saved",
                payload=payload,
            )
        except _MemoryWriteError:
            pass


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension_for(mime_type: str, filename_hint: str | None) -> str:
    if filename_hint:
        suffix = Path(filename_hint).suffix
        if suffix:
            return suffix
    guessed = mimetypes.guess_extension(mime_type)
    return guessed or ".bin"
