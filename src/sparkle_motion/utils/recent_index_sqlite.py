from __future__ import annotations

import time
import sqlite3
from typing import Optional

from ..db.sqlite import get_conn, ensure_schema


DEFAULT_MAX_ENTRIES = 10000


class RecentIndexSqlite:
    """A lightweight SQLite-backed RecentIndex for dedupe canonicalization.

    This class is intentionally small and suitable for single-user workflows.
    It provides get/add/touch/prune semantics used by `images_agent`.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._conn = get_conn(db_path)
        ensure_schema(self._conn)

    def get_canonical(self, phash: str) -> Optional[str]:
        cur = self._conn.execute("SELECT canonical_uri FROM recent_index WHERE phash = ?", (phash,))
        row = cur.fetchone()
        return None if row is None else row["canonical_uri"]

    def add_or_get(self, phash: str, uri: str) -> str:
        now = int(time.time())
        # try insert; if exists, return existing canonical
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR IGNORE INTO recent_index (phash, canonical_uri, last_seen, hit_count) VALUES (?, ?, ?, 1)",
                    (phash, uri, now),
                )
                # ensure we update last_seen/hit_count if already present
                self._conn.execute(
                    "UPDATE recent_index SET last_seen = ?, hit_count = hit_count + 1 WHERE phash = ?",
                    (now, phash),
                )
        except sqlite3.DatabaseError:
            # fall back to read-only select
            pass
        cur = self._conn.execute("SELECT canonical_uri FROM recent_index WHERE phash = ?", (phash,))
        row = cur.fetchone()
        if row is None:
            # this should be rare; insert directly
            with self._conn:
                self._conn.execute(
                    "INSERT INTO recent_index (phash, canonical_uri, last_seen, hit_count) VALUES (?, ?, ?, 1)",
                    (phash, uri, now),
                )
            return uri
        return row["canonical_uri"]

    def touch(self, phash: str, uri: Optional[str] = None) -> None:
        now = int(time.time())
        if uri is None:
            with self._conn:
                self._conn.execute("UPDATE recent_index SET last_seen = ? WHERE phash = ?", (now, phash))
        else:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO recent_index (phash, canonical_uri, last_seen, hit_count) VALUES (?, ?, ?, COALESCE((SELECT hit_count FROM recent_index WHERE phash = ?),1))",
                    (phash, uri, now, phash),
                )

    def prune(self, max_age_s: Optional[int] = None, max_entries: Optional[int] = None) -> None:
        now = int(time.time())
        max_entries = max_entries if max_entries is not None else DEFAULT_MAX_ENTRIES
        with self._conn:
            if max_age_s is not None:
                cutoff = now - int(max_age_s)
                self._conn.execute("DELETE FROM recent_index WHERE last_seen < ?", (cutoff,))
            # enforce max entries: delete oldest rows beyond max_entries
            cur = self._conn.execute("SELECT COUNT(1) as c FROM recent_index")
            row = cur.fetchone()
            total = row["c"] if row is not None else 0
            if total > max_entries:
                to_delete = total - max_entries
                # delete the oldest entries by last_seen asc
                self._conn.execute(
                    "DELETE FROM recent_index WHERE id IN (SELECT id FROM recent_index ORDER BY last_seen ASC LIMIT ?)",
                    (to_delete, ),
                )
