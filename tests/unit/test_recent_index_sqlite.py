from __future__ import annotations

import os
import time
import sqlite3
from pathlib import Path

import pytest

from sparkle_motion.db.sqlite import get_conn, ensure_schema
from sparkle_motion.utils.recent_index_sqlite import RecentIndexSqlite


def test_add_and_get_and_prune(tmp_path: Path):
    db_path = str(tmp_path / "test_sparkle.db")
    conn = get_conn(db_path)
    # load schema from repo file
    repo_root = Path(__file__).resolve().parents[2]
    ddl_path = repo_root / "db" / "schema" / "recent_index.sql"
    ensure_schema(conn, ddl_path.read_text(encoding="utf-8"))

    ri = RecentIndexSqlite(db_path)

    phash = "phash1"
    uri1 = "file://artifact/1"
    uri2 = "file://artifact/2"

    # add and get
    got = ri.add_or_get(phash, uri1)
    assert got == uri1
    assert ri.get_canonical(phash) == uri1

    # adding with different uri should return original canonical
    got2 = ri.add_or_get(phash, uri2)
    assert got2 == uri1

    # touch updates last_seen (no exception)
    ri.touch(phash)

    # simulate old entry and prune by age
    old_ts = int(time.time()) - 3600 * 24 * 7
    with get_conn(db_path) as c:
        c.execute("UPDATE recent_index SET last_seen = ? WHERE phash = ?", (old_ts, phash))
        c.commit()

    ri.prune(max_age_s=60)  # should delete the old row
    assert ri.get_canonical(phash) is None


def test_prune_max_entries(tmp_path: Path):
    db_path = str(tmp_path / "test_sparkle2.db")
    conn = get_conn(db_path)
    repo_root = Path(__file__).resolve().parents[2]
    ddl_path = repo_root / "db" / "schema" / "recent_index.sql"
    ensure_schema(conn, ddl_path.read_text(encoding="utf-8"))

    ri = RecentIndexSqlite(db_path)

    # insert 5 entries
    for i in range(5):
        ri.add_or_get(f"p{i}", f"file://a{i}")

    # prune to max_entries=3
    ri.prune(max_age_s=None, max_entries=3)

    # count rows
    with get_conn(db_path) as c:
        cur = c.execute("SELECT COUNT(1) as c FROM recent_index")
        row = cur.fetchone()
        assert row is not None and row[0] == 3
