from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from sparkle_motion.db.sqlite import ensure_schema, get_conn
from sparkle_motion.utils import recent_index_cli


def _seed_db(path: Path) -> None:
    conn = get_conn(str(path))
    ensure_schema(conn)
    now = int(time.time())
    rows = [
        ("aaaa", "artifact://clip-a", now - 10, 2),
        ("bbbb", "artifact://clip-b", now, 4),
        ("aaab", "artifact://clip-c", now - 5, 1),
    ]
    with conn:
        conn.executemany(
            "INSERT INTO recent_index (phash, canonical_uri, last_seen, hit_count) VALUES (?, ?, ?, ?)",
            rows,
        )
    conn.close()


def test_stats_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "recent.db"
    _seed_db(db_path)

    code = recent_index_cli.main(["--db", str(db_path), "stats", "--json"])
    assert code == 0
    stats_out = capsys.readouterr().out
    payload = json.loads(stats_out)
    assert payload["entries"] == 3
    assert payload["hits"] == 7

    code = recent_index_cli.main(["--db", str(db_path), "list", "--limit", "1"])
    assert code == 0
    list_out = capsys.readouterr().out
    assert "artifact://clip" in list_out


def test_show_and_prune(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "recent.db"
    _seed_db(db_path)

    code = recent_index_cli.main(["--db", str(db_path), "show", "aaaa", "--json"])
    assert code == 0
    show_out = capsys.readouterr().out
    record = json.loads(show_out)
    assert record["canonical_uri"].endswith("clip-a")

    code = recent_index_cli.main(["--db", str(db_path), "show", "zzzz"])
    assert code == 1
    err = capsys.readouterr().err
    assert "No entry" in err

    code = recent_index_cli.main(["--db", str(db_path), "prune", "--max-entries", "1"])
    assert code == 0
    conn = get_conn(str(db_path))
    cur = conn.execute("SELECT COUNT(1) FROM recent_index")
    remaining = cur.fetchone()[0]
    assert remaining == 1
    conn.close()


def test_near_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "recent.db"
    _seed_db(db_path)

    code = recent_index_cli.main(
        [
            "--db",
            str(db_path),
            "near",
            "aaaa",
            "--max-distance",
            "2",
            "--limit",
            "2",
            "--json",
        ]
    )
    assert code == 0
    near_out = capsys.readouterr().out
    entries = json.loads(near_out)
    assert entries[0]["phash"] == "aaaa"
    assert entries[0]["distance"] == 0
    assert any(entry["distance"] == 1 for entry in entries)
