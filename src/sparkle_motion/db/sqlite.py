from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


def get_conn(path: Optional[str] = None) -> sqlite3.Connection:
    """Return a sqlite3.Connection for the given path.

    If `path` is None the function will consult the `SPARKLE_DB_PATH`
    environment variable and fall back to `./artifacts/sparkle.db`.
    The caller is responsible for calling `ensure_schema` once.
    """
    import os

    if path is None:
        path = os.environ.get("SPARKLE_DB_PATH", "./artifacts/sparkle.db")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # reasonable timeout for simple single-user workflows
    conn = sqlite3.connect(str(p), timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection, ddl: Optional[str] = None) -> None:
    """Ensure the DB schema exists.

    If `ddl` is None the default schema file `db/schema/recent_index.sql`
    will be loaded from the repo root.
    """
    if ddl is None:
        repo_root = Path(__file__).resolve().parents[3]
        p = repo_root / "db" / "schema" / "recent_index.sql"
        ddl = p.read_text(encoding="utf-8")
    conn.executescript(ddl)
    conn.commit()
