from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from sparkle_motion.db.sqlite import ensure_schema, get_conn
from sparkle_motion.utils.recent_index_sqlite import RecentIndexSqlite


def _ts_to_iso(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()


def _row_to_dict(row: Mapping[str, Any]) -> MutableMapping[str, Any]:
    payload = {
        "id": row["id"],
        "phash": row["phash"],
        "canonical_uri": row["canonical_uri"],
        "last_seen": row["last_seen"],
        "hit_count": row["hit_count"],
    }
    payload["last_seen_iso"] = _ts_to_iso(row["last_seen"])
    return payload


def _open_db(path: Optional[str]) -> Any:
    conn = get_conn(path)
    ensure_schema(conn)
    return conn


def _cmd_stats(conn: Any, *, as_json: bool) -> int:
    row = conn.execute(
        "SELECT COUNT(1) AS entries, COALESCE(SUM(hit_count), 0) AS hits, MIN(last_seen) AS oldest, MAX(last_seen) AS newest FROM recent_index"
    ).fetchone()
    data = {
        "entries": row["entries"] if row else 0,
        "hits": row["hits"] if row else 0,
        "oldest": row["oldest"] if row else None,
        "newest": row["newest"] if row else None,
        "oldest_iso": _ts_to_iso(row["oldest"] if row else None),
        "newest_iso": _ts_to_iso(row["newest"] if row else None),
    }
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"entries:\t{data['entries']}")
        print(f"hits:\t{data['hits']}")
        print(f"oldest:\t{data['oldest_iso'] or 'n/a'}")
        print(f"newest:\t{data['newest_iso'] or 'n/a'}")
    return 0


def _cmd_list(conn: Any, *, limit: int, order_by: str, descending: bool, as_json: bool) -> int:
    field = order_by if order_by in {"last_seen", "hit_count", "id"} else "last_seen"
    direction = "DESC" if descending else "ASC"
    rows = conn.execute(
        f"SELECT id, phash, canonical_uri, last_seen, hit_count FROM recent_index ORDER BY {field} {direction} LIMIT ?",
        (limit,),
    ).fetchall()
    data = [_row_to_dict(row) for row in rows]
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        for entry in data:
            print(
                f"{entry['phash']}\t{entry['hit_count']}\t{entry['last_seen_iso']}\t{entry['canonical_uri']}"
            )
    return 0


def _cmd_show(conn: Any, *, phash: str, as_json: bool) -> int:
    row = conn.execute(
        "SELECT id, phash, canonical_uri, last_seen, hit_count FROM recent_index WHERE phash = ?",
        (phash,),
    ).fetchone()
    if row is None:
        print(f"No entry for {phash}", file=sys.stderr)
        return 1
    data = _row_to_dict(row)
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        for key, value in data.items():
            print(f"{key}:\t{value}")
    return 0


def _cmd_prune(db_path: Optional[str], *, max_age: Optional[int], max_entries: Optional[int]) -> int:
    with RecentIndexSqlite(db_path) as recent:
        recent.prune(max_age_s=max_age, max_entries=max_entries)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or prune the Sparkle Motion recent-index store")
    parser.add_argument("--db", dest="db_path", help="Path to sqlite DB (defaults to SPARKLE_DB_PATH)")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="Show aggregate stats")
    stats.add_argument("--json", action="store_true", help="Emit JSON")

    ls = sub.add_parser("list", help="List entries")
    ls.add_argument("--limit", type=int, default=20, help="Maximum rows to display")
    ls.add_argument("--order", choices=["last_seen", "hit_count", "id"], default="last_seen")
    ls.add_argument("--desc", action="store_true", help="Sort descending")
    ls.add_argument("--json", action="store_true")

    show = sub.add_parser("show", help="Show a specific hash")
    show.add_argument("phash", help="Hash to display")
    show.add_argument("--json", action="store_true")

    prune = sub.add_parser("prune", help="Prune entries by age/count")
    prune.add_argument("--max-age", type=int, help="Delete entries older than this many seconds")
    prune.add_argument("--max-entries", type=int, help="Maximum entries to keep (default 10000)")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "stats":
        conn = _open_db(args.db_path)
        return _cmd_stats(conn, as_json=args.json)
    if args.command == "list":
        conn = _open_db(args.db_path)
        return _cmd_list(conn, limit=args.limit, order_by=args.order, descending=args.desc, as_json=args.json)
    if args.command == "show":
        conn = _open_db(args.db_path)
        return _cmd_show(conn, phash=args.phash, as_json=args.json)
    if args.command == "prune":
        return _cmd_prune(args.db_path, max_age=args.max_age, max_entries=args.max_entries)
    parser.error("Missing command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
