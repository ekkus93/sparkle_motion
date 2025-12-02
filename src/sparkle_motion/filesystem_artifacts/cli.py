from __future__ import annotations

"""Command-line interface for filesystem artifact utilities."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import shutil
import sys
from typing import Optional, Sequence

from sparkle_motion.utils.env import (
    ARTIFACTS_BACKEND_FILESYSTEM,
    filesystem_backend_enabled,
    resolve_artifacts_backend,
)

from .config import FilesystemArtifactsConfig
from .retention import (
    ArtifactRow,
    RetentionOptions,
    RetentionPlan,
    execute_plan,
    load_artifacts,
    plan_retention,
)


@dataclass
class PruneArgs:
    backend: Optional[str]
    root: Optional[str]
    index: Optional[str]
    max_bytes: Optional[int]
    max_age_days: Optional[float]
    min_free_bytes: Optional[int]
    runs: list[str]
    dry_run: bool
    assume_yes: bool


def parse_size(value: str) -> int:
    token = value.strip().lower()
    if not token:
        raise argparse.ArgumentTypeError("size value cannot be empty")
    suffix_map = {
        "k": 1024,
        "m": 1024 ** 2,
        "g": 1024 ** 3,
        "t": 1024 ** 4,
    }
    if token[-1] in suffix_map:
        base = token[:-1]
        multiplier = suffix_map[token[-1]]
    else:
        base = token
        multiplier = 1
    try:
        number = float(base)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid size value: {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("size must be non-negative")
    return int(number * multiplier)


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filesystem artifact utilities")
    parser.add_argument(
        "--backend",
        help="Override ARTIFACTS_BACKEND (defaults to env)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prune = sub.add_parser("prune", help="Prune artifacts by age, size, or free-space targets")
    prune.add_argument("--root", help="Override ARTIFACTS_FS_ROOT")
    prune.add_argument("--index", help="Override ARTIFACTS_FS_INDEX")
    prune.add_argument(
        "--max-bytes",
        type=parse_size,
        help="Target total bytes for the artifact store (e.g., 200g)",
    )
    prune.add_argument(
        "--max-age-days",
        type=float,
        help="Delete artifacts older than this many days",
    )
    prune.add_argument(
        "--min-free-bytes",
        type=parse_size,
        help="Ensure at least this many bytes remain free on the filesystem",
    )
    prune.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=[],
        help="Limit pruning to specific run IDs (repeatable)",
    )
    prune.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Preview deletions without touching disk (default)",
    )
    prune.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Apply deletions instead of previewing",
    )
    prune.add_argument(
        "--yes",
        dest="assume_yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    return parser


def _ensure_backend(backend: Optional[str]) -> None:
    env = dict(os.environ)
    if backend:
        env["ARTIFACTS_BACKEND"] = backend
    try:
        effective = resolve_artifacts_backend(env)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not filesystem_backend_enabled(env):
        raise SystemExit(
            f"Filesystem prune command requires ARTIFACTS_BACKEND={ARTIFACTS_BACKEND_FILESYSTEM} (currently {effective})."
        )


def _resolve_config(args: PruneArgs) -> FilesystemArtifactsConfig:
    env = dict(os.environ)
    if args.backend:
        env["ARTIFACTS_BACKEND"] = args.backend
    if args.root:
        env["ARTIFACTS_FS_ROOT"] = args.root
    if args.index:
        env["ARTIFACTS_FS_INDEX"] = args.index
    return FilesystemArtifactsConfig.from_env(env)


def _format_candidate(candidate: ArtifactRow) -> str:
    timestamp = datetime.fromtimestamp(candidate.created_at, tz=timezone.utc).isoformat()
    return (
        f"{timestamp} run={candidate.run_id} stage={candidate.stage} "
        f"type={candidate.artifact_type} size={human_bytes(candidate.size_bytes)}"
    )


def run_prune(parsed_args: argparse.Namespace) -> int:
    args = PruneArgs(
        backend=parsed_args.backend,
        root=parsed_args.root,
        index=parsed_args.index,
        max_bytes=parsed_args.max_bytes,
        max_age_days=parsed_args.max_age_days,
        min_free_bytes=parsed_args.min_free_bytes,
        runs=parsed_args.runs,
        dry_run=parsed_args.dry_run,
        assume_yes=parsed_args.assume_yes,
    )
    _ensure_backend(args.backend)
    if args.max_age_days is not None and args.max_age_days < 0:
        print("--max-age-days must be non-negative", file=sys.stderr)
        return 2
    if (
        args.max_bytes is None
        and args.max_age_days is None
        and args.min_free_bytes is None
    ):
        print("At least one of --max-bytes, --max-age-days, or --min-free-bytes is required", file=sys.stderr)
        return 2

    config = _resolve_config(args)
    run_filter = set(args.runs) if args.runs else None
    artifacts = load_artifacts(config, runs=run_filter)
    usage = shutil.disk_usage(config.root)
    max_age_seconds = int(args.max_age_days * 86400) if args.max_age_days is not None else None
    options = RetentionOptions(
        max_bytes=args.max_bytes,
        max_age_seconds=max_age_seconds,
        min_free_bytes=args.min_free_bytes,
    )
    plan = plan_retention(artifacts, options, disk_free_bytes=usage.free)

    print(f"Artifacts indexed: {plan.initial_count} ({human_bytes(plan.initial_bytes)})")
    print(f"Disk free: {human_bytes(usage.free)}")
    print(f"Would prune: {plan.freed_count} ({human_bytes(plan.freed_bytes)})")
    print(f"Projected remaining size: {human_bytes(plan.remaining_bytes)}")

    if not plan.candidates:
        print("No artifacts meet the retention criteria.")
        return 0

    print("\nCandidates (oldest first):")
    for candidate in plan.candidates:
        print(f" - [{candidate.reason}] {_format_candidate(candidate.artifact)}")

    if args.dry_run:
        print("\nDry run complete; no files were deleted.")
        return 0

    if not args.assume_yes:
        response = input("Apply retention plan? Type 'yes' to continue: ").strip().lower()
        if response != "yes":
            print("Aborted; no changes were made.")
            return 1

    execute_plan(plan, config, dry_run=False)
    print(
        f"Deleted {plan.freed_count} artifacts, reclaimed {human_bytes(plan.freed_bytes)}."
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prune":
        return run_prune(args)
    parser.error("Unknown command")
    return 2


__all__ = ["build_parser", "main", "parse_size", "run_prune"]
