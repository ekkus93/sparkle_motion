from __future__ import annotations

"""Command-line interface for filesystem artifact utilities."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import shlex
import shutil
import sys
from typing import Optional, Sequence

import httpx
import uvicorn

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
from .app import create_app


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

    env_cmd = sub.add_parser("env", help="Print environment exports for the filesystem backend")
    _add_config_overrides(env_cmd)
    env_cmd.add_argument(
        "--shell",
        choices=("bash", "powershell"),
        default="bash",
        help="Format exports for the selected shell (default: bash)",
    )
    env_cmd.add_argument(
        "--emit-token",
        action="store_true",
        help="Include ARTIFACTS_FS_TOKEN in the output (token must be provided via env or --token)",
    )

    serve_cmd = sub.add_parser("serve", help="Launch the filesystem ArtifactService shim via uvicorn")
    _add_config_overrides(serve_cmd)
    serve_cmd.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    serve_cmd.add_argument("--port", type=int, default=7077, help="Port to bind (default: 7077)")
    serve_cmd.add_argument("--log-level", default="info", help="uvicorn log level (default: info)")
    serve_cmd.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn autoreload (development only)",
    )

    health_cmd = sub.add_parser("health", help="Probe the shim /healthz endpoint")
    _add_config_overrides(health_cmd)
    health_cmd.add_argument(
        "--url",
        help="Override health endpoint URL (defaults to {base_url}/healthz)",
    )
    health_cmd.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Request timeout in seconds (default: 5.0)",
    )
    return parser


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", help="Override ARTIFACTS_FS_ROOT")
    parser.add_argument("--index", help="Override ARTIFACTS_FS_INDEX")
    parser.add_argument("--base-url", help="Override ARTIFACTS_FS_BASE_URL")
    parser.add_argument("--token", help="Override ARTIFACTS_FS_TOKEN")
    parser.add_argument(
        "--allow-insecure",
        action="store_true",
        help="Allow the shim to run without ARTIFACTS_FS_TOKEN",
    )


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


def run_env(parsed_args: argparse.Namespace) -> int:
    env = _apply_overrides(parsed_args, require_backend=True)
    config = FilesystemArtifactsConfig.from_env(env)
    exports = {
        "ARTIFACTS_BACKEND": ARTIFACTS_BACKEND_FILESYSTEM,
        "ARTIFACTS_FS_ROOT": str(config.root),
        "ARTIFACTS_FS_INDEX": str(config.index_path),
        "ARTIFACTS_FS_BASE_URL": config.base_url,
        "ARTIFACTS_FS_MAX_BYTES": str(config.max_payload_bytes),
    }
    token = config.token if parsed_args.emit_token else None
    if token:
        exports["ARTIFACTS_FS_TOKEN"] = token
    elif parsed_args.emit_token:
        print("ARTIFACTS_FS_TOKEN is not set. Provide --token or export the variable before using --emit-token.", file=sys.stderr)
    formatter = _format_export_bash if parsed_args.shell == "bash" else _format_export_powershell
    for key, value in exports.items():
        print(formatter(key, value))
    if not parsed_args.emit_token:
        print("# Token omitted; run with --emit-token if you want to include ARTIFACTS_FS_TOKEN in the output.")
    return 0


def run_serve(parsed_args: argparse.Namespace) -> int:
    env = _apply_overrides(
        parsed_args,
        require_backend=False,
        default_base_url=f"http://{parsed_args.host}:{parsed_args.port}",
    )
    config = FilesystemArtifactsConfig.from_env(env)
    app = create_app(config)
    uvicorn.run(
        app,
        host=parsed_args.host,
        port=parsed_args.port,
        log_level=parsed_args.log_level,
        reload=parsed_args.reload,
    )
    return 0


def run_health(parsed_args: argparse.Namespace) -> int:
    env = _apply_overrides(parsed_args, require_backend=False)
    config = FilesystemArtifactsConfig.from_env(env)
    url = parsed_args.url or f"{config.base_url.rstrip('/')}/healthz"
    headers = {}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    try:
        response = httpx.get(url, headers=headers, timeout=parsed_args.timeout)
    except httpx.HTTPError as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 3
    if response.status_code == 200:
        print(f"Filesystem backend healthy → {url}")
        try:
            payload = response.json()
            print(payload)
        except ValueError:
            if response.text:
                print(response.text)
        return 0
    print(f"Filesystem backend returned {response.status_code}: {response.text}", file=sys.stderr)
    return 4


def _apply_overrides(
    args: argparse.Namespace,
    *,
    require_backend: bool,
    default_base_url: Optional[str] = None,
) -> dict[str, str]:
    env = dict(os.environ)
    if require_backend or args.backend:
        env["ARTIFACTS_BACKEND"] = args.backend or ARTIFACTS_BACKEND_FILESYSTEM
    if args.root:
        env["ARTIFACTS_FS_ROOT"] = args.root
    if args.index:
        env["ARTIFACTS_FS_INDEX"] = args.index
    base_url = args.base_url or default_base_url
    if base_url:
        env["ARTIFACTS_FS_BASE_URL"] = base_url
    if args.token:
        env["ARTIFACTS_FS_TOKEN"] = args.token
    if args.allow_insecure:
        env["ARTIFACTS_FS_ALLOW_INSECURE"] = "1"
    return env


def _format_export_bash(key: str, value: str) -> str:
    return f"export {key}={shlex.quote(str(value))}"


def _format_export_powershell(key: str, value: str) -> str:
    escaped = str(value).replace('"', '\"')
    return f'$Env:{key} = "{escaped}"'


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prune":
        return run_prune(args)
    if args.command == "env":
        return run_env(args)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "health":
        return run_health(args)
    parser.error("Unknown command")
    return 2


__all__ = [
    "build_parser",
    "main",
    "parse_size",
    "run_prune",
    "run_env",
    "run_serve",
    "run_health",
]
