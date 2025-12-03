"""Colab preflight helpers for the notebook control surface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Sequence
import argparse
import os
import shutil
import subprocess
import sys

import httpx

Status = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != "error"


DEFAULT_ENV_VARS: Sequence[str] = (
    "SPARKLE_DB_PATH",
    "ARTIFACTS_DIR",
    "GOOGLE_ADK_PROFILE",
)


def run_preflight_checks(
    *,
    requirements_path: Path,
    mount_point: Path,
    workspace_dir: Path,
    required_env: Sequence[str] = DEFAULT_ENV_VARS,
    ready_endpoints: Sequence[str] = (),
    pip_mode: Literal["install", "skip"] = "install",
    min_free_gb: float = 15.0,
    require_drive: bool = True,
    skip_gpu_checks: bool = False,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(_check_env_vars(required_env))
    results.append(_handle_requirements(requirements_path, pip_mode))
    results.append(_check_drive_mount(mount_point, workspace_dir, require_drive))
    results.append(_check_disk_space(workspace_dir, min_free_gb))
    if skip_gpu_checks:
        results.append(CheckResult("gpu", "warning", "Skipped GPU check per flag."))
    else:
        results.append(_check_gpu_available())
    results.append(_probe_ready_endpoints(ready_endpoints))
    return results


def format_report(results: Sequence[CheckResult]) -> str:
    lines = []
    for result in results:
        lines.append(f"[{result.status.upper():7}] {result.name}: {result.detail}")
    return "\n".join(lines)


def _check_env_vars(required_env: Sequence[str]) -> CheckResult:
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        return CheckResult("env", "error", f"Missing env vars: {', '.join(missing)}")
    return CheckResult("env", "ok", "All required env vars present.")


def _handle_requirements(path: Path, mode: Literal["install", "skip"]) -> CheckResult:
    path = path.expanduser()
    if not path.exists():
        return CheckResult("pip", "error", f"requirements file not found: {path}")
    if mode == "skip":
        return CheckResult("pip", "warning", "Skipped pip install per flag.")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(path)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        return CheckResult("pip", "error", f"pip install failed: {exc}")
    return CheckResult("pip", "ok", "Dependencies installed via requirements-ml.txt.")


def _running_in_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG"))


def _check_drive_mount(mount_point: Path, workspace_dir: Path, require_drive: bool) -> CheckResult:
    if not require_drive:
        return CheckResult("drive", "warning", "Drive mount not required (flag disabled).")
    if not _running_in_colab():
        return CheckResult("drive", "warning", "Not running inside Colab; skipping Drive mount check.")
    mount_point = mount_point.expanduser()
    if not os.path.ismount(mount_point):
        return CheckResult("drive", "error", f"Google Drive not mounted at {mount_point}.")
    workspace_dir = workspace_dir.expanduser()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return CheckResult("drive", "ok", f"Drive mounted and workspace ready at {workspace_dir}.")


def _check_disk_space(path: Path, min_free_gb: float) -> CheckResult:
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        return CheckResult("disk", "warning", f"Only {free_gb:.1f} GiB free in {path}; recommend >= {min_free_gb} GiB.")
    return CheckResult("disk", "ok", f"{free_gb:.1f} GiB free at {path}.")


def _check_gpu_available() -> CheckResult:
    if shutil.which("nvidia-smi") is None:
        return CheckResult("gpu", "warning", "nvidia-smi not found; GPU may be missing.")
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        return CheckResult("gpu", "error", f"nvidia-smi failed: {exc.stderr or exc}" )
    return CheckResult("gpu", "ok", "nvidia-smi detected; GPU is attached.")


def _probe_ready_endpoints(endpoints: Sequence[str]) -> CheckResult:
    if not endpoints:
        return CheckResult("ready", "warning", "No /ready endpoints provided; skipping probe.")
    client = httpx.Client(timeout=5.0)
    try:
        for endpoint in endpoints:
            response = client.get(endpoint)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return CheckResult("ready", "error", f"/ready probe failed: {exc}")
    finally:
        client.close()
    return CheckResult("ready", "ok", f"Probed {len(endpoints)} /ready endpoints.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run notebook preflight checks.")
    parser.add_argument("--requirements-path", default="requirements-ml.txt", type=Path)
    parser.add_argument("--mount-point", default="/content/drive", type=Path)
    parser.add_argument("--workspace-name", default="SparkleMotion")
    parser.add_argument("--env", dest="env_vars", action="append", default=list(DEFAULT_ENV_VARS))
    parser.add_argument("--ready-endpoint", action="append", default=[])
    parser.add_argument("--pip-mode", choices=["install", "skip"], default="install")
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--no-drive", dest="require_drive", action="store_false")
    parser.add_argument("--skip-gpu-checks", action="store_true")
    args = parser.parse_args(argv)

    workspace_dir = args.mount_point.expanduser() / "MyDrive" / args.workspace_name
    results = run_preflight_checks(
        requirements_path=args.requirements_path,
        mount_point=args.mount_point,
        workspace_dir=workspace_dir,
        required_env=tuple(dict.fromkeys(args.env_vars)),
        ready_endpoints=args.ready_endpoint,
        pip_mode=args.pip_mode,
        min_free_gb=args.min_free_gb,
        require_drive=args.require_drive,
        skip_gpu_checks=args.skip_gpu_checks,
    )
    print(format_report(results))
    if any(result.status == "error" for result in results):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
