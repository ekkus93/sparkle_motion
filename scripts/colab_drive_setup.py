#!/usr/bin/env python3
"""Colab helper to mount Drive, create directories, download models, and run a smoke check."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from sparkle_motion.colab_helper import download_model, ensure_workspace, run_smoke_check


def _mount_drive(mount_point: str) -> Path:
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:  # pragma: no cover - only runs inside Colab
        raise RuntimeError("google.colab is only available inside a Colab runtime") from exc

    drive.mount(mount_point, force_remount=False)
    return Path(mount_point)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare Google Drive workspace for Sparkle Motion runs")
    parser.add_argument("workspace", help="Name of the folder to create under MyDrive", default="SparkleMotion")
    parser.add_argument(
        "--mount-point",
        default="/content/drive",
        help="Path where Google Drive will be mounted",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face repo id to download (e.g., stabilityai/stable-diffusion-xl-base-1.0)",
    )
    parser.add_argument("--revision", help="Specific repo revision or commit", default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip model download (still prepares Drive directories)",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip writing the smoke test artifact",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional allow_patterns entries for snapshot_download",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional ignore_patterns entries for snapshot_download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without performing downloads",
    )
    args = parser.parse_args(argv)

    drive_root = _mount_drive(args.mount_point)
    workspace_root = drive_root / "MyDrive" / args.workspace
    layout = ensure_workspace(workspace_root)

    print(f"[colab] Workspace ready under {layout.root}")
    print(f"[colab] models -> {layout.models}")
    print(f"[colab] assets -> {layout.assets}")
    print(f"[colab] outputs -> {layout.outputs}")

    if args.repo_id and not (args.no_download or args.dry_run):
        target_dir = layout.models / args.repo_id.replace("/", "__")
        print(f"[colab] Downloading {args.repo_id} into {target_dir}")
        download_model(
            repo_id=args.repo_id,
            target_dir=target_dir,
            revision=args.revision,
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
        )
    elif args.repo_id:
        print("[colab] (dry-run) Would download", args.repo_id)

    if not args.no_smoke and not args.dry_run:
        smoke_path = run_smoke_check(layout)
        print(f"[colab] Smoke artifact written to {smoke_path}")
    elif not args.no_smoke:
        print("[colab] (dry-run) Would write smoke artifact")

    print("[colab] Setup complete")


if __name__ == "__main__":
    main()
