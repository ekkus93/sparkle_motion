#!/usr/bin/env python3
"""Install and prepare the Wav2Lip helper assets for GPU lipsync tests."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_URL = "https://github.com/Rudrabha/Wav2Lip.git"
DEFAULT_CHECKPOINT_REPO = "Rudrabha/Wav2Lip"
DEFAULT_CHECKPOINT_FILE = "wav2lip_gan.pth"
DEFAULT_CHECKPOINT_PATH = "checkpoints/wav2lip_gan.pth"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(cmd)
    location = f" (cwd={cwd})" if cwd else ""
    print(f"[wav2lip] $ {printable}{location}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def clone_or_update(repo_url: str, dest: Path, branch: str | None) -> None:
    if dest.exists():
        print(f"[wav2lip] Updating existing repo at {dest}")
        _run(["git", "fetch", "--all"], cwd=dest)
        if branch:
            _run(["git", "checkout", branch], cwd=dest)
            _run(["git", "pull", "--ff-only", "origin", branch], cwd=dest)
        else:
            _run(["git", "pull", "--ff-only"], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = ["git", "clone", repo_url, str(dest)]
        if branch:
            clone_cmd.extend(["--branch", branch])
        _run(clone_cmd)


def install_requirements(repo_dir: Path, *, extra_args: list[str] | None) -> None:
    requirements = repo_dir / "requirements.txt"
    if not requirements.exists():
        print(f"[wav2lip] requirements.txt not found at {requirements}; skipping pip install")
        return
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    if extra_args:
        cmd.extend(extra_args)
    _run(cmd)


def ensure_checkpoint(
    *,
    repo_id: str,
    filename: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[wav2lip] Downloading checkpoint '{filename}' from {repo_id} to {destination}",
    )
    temp_path = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(destination.parent))
    )
    shutil.copy2(temp_path, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the Wav2Lip repo and checkpoint")
    parser.add_argument(
        "--workspace",
        default="artifacts/model_workspace",
        help="Directory where the Wav2Lip repo will live (default: artifacts/model_workspace)",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git URL to clone")
    parser.add_argument("--branch", help="Optional git branch or tag to checkout")
    parser.add_argument(
        "--checkpoint-repo",
        default=DEFAULT_CHECKPOINT_REPO,
        help="Hugging Face repo id containing wav2lip_gan.pth",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=DEFAULT_CHECKPOINT_FILE,
        help="Checkpoint filename inside the Hugging Face repo",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Relative path under the repo where the checkpoint should be stored",
    )
    parser.add_argument(
        "--skip-pip",
        action="store_true",
        help="Skip pip install of Wav2Lip requirements",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip downloading the wav2lip checkpoint",
    )
    parser.add_argument(
        "--pip-extra-arg",
        action="append",
        default=None,
        help="Additional argument forwarded to pip install (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = workspace / "Wav2Lip"

    clone_or_update(args.repo_url, repo_dir, args.branch)

    if not args.skip_pip:
        install_requirements(repo_dir, extra_args=args.pip_extra_arg)
    else:
        print("[wav2lip] Skipping pip install per flag")

    checkpoint_path = repo_dir / args.checkpoint_path
    if not args.skip_checkpoint:
        ensure_checkpoint(
            repo_id=args.checkpoint_repo,
            filename=args.checkpoint_file,
            destination=checkpoint_path,
        )
    else:
        print("[wav2lip] Skipping checkpoint download per flag")

    print("\n[wav2lip] Setup complete")
    print(f"export WAV2LIP_REPO={repo_dir}")
    print(f"export LIPSYNC_WAV2LIP_MODEL={checkpoint_path}")


if __name__ == "__main__":
    main()
