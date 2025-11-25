from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence


@dataclass(frozen=True)
class WorkspaceLayout:
    """Paths for the canonical Colab/Drive workspace directories."""

    root: Path
    models: Path
    assets: Path
    outputs: Path
    runs: Path
    logs: Path


DEFAULT_SUBDIRS: Sequence[str] = ("models", "assets", "outputs", "runs", "logs")


def ensure_workspace(root: Path, subdirs: Optional[Iterable[str]] = None) -> WorkspaceLayout:
    """Create canonical workspace directories under ``root`` and return their paths."""

    root.mkdir(parents=True, exist_ok=True)
    names = list(subdirs) if subdirs is not None else list(DEFAULT_SUBDIRS)
    paths = {}
    for name in names:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        paths[name] = folder

    # fill in any missing defaults so WorkspaceLayout is always complete
    for name in DEFAULT_SUBDIRS:
        paths.setdefault(name, root / name)

    return WorkspaceLayout(
        root=root,
        models=paths["models"],
        assets=paths["assets"],
        outputs=paths["outputs"],
        runs=paths["runs"],
        logs=paths["logs"],
    )


def download_model(
    repo_id: str,
    target_dir: Path,
    *,
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
    downloader: Optional[Callable[..., str]] = None,
) -> Path:
    """Download a Hugging Face model snapshot into ``target_dir``.

    The ``downloader`` argument allows dependency injection for tests. When omitted,
    :func:`huggingface_hub.snapshot_download` is used.
    """

    target_dir.mkdir(parents=True, exist_ok=True)

    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - exercised in Colab only
            raise RuntimeError(
                "huggingface_hub is required to download models. Install requirements-ml.txt first."
            ) from exc
        downloader = snapshot_download

    snapshot_path = downloader(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(target_dir),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
    )
    return Path(snapshot_path)


def run_smoke_check(workspace: WorkspaceLayout, *, message: str = "sparkle-motion ready") -> Path:
    """Write a lightweight smoke artifact to confirm Drive permissions."""

    workspace.outputs.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {"ok": True, "timestamp": timestamp, "message": message}
    smoke_path = workspace.outputs / "colab_smoke.json"
    smoke_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return smoke_path


__all__ = ["WorkspaceLayout", "ensure_workspace", "download_model", "run_smoke_check"]
