from __future__ import annotations

import json
import time
import os
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ModelSmokeResult:
    """Outcome of validating a single downloaded model snapshot."""

    repo_id: str
    status: str
    model_path: str
    files_present: int
    bytes_total: int
    sample_file: Optional[str] = None


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


def _sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _model_dir(workspace: WorkspaceLayout, repo_id: str) -> Path:
    return workspace.models / _sanitize_repo_id(repo_id)


def _dir_file_stats(path: Path, *, limit: int = 1024) -> tuple[int, int, Optional[Path]]:
    files = 0
    total = 0
    sample: Optional[Path] = None
    for root, _, filenames in os.walk(path):
        for name in filenames:
            files += 1
            fp = Path(root) / name
            if sample is None:
                sample = fp
            try:
                total += fp.stat().st_size
            except OSError:
                pass
            if files >= limit:
                return files, total, sample
    return files, total, sample


def collect_model_smoke_checks(workspace: WorkspaceLayout, models: Sequence[str]) -> list[ModelSmokeResult]:
    results: list[ModelSmokeResult] = []
    for repo_id in models:
        model_path = _model_dir(workspace, repo_id)
        if not model_path.exists():
            results.append(
                ModelSmokeResult(
                    repo_id=repo_id,
                    status="missing",
                    model_path=str(model_path),
                    files_present=0,
                    bytes_total=0,
                    sample_file=None,
                )
            )
            continue
        file_count, total_bytes, sample = _dir_file_stats(model_path)
        status = "ok" if file_count > 0 else "empty"
        results.append(
            ModelSmokeResult(
                repo_id=repo_id,
                status=status,
                model_path=str(model_path),
                files_present=file_count,
                bytes_total=total_bytes,
                sample_file=str(sample) if sample else None,
            )
        )
    return results


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


def run_smoke_check(
    workspace: WorkspaceLayout,
    *,
    message: str = "sparkle-motion ready",
    models: Optional[Sequence[str]] = None,
) -> Path:
    """Write a lightweight smoke artifact including per-model validation results."""

    workspace.outputs.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {"ok": True, "timestamp": timestamp, "message": message}
    model_results = collect_model_smoke_checks(workspace, models or []) if models else []
    if model_results:
        payload["models"] = [asdict(result) for result in model_results]
        payload["ok"] = payload["ok"] and all(result.status == "ok" for result in model_results)
    smoke_path = workspace.outputs / "colab_smoke.json"
    smoke_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return smoke_path

__all__ = [
    "WorkspaceLayout",
    "ModelSmokeResult",
    "ensure_workspace",
    "download_model",
    "collect_model_smoke_checks",
    "run_smoke_check",
]
