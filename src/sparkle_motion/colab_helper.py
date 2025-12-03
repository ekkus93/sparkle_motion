from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

_DIFFUSERS_ALLOW_PATTERNS: tuple[str, ...] = (
    "**/*.safetensors",
    "**/*.bin",
    "**/*.pt",
    "**/*.json",
    "**/*.yaml",
    "**/*.txt",
    "**/*.py",
    "**/*.ckpt",
    "**/*.onnx",
    "**/*.model",
)

_REPO_ALLOW_PATTERNS: dict[str, tuple[str, ...]] = {
    "stabilityai/stable-diffusion-xl-base-1.0": _DIFFUSERS_ALLOW_PATTERNS,
    "stabilityai/stable-diffusion-xl-refiner-1.0": _DIFFUSERS_ALLOW_PATTERNS,
    "Wan-AI/Wan2.1-I2V-14B-720P": _DIFFUSERS_ALLOW_PATTERNS,
    "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers": _DIFFUSERS_ALLOW_PATTERNS,
    "ResembleAI/chatterbox": _DIFFUSERS_ALLOW_PATTERNS,
}


@dataclass(frozen=True)
class WorkspaceLayout:
    """Normalized set of directories created under the Drive workspace."""

    root: Path
    models: Path
    assets: Path
    outputs: Path


def ensure_workspace(root: Path) -> WorkspaceLayout:
    """Create the workspace folder structure and return the layout."""

    models_dir = root / "models"
    assets_dir = root / "assets"
    outputs_dir = root / "outputs"

    for path in (root, models_dir, assets_dir, outputs_dir):
        path.mkdir(parents=True, exist_ok=True)

    return WorkspaceLayout(root=root, models=models_dir, assets=assets_dir, outputs=outputs_dir)


def download_model(
    *,
    repo_id: str,
    target_dir: Path,
    revision: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
) -> Path:
    """Download (or update) a Hugging Face repo into ``target_dir``."""

    target_dir.mkdir(parents=True, exist_ok=True)
    class _RepoTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("desc", repo_id)
            super().__init__(*args, **kwargs)

    if allow_patterns is not None:
        effective_allow_patterns = allow_patterns
    else:
        effective_allow_patterns = _REPO_ALLOW_PATTERNS.get(repo_id)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        allow_patterns=list(effective_allow_patterns) if effective_allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        tqdm_class=_RepoTqdm,
    )
    return target_dir


def run_smoke_check(layout: WorkspaceLayout, *, models: Sequence[str] | None = None) -> Path:
    """Write a simple JSON artifact describing the workspace state."""

    models = models or []
    model_entries = [
        _summarize_model(layout.models / repo.replace("/", "__"), repo) for repo in models
    ]
    ok = all(entry["status"] == "present" for entry in model_entries) if model_entries else True
    payload = {
        "ok": ok,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "workspace": {
            "root": str(layout.root),
            "models": str(layout.models),
            "assets": str(layout.assets),
            "outputs": str(layout.outputs),
        },
        "models": model_entries,
    }
    smoke_path = layout.outputs / "colab_smoke.json"
    smoke_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return smoke_path


def _summarize_model(model_dir: Path, repo_id: str) -> dict[str, object]:
    if not model_dir.exists():
        return {
            "repo_id": repo_id,
            "status": "missing",
            "files_present": 0,
            "total_bytes": 0,
            "sample_file": None,
        }

    files = [p for p in model_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    sample_file = str(files[0]) if files else None
    return {
        "repo_id": repo_id,
        "status": "present",
        "files_present": len(files),
        "total_bytes": total_bytes,
        "sample_file": sample_file,
    }
