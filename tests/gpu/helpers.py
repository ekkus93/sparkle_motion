from __future__ import annotations

import hashlib
import os
import subprocess
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

MonkeyPatch = Any

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional on CPU-only hosts
    torch = None  # type: ignore

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "assets"
GPU_FAIL_REASON = "GPU tests require CUDA-enabled hardware (set CUDA_VISIBLE_DEVICES accordingly)."


def asset_path(name: str) -> Path:
    path = ASSETS_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"missing GPU test asset: {path}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pytest_module() -> Any:
    try:
        return importlib.import_module("pytest")
    except ImportError as exc:  # pragma: no cover - pytest always available in test envs
        raise RuntimeError("pytest is required to run GPU tests") from exc


def require_gpu_available() -> None:
    pytest = _pytest_module()
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.fail(GPU_FAIL_REASON)
    if torch is None:
        pytest.fail(GPU_FAIL_REASON)
    if not torch.cuda.is_available():
        pytest.fail(GPU_FAIL_REASON)


def set_env(monkeypatch: MonkeyPatch, values: Mapping[str, str]) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def unset_env(monkeypatch: MonkeyPatch, keys: Iterable[str]) -> None:
    for key in keys:
        monkeypatch.delenv(key, raising=False)


@contextmanager
def temp_output_dir(monkeypatch: MonkeyPatch, env_key: str, path: Path) -> Iterator[Path]:
    set_env(monkeypatch, {env_key: str(path)})
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        # leave artifacts for investigation; caller is responsible for cleanup if needed
        pass


def ensure_real_adapter(monkeypatch: MonkeyPatch, *, flags: Sequence[str], disable_keys: Sequence[str] = ()) -> None:
    mapping: Dict[str, str] = {flag: "1" for flag in flags}
    set_env(monkeypatch, mapping)
    unset_env(monkeypatch, disable_keys)


def probe_video(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
