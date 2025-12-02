from __future__ import annotations

"""Configuration helpers for the filesystem ArtifactService shim."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from sparkle_motion.utils.env import env_flag

_DEFAULT_MAX_BYTES = 12 * 1024 * 1024 * 1024  # 12 GiB ceiling keeps local disks safe


@dataclass(frozen=True)
class FilesystemArtifactsConfig:
    """Resolved configuration for the shim service."""

    root: Path
    index_path: Path
    base_url: str
    token: str | None
    allow_insecure: bool
    max_payload_bytes: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FilesystemArtifactsConfig:
        data = env or {}
        root_hint = data.get("ARTIFACTS_FS_ROOT") or "./artifacts_fs"
        root = Path(root_hint).expanduser().resolve()
        index_hint = data.get("ARTIFACTS_FS_INDEX") or str(root / "index.db")
        index_path = Path(index_hint).expanduser().resolve()
        base_url = data.get("ARTIFACTS_FS_BASE_URL") or "http://127.0.0.1:7077"
        token = data.get("ARTIFACTS_FS_TOKEN") or None
        allow_insecure = env_flag(data.get("ARTIFACTS_FS_ALLOW_INSECURE"), default=False)
        max_payload_bytes = _coerce_int(data.get("ARTIFACTS_FS_MAX_BYTES"), default=_DEFAULT_MAX_BYTES)
        if max_payload_bytes <= 0:
            raise ValueError("ARTIFACTS_FS_MAX_BYTES must be greater than zero")
        if not allow_insecure and (token is None or not token.strip()):
            raise ValueError(
                "ARTIFACTS_FS_TOKEN is required unless ARTIFACTS_FS_ALLOW_INSECURE=1"
            )
        root.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            index_path=index_path,
            base_url=base_url.rstrip("/"),
            token=token.strip() if token else None,
            allow_insecure=allow_insecure,
            max_payload_bytes=max_payload_bytes,
        )


def _coerce_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid integer value: {value!r}") from exc
