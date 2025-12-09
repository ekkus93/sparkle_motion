"""Environment helpers shared across runtime modules."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Literal, cast
import os

TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}
ARTIFACTS_BACKEND_DEFAULT = "adk"
ARTIFACTS_BACKEND_FILESYSTEM = "filesystem"
_ARTIFACTS_BACKEND_ALLOWED = {ARTIFACTS_BACKEND_DEFAULT, ARTIFACTS_BACKEND_FILESYSTEM}

def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()

def env_flag(value: str | None, *, default: bool = False) -> bool:
    token = _normalize(value)
    if not token:
        return default
    if token in TRUTHY:
        return True
    if token in FALSY:
        return False
    return default

def fixture_mode_enabled(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
    *,
    default: bool = True,
) -> bool:
    data = env or os.environ
    return env_flag(data.get("ADK_USE_FIXTURE"), default=default)


def resolve_artifacts_backend(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> Literal["adk", "filesystem"]:
    """Return the configured artifacts backend with validation."""

    data = env or os.environ
    backend = (data.get("ARTIFACTS_BACKEND") or ARTIFACTS_BACKEND_DEFAULT).strip().lower()
    if backend not in _ARTIFACTS_BACKEND_ALLOWED:
        allowed = ", ".join(sorted(_ARTIFACTS_BACKEND_ALLOWED))
        raise ValueError(f"Unsupported ARTIFACTS_BACKEND '{backend}'. Expected one of: {allowed}.")
    return cast(Literal["adk", "filesystem"], backend)


def filesystem_backend_enabled(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> bool:
    """Convenience helper for checking whether the filesystem shim is active."""

    return resolve_artifacts_backend(env) == ARTIFACTS_BACKEND_FILESYSTEM

__all__ = [
    "ARTIFACTS_BACKEND_DEFAULT",
    "ARTIFACTS_BACKEND_FILESYSTEM",
    "TRUTHY",
    "FALSY",
    "env_flag",
    "fixture_mode_enabled",
    "filesystem_backend_enabled",
    "resolve_artifacts_backend",
]
