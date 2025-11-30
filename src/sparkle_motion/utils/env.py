from __future__ import annotations

"""Environment helpers shared across runtime modules."""

from typing import Mapping, MutableMapping
import os

TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}

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

__all__ = ["fixture_mode_enabled", "env_flag", "TRUTHY", "FALSY"]
