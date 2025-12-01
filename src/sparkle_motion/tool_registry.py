"""Helpers to load the local `configs/tool_registry.yaml` and expose
convenience accessors for the local-colab profile.

This is intentionally small and dependency-free (uses `yaml` which is already
declared in requirements). Callers should import and use the helpers rather than
hardcoding endpoints in scripts or notebooks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from sparkle_motion import schema_registry


class SchemaResolutionError(RuntimeError):
    """Raised when a tool registry schema entry cannot be resolved."""


def _resolve_schema_value(entry: Any) -> str:
    """Normalize a schema entry to its artifact URI (respects registry helpers)."""

    if isinstance(entry, str):
        return entry

    if isinstance(entry, dict):
        registry_name = (
            entry.get("registry")
            or entry.get("registry_name")
            or entry.get("schema")
            or entry.get("schema_name")
        )
        prefer_local = entry.get("prefer_local")
        if registry_name:
            return schema_registry.resolve_schema_uri(registry_name, prefer_local=prefer_local)

        artifact_uri = entry.get("artifact_uri") or entry.get("uri")
        if artifact_uri:
            return artifact_uri

    raise SchemaResolutionError(f"Unsupported schema entry format: {entry!r}")


def resolve_schema_references(schemas: Mapping[str, Any]) -> Dict[str, str]:
    """Return a copy of *schemas* with every entry coerced to an artifact URI."""

    resolved: Dict[str, str] = {}
    for name, entry in schemas.items():
        if entry is None:
            continue
        resolved[name] = _resolve_schema_value(entry)
    return resolved


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_PATHS = [
    _PACKAGE_ROOT / "configs" / "tool_registry.yaml",
    _REPO_ROOT / "configs" / "tool_registry.yaml",
]


def load_tool_registry(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load and return the tool registry YAML as a dict.

    Args:
        path: optional path to the YAML file. If omitted, the repository default
            `configs/tool_registry.yaml` is used.

    Returns:
        Parsed YAML as a Python dict.

    Raises:
        FileNotFoundError: if the resolved YAML path does not exist.
        yaml.YAMLError: if the YAML fails to parse.
    """
    if path:
        candidate_paths = [Path(path)]
    else:
        candidate_paths = _DEFAULT_PATHS

    for candidate in candidate_paths:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        "Tool registry not found at any of: "
        + ", ".join(str(p) for p in candidate_paths)
    )


def get_local_endpoint(tool_id: str, profile: str = "local-colab") -> Optional[str]:
    """Return the endpoint URL for a tool and profile, or None if missing.

    Example:
        get_local_endpoint("script_agent") -> "http://127.0.0.1:5001/invoke"
    """
    data = load_tool_registry()
    tools = data.get("tools", {}) if isinstance(data, dict) else {}
    tool = tools.get(tool_id)
    if not tool:
        return None
    endpoints = tool.get("endpoints", {})
    return endpoints.get(profile)


def list_local_endpoints(profile: str = "local-colab") -> Dict[str, str]:
    """Return a mapping tool_id -> endpoint for the given profile.

    Tools without the given profile are omitted.
    """
    data = load_tool_registry()
    tools = data.get("tools", {}) if isinstance(data, dict) else {}
    out: Dict[str, str] = {}
    for tid, meta in tools.items():
        ep = meta.get("endpoints", {}).get(profile)
        if ep:
            out[tid] = ep
    return out
