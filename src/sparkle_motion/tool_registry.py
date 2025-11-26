"""Helpers to load the local `configs/tool_registry.yaml` and expose
convenience accessors for the local-colab profile.

This is intentionally small and dependency-free (uses `yaml` which is already
declared in requirements). Callers should import and use the helpers rather than
hardcoding endpoints in scripts or notebooks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "tool_registry.yaml"


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
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Tool registry not found at {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


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
