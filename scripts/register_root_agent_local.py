#!/usr/bin/env python3
"""Local validator for the root LlmAgent configuration."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_repo_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def validate_root_agent(config: Dict[str, Any], *, config_path: Path | None = None) -> bool:
    ok = True

    def _error(message: str) -> None:
        nonlocal ok
        print(message, file=sys.stderr)
        ok = False

    if not isinstance(config, dict):
        _error("Root agent config must be a mapping")
        return False

    agent_class = config.get("agent_class")
    if agent_class != "LlmAgent":
        _error("agent_class must be 'LlmAgent'")

    for field in ("name", "model", "instruction"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            _error(f"Missing or invalid '{field}' field")

    metadata = config.get("metadata")
    if metadata and isinstance(metadata, dict):
        workflow_conf = metadata.get("workflow_config")
        if isinstance(workflow_conf, str) and workflow_conf.strip():
            wf_path = _resolve_repo_path(workflow_conf)
            if not wf_path.exists():
                _error(f"workflow_config path '{workflow_conf}' does not exist")
        else:
            _error("metadata.workflow_config must be a non-empty string")
    else:
        _error("metadata.workflow_config must be specified")

    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a root agent config")
    parser.add_argument("--file", "-f", required=True)
    args = parser.parse_args()
    cfg = load_yaml(Path(args.file))
    is_valid = validate_root_agent(cfg, config_path=Path(args.file))
    print("OK" if is_valid else "INVALID")
    raise SystemExit(0 if is_valid else 1)
