#!/usr/bin/env python3
"""Local dry-run validator for WorkflowAgent YAML.

This module is a minimal, safe validator for local tests and dry-run checks.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_workflow(workflow: Dict[str, Any], tool_registry: Optional[Dict[str, Any]] = None, schema_artifacts: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(workflow, dict):
        print("Workflow document must be a mapping", file=sys.stderr)
        return False
    stages = workflow.get("stages")
    if not isinstance(stages, list) or not stages:
        print("Workflow must include a non-empty 'stages' list", file=sys.stderr)
        return False

    ok = True
    for idx, s in enumerate(stages):
        if not isinstance(s, dict):
            print(f"Stage at index {idx} must be a mapping", file=sys.stderr)
            ok = False
            continue
        for key in ("id", "tool_id", "input_schema", "output_schema"):
            if key not in s:
                print(f"Stage {s.get('id', idx)} missing required key: {key}", file=sys.stderr)
                ok = False

        if tool_registry is not None:
            tool_id = s.get("tool_id")
            if tool_id and tool_id not in tool_registry:
                print(f"Warning: tool_id '{tool_id}' not found in tool registry", file=sys.stderr)

    return ok


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    wf = load_yaml(Path(args.file))
    ok = validate_workflow(wf, tool_registry=None, schema_artifacts=None)
    print("OK" if ok else "INVALID")
    raise SystemExit(0 if ok else 1)
