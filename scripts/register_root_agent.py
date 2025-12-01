#!/usr/bin/env python3
"""Register or dry-run validate the root LlmAgent configuration."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from scripts import register_root_agent_local as validator


def _summary(config: Dict[str, Any]) -> Dict[str, Any]:
    metadata = config.get("metadata") if isinstance(config, dict) else {}
    workflow_config = metadata.get("workflow_config") if isinstance(metadata, dict) else None
    return {
        "name": config.get("name"),
        "model": config.get("model"),
        "workflow_config": workflow_config,
    }


def register_with_cli(config_path: Path) -> tuple[bool, str]:
    adk_path = shutil.which("adk")
    if not adk_path:
        return False, "adk CLI not found"
    cmd = [adk_path, "agents", "register", "--file", str(config_path.resolve())]
    try:
        proc = subprocess.run(cmd, check=False)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"adk CLI error: {exc}"
    if getattr(proc, "returncode", 1) == 0:
        return True, "Registered root agent via adk CLI"
    return False, f"adk CLI returned {getattr(proc, 'returncode', 'unknown')}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Register the Sparkle Motion root agent")
    parser.add_argument("--config", type=Path, default=Path("configs/root_agent.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not call ADK CLI")
    parser.add_argument("--confirm", action="store_true", help="Required to perform a non-dry-run registration")
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        return 2

    config_doc = validator.load_yaml(args.config)
    if not validator.validate_root_agent(config_doc, config_path=args.config):
        print("Root agent config validation failed")
        return 3

    if not args.dry_run and not args.confirm:
        print("No --confirm provided; running in dry-run mode instead")
        args = argparse.Namespace(**{**vars(args), "dry_run": True})

    if args.dry_run:
        summary = _summary(config_doc)
        print("Dry run: root agent config is valid")
        print(json.dumps(summary, indent=2))
        return 0

    ok, msg = register_with_cli(args.config)
    print(msg)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
