#!/usr/bin/env python3
"""Register FunctionTools from a tool registry config.

Usage: python scripts/register_tools.py --config configs/tool_registry.yaml [--dry-run]

Behavior:
- Reads the YAML at `configs/tool_registry.yaml` and expects a top-level
  `tools:` mapping where each entry is a dict describing a FunctionTool.
- Tries to use the `google.adk` SDK first (guarded import). If the SDK is
  available and exposes a registration client, it will attempt to register
  tools via the SDK. Otherwise it falls back to invoking the `adk` CLI.

This script is conservative: it supports `--dry-run` to preview actions and
`--confirm` to proceed when not in dry-run.
"""
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from sparkle_motion.adk_helpers import probe_sdk, register_entity_with_sdk, register_entity_with_cli
from sparkle_motion.tool_registry import resolve_schema_references
import argparse

LOG = logging.getLogger("register_tools")


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def try_register_with_sdk(tool: Dict[str, Any], dry_run: bool) -> Optional[str]:
    """Attempt to register a tool using the `sparkle_motion.adk_helpers` SDK helper.

    Returns a registration id/uri on success or None.
    """
    res = probe_sdk()
    if not res:
        LOG.debug("google.adk not available; skipping SDK path")
        return None
    adk_mod, _ = res

    try:
        return register_entity_with_sdk(adk_mod, tool, entity_kind="tool", name=tool.get("id") or tool.get("name"), dry_run=dry_run)
    except Exception:
        LOG.exception("SDK register attempt failed")
        return None


def register_with_cli(tool: Dict[str, Any], dry_run: bool) -> Optional[str]:
    """Register a tool via the `adk` CLI using the central helper.

    Writes a temp file and delegates to `register_entity_with_cli` to run the
    command and parse the result.
    """
    payload = json.dumps(tool, ensure_ascii=False, indent=2)

    if dry_run:
        LOG.info("[dry-run] CLI would run: adk tools register --file <tool.json> for %s", tool.get("id") or tool.get("name"))
        return "dry-run://cli/" + (tool.get("id") or tool.get("name", "unnamed"))

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        fh.write(payload)
        tmp_path = Path(fh.name)

    try:
        cmd = ["adk", "tools", "register", "--file", str(tmp_path)]
        LOG.info("Running CLI: %s", " ".join(cmd))
        return register_entity_with_cli(cmd, dry_run=dry_run)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Register FunctionTools from config")
    p.add_argument("--config", type=Path, default=Path("configs/tool_registry.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm", action="store_true", help="Require explicit confirmation to perform changes")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.config.exists():
        LOG.error("Config not found: %s", args.config)
        return 2

    cfg = load_config(args.config)
    tools = cfg.get("tools") or cfg
    if not tools:
        LOG.error("No tools found in config: %s", args.config)
        return 1

    # If confirm required and not provided, show planned actions and exit
    if args.confirm is False and not args.dry_run:
        LOG.info("No --confirm provided; running in dry-run unless --confirm is supplied")
        args = argparse.Namespace(**vars(args), dry_run=True)

    results = {}
    for name, tool in tools.items() if isinstance(tools, dict) else enumerate(tools):
        # normalize tool dict
        t = tool if isinstance(tool, dict) else dict(tool)
        t.setdefault("id", name if isinstance(name, str) else t.get("id", f"tool-{name}"))
        if isinstance(t.get("schemas"), dict):
            try:
                t["schemas"] = resolve_schema_references(t["schemas"])
            except Exception as exc:
                LOG.error("Failed to resolve schema references for %s: %s", t.get("id"), exc)
                return 4

        LOG.info("Processing tool: %s", t.get("id"))

        # Try SDK first
        sdk_res = try_register_with_sdk(t, args.dry_run)
        if sdk_res:
            LOG.info("Registered via SDK: %s -> %s", t.get("id"), sdk_res)
            results[t.get("id")] = {"method": "sdk", "result": sdk_res}
            continue

        # Fallback to CLI
        cli_res = register_with_cli(t, args.dry_run)
        if cli_res:
            LOG.info("Registered via CLI: %s -> %s", t.get("id"), cli_res)
            results[t.get("id")] = {"method": "cli", "result": cli_res}
            continue

        LOG.error("Failed to register tool: %s", t.get("id"))
        results[t.get("id")] = {"method": "none", "result": None}

    # Summarize
    LOG.info("Registration summary:\n%s", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
