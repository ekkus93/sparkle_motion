"""Register FunctionTools from `configs/tool_registry.yaml` into ADK.

This script prefers a Python SDK integration (`google.adk`) when available and
falls back to the `adk` CLI. Both paths are guarded so the script can run in
Colab (where CLI may be available) or in dev environments with the SDK.

Usage:
  PYTHONPATH=src python scripts/register_tools.py
  PYTHONPATH=src python scripts/register_tools.py --path configs/tool_registry.yaml --dry-run

Notes:
  - The ADK SDK surface may differ across versions; the SDK helper probes a
    few plausible entrypoints and methods and attempts a best-effort call.
  - The CLI fallback writes a temporary JSON metadata file and calls
    `adk tools register --file <tmpfile>`; adjust if your ADK CLI differs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from sparkle_motion.tool_registry import load_tool_registry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, help="Path to tool_registry.yaml (defaults to configs/tool_registry.yaml)")
    p.add_argument("--dry-run", action="store_true", help="Do not register; only print what would be done")
    p.add_argument("--use-cli", action="store_true", help="Force use of the 'adk' CLI instead of SDK")
    p.add_argument("--force", action="store_true", help="Force re-registration where applicable")
    return p.parse_args()


def register_with_sdk(tool_id: str, meta: Dict[str, Any]) -> Tuple[bool, str]:
    """Best-effort attempt to register a tool via google.adk SDK.

    Returns (success, message).
    """
    try:
        import google.adk as adk  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"SDK not importable: {e}"

    # Probe plausible SDK registration entrypoints.
    candidates = [
        ("tools", "register"),
        ("tool_registry", "register_tool"),
        ("tooling", "register_tool"),
    ]

    for attr, method in candidates:
        hub = getattr(adk, attr, None)
        if hub is None:
            continue
        fn = getattr(hub, method, None)
        if not fn:
            continue
        try:
            # Many SDKs accept dict-like metadata; call with meta and return.
            res = fn(tool_id, meta) if fn.__code__.co_argcount >= 2 else fn(meta)
            return True, f"SDK: called {attr}.{method} -> {res}"
        except Exception as e:  # pragma: no cover - depends on SDK
            return False, f"SDK {attr}.{method} raised: {e}"

    return False, "SDK present but no known registration API found"


def register_with_cli(tool_id: str, meta: Dict[str, Any]) -> Tuple[bool, str]:
    """Fallback to the `adk` CLI by writing a temporary JSON and invoking register.

    Returns (success, message).
    """
    if shutil.which("adk") is None:
        return False, "adk CLI not found in PATH"

    # Write metadata to a temp file and call the CLI. Adjust CLI flags as needed.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"id": tool_id, "metadata": meta}, tf, indent=2)
        tmpname = tf.name

    cmd = ["adk", "tools", "register", "--file", tmpname]
    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            return True, f"CLI: registered {tool_id} via {tmpname}"
        return False, f"CLI returned code {proc.returncode}"
    finally:
        try:
            Path(tmpname).unlink()
        except Exception:
            pass


def normalize_meta(tool_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the registry metadata into a shape acceptable to SDK/CLI.

    This keeps the YAML expressive but produces a compact dict for registration.
    """
    return {
        "id": tool_id,
        "description": raw.get("description"),
        "endpoint": raw.get("endpoints", {}).get("local-colab") or raw.get("endpoints", {}).get("default"),
        "ports": raw.get("ports", {}),
        "schemas": raw.get("schemas", {}),
        "retry_hints": raw.get("retry_hints", {}),
    }


def main() -> int:
    args = parse_args()
    path = args.path
    try:
        data = load_tool_registry(path) if path else load_tool_registry()
    except Exception as e:
        print(f"Failed to load tool registry: {e}")
        return 3

    tools = data.get("tools", {}) if isinstance(data, dict) else {}
    if not tools:
        print("No tools found in registry; nothing to do.")
        return 0

    summary = []
    for tid, raw in tools.items():
        meta = normalize_meta(tid, raw)
        summary.append((tid, meta))

    if args.dry_run:
        print("Dry run: would register the following tools:")
        for tid, meta in summary:
            print(f"- {tid}: endpoint={meta.get('endpoint')}, schemas={list(meta.get('schemas', {}).keys())}")
        return 0

    for tid, meta in summary:
        print(f"Registering {tid} -> {meta.get('endpoint')}")
        if not args.use_cli:
            ok, msg = register_with_sdk(tid, meta)
            print("  SDK attempt:", msg)
            if ok:
                continue

        ok, msg = register_with_cli(tid, meta)
        print("  CLI attempt:", msg)
        if not ok:
            print(f"Failed to register {tid} with SDK and CLI; skipping.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
