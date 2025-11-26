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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

LOG = logging.getLogger("register_tools")


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def try_register_with_sdk(tool: Dict[str, Any], dry_run: bool) -> Optional[str]:
    """Attempt to register a tool using the google.adk SDK.

    Returns a registration id or URI on success, or None on failure / not supported.
    The SDK import is guarded so this function will not fail import-time when
    `google.adk` is not installed.
    """
    try:
        import google.adk as adk  # guarded import
    except Exception:  # pragma: no cover - guarded
        LOG.debug("google.adk not available; skipping SDK path")
        return None

    # Best-effort probes for common SDK surfaces. The concrete SDK in the
    # environment may expose a different API; if so, this function should be
    # adjusted to call the actual client.
    try:
        # Try a common client name
        client = None
        if hasattr(adk, "ToolRegistry"):
            client = adk.ToolRegistry()
        elif hasattr(adk, "ToolRegistryClient"):
            client = adk.ToolRegistryClient()

        if client is None:
            LOG.debug("ADK SDK present but no ToolRegistry client found; skipping SDK path")
            return None

        payload = dict(tool)  # copy; ensure JSON-serializable
        if dry_run:
            LOG.info("[dry-run] SDK would register tool: %s", payload.get("id") or payload.get("name"))
            return "dry-run://sdk/" + (payload.get("id") or payload.get("name", "unnamed"))

        # The exact method name may vary; try common names.
        if hasattr(client, "register_tool"):
            res = client.register_tool(payload)
            # Try to extract an identifier
            return getattr(res, "id", None) or getattr(res, "uri", None) or str(res)
        elif hasattr(client, "create_tool"):
            res = client.create_tool(payload)
            return getattr(res, "id", None) or getattr(res, "uri", None) or str(res)
        else:
            LOG.debug("ToolRegistry client found but no register/create method available")
            return None
    except Exception as e:  # pragma: no cover - runtime environment differences
        LOG.exception("SDK register attempt failed: %s", e)
        return None


def register_with_cli(tool: Dict[str, Any], dry_run: bool) -> Optional[str]:
    """Register a tool using the `adk` CLI as a fallback.

    This writes a temporary JSON file and calls `adk tools register --file <path>`.
    Returns the CLI output-parsed id/uri on success, or None on failure.
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
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            LOG.error("adk CLI returned %s: %s", proc.returncode, out)
            return None

        # Try to parse a returned URI or id from the output using JSON or simple heuristics
        try:
            j = json.loads(proc.stdout)
            candidate = j.get("id") or j.get("uri") or j.get("artifact_uri")
            if candidate:
                return candidate
        except Exception:
            pass

        # fallback: search for artifact:// or tool:// style tokens
        for token in out.split():
            if token.startswith("artifact://") or token.startswith("tool://") or token.startswith("https://"):
                return token.strip()

        # As a last resort return stdout trimmed
        return proc.stdout.strip() or None
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
