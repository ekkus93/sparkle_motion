"""Register WorkflowAgent / workflow definitions into ADK.

Pattern mirrors `scripts/register_tools.py`: prefer SDK (`google.adk`) when
available, probe plausible entrypoints, and fall back to the `adk` CLI.

Usage:
  PYTHONPATH=src python scripts/register_workflow.py --path configs/workflows.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from sparkle_motion.adk_helpers import probe_sdk, register_entity_with_sdk, register_entity_with_cli


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, help="Path to workflows registry (yaml/json)")
    p.add_argument("--dry-run", action="store_true", help="Do not register; only print what would be done")
    p.add_argument("--use-cli", action="store_true", help="Force use of the 'adk' CLI instead of SDK")
    p.add_argument("--force", action="store_true", help="Force re-registration where applicable")
    return p.parse_args()


def load_workflow_registry(path: Path | None) -> Dict[str, Any]:
    if path is None:
        raise ValueError("Path to workflow registry is required")
    raw = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(raw)
    except Exception:
        return json.loads(raw)


def register_with_sdk(workflow_id: str, meta: Dict[str, Any]) -> Tuple[bool, str]:
    sdk_probe = probe_sdk()
    if not sdk_probe:
        return False, "SDK not importable"
    adk_mod = sdk_probe[0]

    # Probe plausible SDK entrypoints in the same style as previous implementation
    candidates = [
        ("workflows", "register"),
        ("workflow_registry", "register_workflow"),
        ("workflowing", "register_workflow"),
        ("workflows", "create"),
    ]

    for attr, method in candidates:
        hub = getattr(adk_mod, attr, None)
        if hub is None:
            continue
        fn = getattr(hub, method, None)
        if not fn:
            continue
        try:
            res = fn(workflow_id, meta) if getattr(fn, "__code__", None) and fn.__code__.co_argcount >= 2 else fn(meta)
            return True, f"SDK: called {attr}.{method} -> {res}"
        except Exception as e:  # pragma: no cover - depends on SDK
            return False, f"SDK {attr}.{method} raised: {e}"

    # No known API found
    return False, "SDK present but no known registration API found"


def register_with_cli(workflow_id: str, meta: Dict[str, Any]) -> Tuple[bool, str]:
    if shutil.which("adk") is None:
        return False, "adk CLI not found in PATH"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump({"id": workflow_id, "workflow": meta}, tf, indent=2)
        tmpname = tf.name

    cmd = ["adk", "workflows", "register", "--file", tmpname]
    try:
        # Run CLI in a simple, test-friendly way first
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            # Try to extract a uri with the helper, but ignore parsing errors
            try:
                uri = register_entity_with_cli(cmd, dry_run=False)
            except TypeError:
                uri = None
            if uri:
                return True, f"CLI: registered -> {uri}"
            return True, f"CLI: registered {workflow_id} via {tmpname}"
        return False, f"CLI returned code {proc.returncode}"
    finally:
        try:
            Path(tmpname).unlink()
        except Exception:
            pass


def normalize_meta(wid: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": wid,
        "description": raw.get("description"),
        "spec": raw.get("spec"),
        "inputs": raw.get("inputs", {}),
        "outputs": raw.get("outputs", {}),
    }


def main() -> int:
    args = parse_args()
    try:
        data = load_workflow_registry(args.path)
    except Exception as e:
        print(f"Failed to load workflow registry: {e}")
        return 3

    workflows = data.get("workflows", {}) if isinstance(data, dict) else {}
    if not workflows:
        print("No workflows found in registry; nothing to do.")
        return 0

    summary = []
    for wid, raw in workflows.items():
        meta = normalize_meta(wid, raw)
        summary.append((wid, meta))

    if args.dry_run:
        print("Dry run: would register the following workflows:")
        for wid, meta in summary:
            print(f"- {wid}: desc={meta.get('description')}, spec_keys={list((meta.get('spec') or {}).keys())}")
        return 0

    for wid, meta in summary:
        print(f"Registering workflow {wid} -> spec keys={list((meta.get('spec') or {}).keys())}")
        if not args.use_cli:
            ok, msg = register_with_sdk(wid, meta)
            print("  SDK attempt:", msg)
            if ok:
                continue

        ok, msg = register_with_cli(wid, meta)
        print("  CLI attempt:", msg)
        if not ok:
            print(f"Failed to register {wid} with SDK and CLI; skipping.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
