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

import shutil
from pathlib import Path
from typing import Any, Dict, Tuple
from typing import Any, Dict, List

from sparkle_motion.tool_registry import load_tool_registry, resolve_schema_references


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, help="Path to tool_registry.yaml (defaults to configs/tool_registry.yaml)")
    p.add_argument("--dry-run", action="store_true", help="Do not register; only print what would be done")
    p.add_argument("--use-cli", action="store_true", help="Force use of the 'adk' CLI instead of SDK")
    p.add_argument("--force", action="store_true", help="Force re-registration where applicable")
    p.add_argument("--metadata-dir", type=Path, help="Optional directory to read per-tool metadata files from (overrides function_tools/*)")
    p.add_argument("--strict", action="store_true", help="Fail if per-tool metadata is present but invalid")
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
    raw_schemas = raw.get("schemas") or {}
    schemas = resolve_schema_references(raw_schemas) if isinstance(raw_schemas, dict) else raw_schemas
    return {
        "id": tool_id,
        "description": raw.get("description"),
        "endpoint": raw.get("endpoints", {}).get("local-colab") or raw.get("endpoints", {}).get("default"),
        "ports": raw.get("ports", {}),
        "schemas": schemas,
        "retry_hints": raw.get("retry_hints", {}),
    }


def validate_metadata(raw: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a per-tool metadata.json minimally.

    Returns (is_valid, errors).
    The validation is intentionally conservative: metadata must include an
    identifier (`id` or `name`) and an endpoint (one of `endpoint` or
    `endpoints`). It should also include either `schemas` or
    `response_json_schema` to be useful for registration payloads.
    """
    errs: List[str] = []
    if not isinstance(raw, dict):
        return False, ["metadata is not a JSON object"]

    # identity
    if not (raw.get("id") or raw.get("name")):
        errs.append("missing 'id' or 'name'")

    # package / distribution info
    if not (raw.get("package_name") or raw.get("package")):
        errs.append("missing 'package_name' or 'package'")

    # version must be present and a non-empty string
    ver = raw.get("version")
    if not ver or not isinstance(ver, str) or not ver.strip():
        errs.append("missing or invalid 'version' (non-empty string required)")

    # endpoint(s): accept a single 'endpoint' or an 'endpoints' mapping
    if raw.get("endpoint"):
        if not isinstance(raw.get("endpoint"), str) or not raw.get("endpoint").strip():
            errs.append("'endpoint' must be a non-empty string")
    elif raw.get("endpoints"):
        eps = raw.get("endpoints")
        if not isinstance(eps, dict) or not eps:
            errs.append("'endpoints' must be a non-empty mapping")
    else:
        errs.append("missing 'endpoint' or 'endpoints'")

    # schemas or response_json_schema must be present and non-empty dict
    if raw.get("schemas"):
        if not isinstance(raw.get("schemas"), dict) or not raw.get("schemas"):
            errs.append("'schemas' must be a non-empty mapping if present")
    elif raw.get("response_json_schema"):
        if not isinstance(raw.get("response_json_schema"), dict) or not raw.get("response_json_schema"):
            errs.append("'response_json_schema' must be a non-empty mapping if present")
    else:
        errs.append("missing 'schemas' or 'response_json_schema'")

    return (len(errs) == 0), errs


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
        # Prefer per-tool generated metadata when present. Where the caller
        # supplied `--metadata-dir` use that first; otherwise look under
        # function_tools/<tid>/metadata.json. If a metadata file is present,
        # validate it and either use it or (optionally) fail depending on
        # `--strict`.
        meta = None
        candidate_paths: List[Path] = []
        try:
            if args.metadata_dir:
                candidate_paths.append(Path(args.metadata_dir) / tid / "metadata.json")
            candidate_paths.append(Path(__file__).resolve().parents[1] / "function_tools" / tid / "metadata.json")

            for ft_meta_path in candidate_paths:
                if not ft_meta_path.exists():
                    continue
                try:
                    raw_meta = json.loads(ft_meta_path.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"Warning: failed to load {ft_meta_path}: {e}", file=sys.stderr)
                    raw_meta = None

                if raw_meta is None:
                    continue

                ok, errors = validate_metadata(raw_meta)
                if not ok:
                    msg = f"Invalid metadata at {ft_meta_path}: {errors}"
                    if args.strict:
                        print(msg, file=sys.stderr)
                        return 5
                    else:
                        print("Warning:", msg, file=sys.stderr)
                        # ignore invalid metadata and fall back
                        continue

                meta = raw_meta
                break
        except Exception:
            meta = None

        if meta is None:
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
