"""Generate FunctionTool metadata JSON files from `configs/tool_registry.yaml`.

Usage: run from repo root (PYTHONPATH optional). This helper writes a
`metadata.json` file under each `function_tools/<tool>/` directory containing
fields suitable for ADK ToolRegistry registration and for packaging.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tool_registry.yaml"
FT_DIR = ROOT / "function_tools"
DEFAULT_VERSION = "0.1.0"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalize_tool_name(name: str) -> str:
    return name.replace("-", "_")


def build_metadata(name: str, spec: dict) -> dict:
    metadata = {
        "name": name,
        "package_name": f"sparkle_motion.{name}",
        "version": DEFAULT_VERSION,
        "description": spec.get("description", ""),
        "endpoints": spec.get("endpoints", {}),
        "schemas": spec.get("schemas", {}),
        "retry_hints": spec.get("retry_hints", {}),
        "health_path": "/health",
        "invoke_path": "/invoke",
        "response_json_schema": spec.get("schemas", {}).get("output"),
    }
    return metadata


def main() -> int:
    if not CONFIG.exists():
        print(f"Config not found: {CONFIG}", file=sys.stderr)
        return 2

    cfg = load_config(CONFIG)
    tools = cfg.get("tools", {})
    created = []

    for key, spec in tools.items():
        # map YAML key to function_tools dir (some keys use underscores already)
        dir_name = key
        target_dir = FT_DIR / dir_name
        if not target_dir.exists():
            print(f"Skipping missing tool dir: {target_dir}")
            continue
        metadata = build_metadata(dir_name, spec)
        out_path = target_dir / "metadata.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        created.append(out_path)

    print(f"Wrote {len(created)} metadata files")
    for p in created:
        print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
