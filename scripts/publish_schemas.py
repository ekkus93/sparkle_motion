#!/usr/bin/env python3
"""Publish JSON Schema files to ADK artifact registry.

Behavior:
- SDK-first: attempts a guarded import of `google.adk` and probes for an
  artifacts client or top-level publish helpers. If a known SDK surface is
  detected, it will attempt a best-effort push.
- CLI-fallback: if the SDK is not available or the SDK push attempt fails,
  the script calls the `adk` CLI (must be on PATH and authenticated).

This script is intentionally conservative: where the SDK surface cannot be
automatically discovered it falls back to the CLI path.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None

from sparkle_motion.adk_helpers import probe_sdk, publish_with_sdk, publish_with_cli


def load_artifact_map(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read artifact mapping; install PyYAML in the env")
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact mapping file not found: {config_path}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def find_schema_files(schema_dir: str) -> list:
    p = Path(schema_dir)
    if not p.exists():
        raise FileNotFoundError(f"Schemas directory not found: {schema_dir}")
    # match *.schema.json recursively
    return sorted(str(pf) for pf in p.rglob("*.schema.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish JSON Schemas to ADK (SDK-first, CLI-fallback)")
    parser.add_argument("--schemas-dir", default="schemas", help="Directory containing .schema.json files (default: schemas)")
    parser.add_argument("--artifacts-config", default=None, help="Optional YAML mapping file: schema filename -> artifact name")
    parser.add_argument("--project", default=None, help="ADK project name to pass to CLI (optional)")
    parser.add_argument("--use-cli", action="store_true", help="Force CLI fallback even if SDK is available")
    parser.add_argument("--dry-run", action="store_true", help="Don't publish; just print what would be done")
    parser.add_argument("--local-only", action="store_true", help="Do a local-only publish: copy schemas to artifacts/schemas and write file:// URIs into the artifacts config")
    parser.add_argument("--backup", action="store_true", help="When writing the artifacts config, create a timestamped backup of the existing file")
    parser.add_argument("--confirm", action="store_true", help="Don't prompt for confirmation when overwriting an existing artifacts config")
    args = parser.parse_args()

    try:
        artifact_map = load_artifact_map(args.artifacts_config)
    except Exception as e:
        print(f"Failed loading artifacts mapping: {e}", file=sys.stderr)
        artifact_map = {}

    try:
        files = find_schema_files(args.schemas_dir)
    except Exception as e:
        print(f"No schema files found: {e}", file=sys.stderr)
        return 2

    # Local-only publish: copy schema files to artifacts/schemas/ and update
    # the artifacts config with file:// URIs. This is intended for isolated
    # servers where ADK credentials are not available.
    if args.local_only:
        if yaml is None:
            print("PyYAML is required for local-only publish; install PyYAML.", file=sys.stderr)
            return 4

        out_dir = Path("artifacts/schemas")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        # load or initialize the artifacts config structure
        cfg = artifact_map if isinstance(artifact_map, dict) else {}
        if "schemas" not in cfg:
            cfg.setdefault("schemas", {})

        def to_snake(name: str) -> str:
            # Simple PascalCase/CamelCase -> snake_case converter
            import re

            s1 = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", name)
            s2 = re.sub('([a-z0-9])([A-Z])', r"\1_\2", s1)
            return s2.replace('-', '_').lower()

        for fpath in files:
            fname = os.path.basename(fpath)
            dest = out_dir / fname
            abs_dest = dest.resolve()
            if args.dry_run:
                print(f"[dry-run] Would copy {fpath} -> {dest}")
                print(f"[dry-run] Would set artifact uri for {fname} -> file://{abs_dest}")
                continue

            # copy the file
            shutil.copy2(fpath, dest)

            # derive schema key (movie_plan, asset_refs, etc.) from filename stem
            stem = Path(fpath).stem.replace('.schema', '')
            key = to_snake(stem)

            cfg.setdefault("schemas", {})
            cfg["schemas"][key] = {
                "uri": f"file://{abs_dest}",
                "local_path": str(dest)
            }

        # write back the artifacts config
        if args.artifacts_config:
            cfg_path = Path(args.artifacts_config)

            # if the config exists and we're not auto-confirming, prompt the user
            if cfg_path.exists() and not args.confirm:
                try:
                    resp = input(f"Artifacts config {cfg_path} exists. Overwrite? [y/N]: ").strip().lower()
                except Exception:
                    resp = "n"
                if resp != "y":
                    print("Aborted: artifacts config not modified.")
                    return 5

            # optionally create a timestamped backup of the existing config
            if cfg_path.exists() and args.backup:
                bak = cfg_path.with_name(cfg_path.name + f".bak.{int(time.time())}")
                shutil.copy2(cfg_path, bak)
                print(f"Created backup of artifacts config: {bak}")

            with cfg_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)
            print(f"Wrote local-only artifacts config to {cfg_path}")
        else:
            print("Local-only publish complete; no artifacts config path provided, so changes were not recorded.")

        print("All schema artifacts processed (local-only).")
        return 0

    sdk_probe = None
    if not args.use_cli:
        sdk_probe = probe_sdk()

    failures = []
    published_map = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        # derive artifact name from mapping or filename without suffix
        artifact_name = artifact_map.get(fname) or Path(fpath).stem.replace('.schema','')

        artifact_uri = None
        if sdk_probe is not None:
            adk_module, client = sdk_probe
            try:
                artifact_uri = publish_with_sdk(adk_module, client, fpath, artifact_name, args.dry_run, args.project)
            except Exception as e:
                print(f"SDK path raised during publish of {fpath}: {e}", file=sys.stderr)
                artifact_uri = None

        if artifact_uri is None:
            artifact_uri = publish_with_cli(fpath, artifact_name, args.project, args.dry_run, artifact_map)
            if artifact_uri is None:
                failures.append(fpath)
            else:
                published_map[artifact_name] = artifact_uri
        else:
            published_map[artifact_name] = artifact_uri

    if failures:
        print("Failed to publish the following schema files:")
        for f in failures:
            print(" -", f)
        return 3

    # If we have published artifacts and an artifacts_config was provided,
    # update the config mapping with the returned artifact:// URIs.
    if published_map and args.artifacts_config:
        try:
            if yaml is None:
                print("PyYAML required to update artifacts config; install PyYAML.", file=sys.stderr)
            else:
                cfg_path = Path(args.artifacts_config)
                cfg = artifact_map if isinstance(artifact_map, dict) else {}
                cfg.setdefault("schemas", {})
                for name, uri in published_map.items():
                    # derive key name from artifact filename pattern
                    key = name.replace('-', '_')
                    cfg["schemas"].setdefault(key, {})
                    cfg["schemas"][key]["uri"] = uri
                    # keep any existing local_path if present
                    existing = cfg["schemas"][key].get("local_path")
                    if existing:
                        cfg["schemas"][key]["local_path"] = existing

                # backup and write (respect existing --backup/--confirm behavior)
                if cfg_path.exists() and not args.confirm:
                    try:
                        resp = input(f"Artifacts config {cfg_path} exists. Overwrite? [y/N]: ").strip().lower()
                    except Exception:
                        resp = "n"
                    if resp != "y":
                        print("Aborted: artifacts config not modified.")
                        return 5

                if cfg_path.exists() and args.backup:
                    bak = cfg_path.with_name(cfg_path.name + f".bak.{int(time.time())}")
                    shutil.copy2(cfg_path, bak)
                    print(f"Created backup of artifacts config: {bak}")

                with cfg_path.open("w", encoding="utf-8") as fh:
                    yaml.safe_dump(cfg, fh, sort_keys=False)
                print(f"Wrote updated artifacts config to {cfg_path}")
        except Exception as e:
            print(f"Failed updating artifacts config: {e}", file=sys.stderr)
            return 6

    print("All schema artifacts processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
