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
from pathlib import Path
from typing import Optional, Tuple
import asyncio
import getpass
import shutil
import time
import re
import json

try:
    import yaml
except Exception:
    yaml = None


def probe_sdk() -> Optional[Tuple[object, Optional[object]]]:
    """Try to import google.adk and discover an artifacts client.

    Returns (adk_module, client_candidate) or None when SDK import fails.
    The client_candidate may be None when the module exists but no obvious
    artifacts helper is present.
    """
    try:
        import google.adk as adk  # type: ignore
    except Exception:
        return None

    for cand in ("artifacts", "ArtifactService", "artifacts_client", "artifact_client"):
        client = getattr(adk, cand, None)
        if client is not None:
            return adk, client

    # no obvious client found, return the module so callers can probe further
    return adk, None


def _publish_with_sdk_service(adk_module, file_path: str, artifact_name: str, dry_run: bool, project: Optional[str] = None) -> Optional[str]:
    """Publish a file using ADK artifact service classes (best-effort).

    This attempts to instantiate an available artifact service (GCS or
    filesystem-backed) and call its async `save_artifact` method with a
    `Part` constructed from the local file URI. Returns a constructed
    artifact:// URI on success, or None on failure.
    """
    try:
        # Prefer GCS service if a bucket is configured via env
        bucket = os.environ.get("ADK_ARTIFACTS_GCS_BUCKET")
        if bucket:
            from google.adk.artifacts.gcs_artifact_service import GcsArtifactService as _Gcs

            svc = _Gcs(bucket)
        else:
            from google.adk.artifacts.file_artifact_service import FileArtifactService as _FileSvc

            root = os.environ.get("ADK_ARTIFACTS_ROOT", "artifacts/adk")
            svc = _FileSvc(root)

        # construct a Part referencing the file URI; the service will
        # associate the file if needed
        try:
            from google.genai.types import Part
        except Exception:
            # fallback: try to find types via the artifacts package
            types_mod = getattr(adk_module, "artifacts", None)
            Part = None
            if types_mod is not None:
                try:
                    from google.genai.types import Part  # try once more
                except Exception:
                    Part = None

        if dry_run:
            print(f"[dry-run] SDK would save artifact via {svc.__class__.__name__}: file={file_path}, name={artifact_name}")
            return f"artifact://{project or artifact_name}/schemas/{artifact_name}/v1"

        if Part is None:
            print("Unable to locate Part type for SDK publish", file=sys.stderr)
            return None

        file_uri = str(Path(file_path).resolve())
        # Build a Part containing the file bytes (some artifact services
        # require inline data or text). Prefer from_bytes for full
        # compatibility; fall back to from_uri if that fails.
        try:
            import mimetypes

            mime_type, _ = mimetypes.guess_type(file_uri)
            mime_type = mime_type or "application/octet-stream"
            with open(file_path, "rb") as _fh:
                data = _fh.read()
            part = Part.from_bytes(data=data, mime_type=mime_type)
        except Exception:
            try:
                part = Part.from_uri(file_uri=file_uri)
            except Exception:
                raise

        app_name = project or os.environ.get("ADK_PROJECT") or Path.cwd().name
        user_id = getpass.getuser() or "user"

        # save_artifact is async; call it with asyncio.run
        try:
            rev = asyncio.run(svc.save_artifact(app_name=app_name, user_id=user_id, filename=artifact_name, artifact=part))
        except TypeError:
            # some implementations expect filename as positional/other args
            rev = asyncio.run(svc.save_artifact(app_name=app_name, user_id=user_id, filename=artifact_name, artifact=part))

        # rev is an integer revision id; craft a canonical artifact URI
        return f"artifact://{app_name}/schemas/{artifact_name}/v{rev}"
    except Exception as e:
        print(f"SDK artifact-service publish failed: {e}", file=sys.stderr)
        return None


def publish_with_sdk(adk_module, client, file_path: str, artifact_name: str, dry_run: bool, project: Optional[str] = None) -> Optional[str]:
    """Best-effort attempts to publish using discovered SDK objects.

    This function tries a small set of common method names used by SDKs.
    It purposefully traps exceptions and returns False on any failure so the
    caller can fall back to CLI.
    """
    candidates = []
    if client is not None:
        candidates.extend([getattr(client, n, None) for n in ("push", "publish", "upload", "create")])
    # check top-level module helper names too
    candidates.extend([getattr(adk_module, n, None) for n in ("push_schema", "publish_schema", "push_artifact", "publish_artifact")])

    for fn in [c for c in candidates if callable(c)]:
        try:
            if dry_run:
                print(f"[dry-run] SDK would call: {fn.__qualname__}({file_path!r}, name={artifact_name!r})")
                # return a plausible artifact URI for dry-run if possible
                return f"artifact://{artifact_name}/v1"

            # try a few calling conventions
            try:
                res = fn(file_path)
            except TypeError:
                try:
                    res = fn(path=file_path)
                except TypeError:
                    # last resort: pass a file object
                    with open(file_path, "rb") as fh:
                        res = fn(fh)

            print(f"Published {file_path} via SDK using {fn.__qualname__}")
            # Try to extract a returned artifact URI from the result, if available
            try:
                if isinstance(res, dict) and "uri" in res:
                    return res["uri"]
                if hasattr(res, "uri"):
                    return getattr(res, "uri")
            except Exception:
                pass
            # Best-effort fallback: return a generic artifact URI using artifact_name
            return f"artifact://{artifact_name}/v1"
        except Exception as e:
            print(f"SDK publish attempt using {getattr(fn, '__qualname__', fn)} failed: {e}", file=sys.stderr)
            continue

    # If we couldn't find a publish helper, try using the artifact service
    try:
        svc_uri = _publish_with_sdk_service(adk_module, file_path, artifact_name, dry_run, project)
        if svc_uri:
            return svc_uri
    except Exception as e:
        print(f"artifact-service publish attempt failed: {e}", file=sys.stderr)

    print("No usable SDK publish method discovered; falling back to CLI.", file=sys.stderr)
    return None


def publish_with_cli(file_path: str, artifact_name: str, project: Optional[str], dry_run: bool, artifact_map: Optional[dict] = None) -> Optional[str]:
    """Call the `adk` CLI to push a single file as an artifact.

    Command structure (best-effort):
      adk artifacts push --file <file> --name <artifact_name> [--project <project>]

    The exact CLI flags may vary by ADK version; this script uses a conservative
    and commonly available form. Check your local `adk --help` for exact flags.
    """
    cmd = ["adk", "artifacts", "push", "--file", file_path]
    if artifact_name:
        cmd += ["--name", artifact_name]
    if project:
        cmd += ["--project", project]

    if dry_run:
        print("[dry-run] CLI would run:", " ".join(cmd))
        return f"artifact://{project or artifact_name}/{artifact_name}/v1"

    print("Running CLI:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if out.strip():
        print(out)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None

    # Try to construct a reasonable artifact URI. Prefer explicit project, else
    # try to discover project from existing artifact_map entries (if provided).
    # First, try to extract an artifact:// URI from the CLI output (stdout/stderr)
    m = re.search(r"artifact://[^\s'\"]+", out)
    if m:
        return m.group(0)

    # If stdout looks like JSON, try parsing and finding a 'uri' field
    try:
        j = json.loads(proc.stdout) if proc.stdout and (proc.stdout.strip().startswith("{") or proc.stdout.strip().startswith("[")) else None
        if j:
            # search for 'uri' keys in the JSON structure
            def find_uri(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() == "uri" and isinstance(v, str) and v.startswith("artifact://"):
                            return v
                        res = find_uri(v)
                        if res:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = find_uri(item)
                        if res:
                            return res
                return None

            uri = find_uri(j)
            if uri:
                return uri
    except Exception:
        pass

    proj = project
    if not proj and artifact_map:
        # look for any artifact:// URI and extract its project
        for v in artifact_map.get("schemas", {}).values():
            uri = v.get("uri") if isinstance(v, dict) else None
            if isinstance(uri, str) and uri.startswith("artifact://"):
                try:
                    proj = uri.split("/")[2]
                    break
                except Exception:
                    continue

    if not proj:
        proj = artifact_name

    return f"artifact://{proj}/schemas/{artifact_name}/v1"


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
