from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _tool_module_from_tool_id(tool_id: str) -> str:
    # map e.g. 'images_sdxl:local-colab' -> 'sparkle_motion.function_tools.images_sdxl.entrypoint'
    name = tool_id.split(":", 1)[0]
    return f"sparkle_motion.function_tools.{name}.entrypoint"


def validate_workflow_dryrun(path: Path) -> bool:
    from scripts import register_workflow_local as validator

    wf = load_yaml(path)
    try:
        return bool(validator.validate_workflow(wf))
    except TypeError:
        # older/newer signatures may accept more args
        return bool(validator.validate_workflow(wf, tool_registry=None, schema_artifacts=None))


def run_workflow(path: Path, outdir: Path, *, dry_run: bool = True) -> int:
    wf = load_yaml(path)
    if dry_run:
        ok = validate_workflow_dryrun(path)
        print("Dry-run OK" if ok else "Dry-run FAILED")
        return 0 if ok else 1

    # run in-process: sequential simple runner for local development
    os.environ.setdefault("ADK_USE_FIXTURE", "1")
    os.environ.setdefault("DETERMINISTIC", "1")

    outdir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"workflow": str(path), "stages": []}

    for stage in wf.get("stages", []):
        sid = stage.get("id")
        tool_id = stage.get("tool_id")
        print(f"Running stage: {sid} -> {tool_id}")
        mod_path = _tool_module_from_tool_id(tool_id)
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except Exception as e:
            print(f"Failed to import tool module {mod_path}: {e}")
            return 2

        app = mod.make_app() if hasattr(mod, "make_app") else getattr(mod, "app")

        # use TestClient to exercise invoke endpoint in-process
        try:
            from fastapi.testclient import TestClient

            client = TestClient(app)
            payload = {"prompt": f"operator-run:{sid}"}
            r = client.post("/invoke", json=payload)
            if r.status_code != 200:
                print(f"Stage {sid} failed: {r.status_code} {r.text}")
                return 3
            data = r.json()
            manifest["stages"].append({"id": sid, "artifact_uri": data.get("artifact_uri")})
            print(f"Stage {sid} completed, artifact={data.get('artifact_uri')}")
        except Exception as e:
            print(f"Stage {sid} runtime error: {e}")
            return 4

        time.sleep(0.01)

    # write manifest
    mf = outdir / "run_manifest.json"
    with mf.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Workflow run completed. Manifest: {mf}")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("file", help="Path to workflow YAML")
    p.add_argument("--out", default="./out", help="Output directory for artifacts/manifest")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    raise SystemExit(run_workflow(Path(args.file), Path(args.out), dry_run=args.dry_run))
