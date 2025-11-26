import os
import stat
import subprocess
import sys
from pathlib import Path
import yaml


def write_fake_adk(shim_path: Path, printed_uri: str):
    shim = f"""#!/usr/bin/env bash
case "$1 $2" in
  "artifacts push")
    # simulate CLI printing the artifact URI after upload
    echo "Created {printed_uri}"
    exit 0
    ;;
  *)
    echo "adk shim: unhandled args: $@" >&2
    exit 2
    ;;
esac
"""
    shim_path.write_text(shim)
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def run_publish_with_shim(tmp_path: Path):
    # prepare directories
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    sample = schemas / "MoviePlan.schema.json"
    sample.write_text('{"$id":"MoviePlan","type":"object"}')

    cfg = tmp_path / "schema_artifacts.yaml"
    cfg.write_text("version: v1\nschemas: {}\n")

    # create fake adk shim
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    printed_uri = "artifact://sdkproj/schemas/movie_plan/v1"
    shim_path = bin_dir / "adk"
    write_fake_adk(shim_path, printed_uri)

    env = os.environ.copy()
    # put shim bin at front of PATH
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    # ensure python can import scripts/ module
    cmd = [sys.executable, str(Path.cwd() / "scripts" / "publish_schemas.py"),
           "--schemas-dir", str(schemas), "--artifacts-config", str(cfg), "--project", "sdkproj", "--confirm"]

    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    data = yaml.safe_load(cfg.read_text())
    assert "schemas" in data
    # Accept either snake_case key or original filename stem key; ensure an artifact:// uri exists
    uris = [v.get("uri") for v in data["schemas"].values() if isinstance(v, dict)]
    assert any(isinstance(u, str) and u.startswith("artifact://") for u in uris)


def test_publish_schemas_with_adk_shim(tmp_path: Path):
    run_publish_with_shim(tmp_path)
