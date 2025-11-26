import os
import stat
import subprocess
import sys
from pathlib import Path
import yaml


def write_fake_adk(shim_path: Path, printed_uri: str):
    # shim supports MODE env var to emit different URI formats
    shim = f"""#!/usr/bin/env bash
MODE=${{MODE:-plain}}
case "$1 $2" in
  "artifacts push")
    case "$MODE" in
      plain)
        # plain-line stdout
        echo "Created {printed_uri}"
        ;;
      json)
        # JSON stdout with nested uri
        echo '{{"result": {{"uri": "{printed_uri}"}}}}'
        ;;
      stderr)
        # only stderr contains the uri
        >&2 echo "{printed_uri}"
        ;;
      multi)
        # produce multiple lines and both stdout/stderr
        echo "info: uploaded"
        echo '{{"uri": "{printed_uri}"}}'
        >&2 echo "note: fallback {printed_uri}-alt"
        ;;
      *)
        echo "Created {printed_uri}"
        ;;
    esac
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


def run_publish_with_shim(tmp_path: Path, mode: str):
    # prepare directories
    schemas = tmp_path / f"schemas_{mode}"
    schemas.mkdir(exist_ok=True)
    sample = schemas / "MoviePlan.schema.json"
    sample.write_text('{"$id":"MoviePlan","type":"object"}')

    cfg = tmp_path / f"schema_artifacts_{mode}.yaml"
    cfg.write_text("version: v1\nschemas: {}\n")

    # create fake adk shim
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    printed_uri = f"artifact://sdkproj/schemas/movie_plan/{mode}"
    shim_path = bin_dir / "adk"
    write_fake_adk(shim_path, printed_uri)

    env = os.environ.copy()
    env["MODE"] = mode
    # put shim bin at front of PATH
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    # ensure python can import scripts/ module
    cmd = [sys.executable, str(Path.cwd() / "scripts" / "publish_schemas.py"),
      "--schemas-dir", str(schemas), "--artifacts-config", str(cfg), "--project", "sdkproj", "--confirm", "--use-cli"]

    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    data = yaml.safe_load(cfg.read_text())
    assert "schemas" in data
    # Accept either snake_case key or original filename stem key; ensure an artifact:// uri exists
    uris = [v.get("uri") or v.get("artifact") for v in data["schemas"].values() if isinstance(v, dict)]
    assert any(isinstance(u, str) and u.startswith("artifact://") for u in uris), f"no artifact uri in {uris} for mode {mode}"


def test_publish_schemas_with_adk_shim(tmp_path: Path):
  # run multiple modes to exercise different CLI output formats
  for mode in ("plain", "json", "stderr", "multi"):
    run_publish_with_shim(tmp_path, mode)
