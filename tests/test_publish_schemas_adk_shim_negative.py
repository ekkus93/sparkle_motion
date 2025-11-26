import os
import stat
import subprocess
import sys
from pathlib import Path
import yaml


def test_publish_handles_nonzero_adk_exit(tmp_path: Path):
    """Simulate `adk` returning a non-zero exit code and ensure the
    publisher does not write artifact entries (artifacts config stays empty).
    """
    # prepare schema
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    sample = schemas / "MoviePlan.schema.json"
    sample.write_text('{"$id":"MoviePlan","type":"object"}')

    # empty artifacts config
    cfg = tmp_path / "schema_artifacts.yaml"
    cfg.write_text("version: v1\nschemas: {}\n")

    # create fake adk shim that fails with non-zero exit and stderr
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim_path = bin_dir / "adk"
    shim_path.write_text("""#!/usr/bin/env bash
case "$1 $2" in
  "artifacts push")
    >&2 echo "error: upload failed"
    exit 2
    ;;
  *)
    >&2 echo "adk shim: unhandled args: $@"
    exit 2
    ;;
esac
""")
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

    cmd = [
        sys.executable,
        str(Path.cwd() / "scripts" / "publish_schemas.py"),
        "--schemas-dir",
        str(schemas),
        "--artifacts-config",
        str(cfg),
        "--project",
        "sdkproj",
        "--confirm",
        "--use-cli",
    ]

    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # the publisher should exit non-zero when the CLI fails
    assert proc.returncode != 0, f"expected non-zero exit, stdout: {proc.stdout}\nstderr: {proc.stderr}"

    # artifacts config must remain unchanged (empty schemas)
    data = yaml.safe_load(cfg.read_text())
    assert "schemas" in data
    assert data["schemas"] == {}, f"artifacts config changed on failed publish: {data}"
