import subprocess
import sys
from pathlib import Path
import yaml


def run_script(args, input_bytes=None, cwd=None):
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "publish_schemas.py")] + args
    proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    return proc


def test_backup_and_confirm_creates_backup(tmp_path):
    # Setup: create a sample schema and an existing artifacts config
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    sample = schemas_dir / "MoviePlan.schema.json"
    sample.write_text('{"$id": "MoviePlan", "type": "object"}')

    cfg = tmp_path / "configs.yaml"
    cfg.write_text("schemas: {}\n")

    # Run script with --local-only --backup --confirm
    proc = run_script(["--schemas-dir", str(schemas_dir), "--artifacts-config", str(cfg), "--local-only", "--backup", "--confirm"]) 
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    # backup file should exist
    bak_files = list(tmp_path.glob("configs.yaml.bak.*"))
    assert len(bak_files) == 1

    # artifacts config should have been updated with file:// uri
    data = yaml.safe_load(cfg.read_text())
    assert "schemas" in data and any("file://" in v.get("uri", "") for v in data["schemas"].values())


def test_abort_without_confirm(tmp_path):
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    sample = schemas_dir / "AssetRefs.schema.json"
    sample.write_text('{"$id": "AssetRefs", "type": "object"}')

    cfg = tmp_path / "configs.yaml"
    cfg.write_text("schemas: {existing: {uri: 'artifact://x'}}\n")

    # Run script with --local-only but do NOT pass --confirm; send 'n' to the prompt
    proc = run_script(["--schemas-dir", str(schemas_dir), "--artifacts-config", str(cfg), "--local-only"], input_bytes="n\n")
    # Expect non-zero return code and no backup created
    assert proc.returncode == 5
    bak_files = list(tmp_path.glob("configs.yaml.bak.*"))
    assert len(bak_files) == 0
    # original config unchanged
    txt = cfg.read_text()
    assert "artifact://x" in txt
