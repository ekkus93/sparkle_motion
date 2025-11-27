import json
import subprocess
import sys
from pathlib import Path


from sparkle_motion.cli import run_workflow


def _run_wrapper(args, env=None):
    # Run the CLI wrapper script via subprocess, return CompletedProcess
    # prefer running the module to avoid executable bit issues in tests
    cmd = [sys.executable, "-m", "sparkle_motion.cli", *args]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def test_missing_workflow_file_returns_nonzero(tmp_path: Path):
    # point to a non-existent file using the wrapper
    cp = _run_wrapper([str(tmp_path / "no-such.yaml"), "--dry-run"], env={**dict(**{})})
    assert cp.returncode != 0


def test_invalid_yaml_returns_nonzero(tmp_path: Path):
    f = tmp_path / "bad.yaml"
    f.write_text("::: this is not yaml :::")
    cp = _run_wrapper([str(f), "--dry-run"])
    assert cp.returncode != 0


def test_tool_import_failure_returns_2(tmp_path: Path, monkeypatch):
    # Create a workflow pointing at a non-existent tool
    wf = {
        "stages": [
            {"id": "s1", "tool_id": "no_such_tool:local"}
        ]
    }
    p = tmp_path / "wf.yaml"
    p.write_text(json.dumps(wf))

    # run in-process so we can inspect return code directly
    rc = run_workflow(p, tmp_path / "out", dry_run=False)
    assert rc == 2


def test_stage_invoke_failure_returns_nonzero(tmp_path: Path, monkeypatch):
    # Monkeypatch an existing tool entrypoint to return an app that 500s on /invoke
    mod_path = "sparkle_motion.function_tools.script_agent.entrypoint"
    mod = __import__(mod_path, fromlist=["*"])

    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/invoke")
    def bad_invoke():
        return ("error", 500)

    # replace make_app or app
    if hasattr(mod, "make_app"):
        monkeypatch.setattr(mod, "make_app", lambda: app)
    else:
        monkeypatch.setattr(mod, "app", app)

    wf = {"stages": [{"id": "s1", "tool_id": "script_agent:local"}]}
    p = tmp_path / "wf2.yaml"
    p.write_text(json.dumps(wf))

    rc = run_workflow(p, tmp_path / "out", dry_run=False)
    # run_workflow returns 3 when a stage returns non-200
    assert rc in (3, 4)


def test_cli_wrapper_executes_dryrun(tmp_path: Path):
    cfg = Path("configs/workflow_agent.yaml")
    assert cfg.exists()
    cp = _run_wrapper([str(cfg), "--dry-run"])
    assert cp.returncode == 0, cp.stderr.decode()[:400]
