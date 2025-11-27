from pathlib import Path

from sparkle_motion.cli import run_workflow


def test_cli_dryrun_works():
    cfg = Path("configs/workflow_agent.yaml")
    assert cfg.exists(), "workflow config must exist for this test"
    rc = run_workflow(cfg, Path("./out-test"), dry_run=True)
    assert rc == 0
