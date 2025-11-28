from pathlib import Path

import json

from sparkle_motion.cli import run_workflow


def test_cli_run_in_process(tmp_path: Path):
    cfg = Path("configs/workflow_agent.yaml")
    assert cfg.exists()
    out = tmp_path / "out"
    rc = run_workflow(cfg, out, dry_run=False)
    assert rc == 0
    mf = out / "run_manifest.json"
    assert mf.exists()
    data = json.loads(mf.read_text(encoding="utf-8"))
    assert "stages" in data and len(data["stages"]) > 0
    for stage in data["stages"]:
        resp_path = stage.get("response_path")
        assert resp_path is not None
        assert Path(resp_path).exists()
        artifact_path = stage.get("artifact_payload_path")
        if artifact_path:
            assert Path(artifact_path).exists()
