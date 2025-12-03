from pathlib import Path

import json

from sparkle_motion import cli
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


def _production_stage(**overrides):
    stage = {
        "id": "production_stage",
        "tool_id": "production_agent:local-cli",
    }
    stage.update(overrides)
    return stage


def _stage_outputs_with_plan():
    return {"script": {"artifact_payload": cli._fallback_movie_plan()}}


def test_build_stage_payload_excludes_qa_mode():
    payload = cli._build_stage_payload(_production_stage(), _stage_outputs_with_plan(), "production_stage")
    assert "qa_mode" not in payload

