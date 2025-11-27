"""Basic unit test for `scripts/register_workflow.py` dry-run validation."""
from pathlib import Path
import importlib


def test_register_workflow_dryrun_loads_and_validates(tmp_path: Path):
    mod = importlib.import_module("scripts.register_workflow_local")
    wf_path = Path("configs") / "workflow_agent.yaml"
    assert wf_path.exists(), "configs/workflow_agent.yaml must exist for this test"
    workflow = mod.load_yaml(wf_path)
    ok = mod.validate_workflow(workflow, tool_registry=None, schema_artifacts=None)
    assert ok
