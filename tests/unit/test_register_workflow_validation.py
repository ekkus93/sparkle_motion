import importlib
from pathlib import Path
import pytest


mod = importlib.import_module("scripts.register_workflow_local")


def load_workflow_fixture() -> dict:
    p = Path("configs") / "workflow_agent.yaml"
    return mod.load_yaml(p)


def test_validate_workflow_missing_tool_warns(capsys):
    wf = load_workflow_fixture()
    # craft a tool registry that is missing the 'images_sdxl:local-colab' entry
    tool_registry = {"script_agent:local-colab": {"id": "script_agent:local-colab"}}

    ok = mod.validate_workflow(wf, tool_registry=tool_registry)
    captured = capsys.readouterr()
    # validator should return True (structure intact) but warn about missing tool ids
    assert ok is True
    assert "not found in tool registry" in captured.err


def test_validate_workflow_all_tools_present_no_warnings(capsys):
    wf = load_workflow_fixture()
    # include all tool_ids referenced in the workflow
    tool_registry = {s["tool_id"]: {} for s in wf.get("stages", [])}

    ok = mod.validate_workflow(wf, tool_registry=tool_registry)
    captured = capsys.readouterr()
    assert ok is True
    assert captured.err == ""


def test_validate_workflow_missing_required_key_fails(capsys):
    # stage missing 'tool_id' should fail validation
    bad_wf = {"stages": [{"id": "only-id", "input_schema": "foo", "output_schema": "bar"}]}
    ok = mod.validate_workflow(bad_wf)
    captured = capsys.readouterr()
    assert ok is False
    assert "missing required key" in captured.err
