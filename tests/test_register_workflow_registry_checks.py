import importlib
from pathlib import Path
import sys
import pytest


mod = importlib.import_module("scripts.register_workflow_local")


def load_workflow_fixture() -> dict:
    p = Path("configs") / "workflow_agent.yaml"
    return mod.load_yaml(p)


def test_validate_workflow_with_no_tool_registry_warns(capsys):
    wf = load_workflow_fixture()
    ok = mod.validate_workflow(wf, tool_registry=None)
    captured = capsys.readouterr()
    # current validator returns True for structure but may warn when registry missing
    assert ok is True
    # no error message on stderr for missing registry since it is optional, but shape must be present
    assert "Workflow must include" not in captured.err


def test_validate_workflow_with_mismatched_registry_warns(capsys):
    wf = load_workflow_fixture()
    # provide a registry intentionally missing some tool ids
    tool_registry = {"script_agent:local-colab": {}}
    ok = mod.validate_workflow(wf, tool_registry=tool_registry)
    captured = capsys.readouterr()
    assert ok is True
    assert "not found in tool registry" in captured.err


def test_validate_workflow_malformed_fails(capsys):
    bad_wf = {"stages": [{"id": "only-id"}]}
    ok = mod.validate_workflow(bad_wf)
    captured = capsys.readouterr()
    assert ok is False
    assert "missing required key" in captured.err
