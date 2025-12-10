import importlib
import json
import sys

import pytest


REG_MODULE = "scripts.register_tools"


@pytest.fixture(autouse=True)
def reload_module():
    if REG_MODULE in sys.modules:
        del sys.modules[REG_MODULE]
    yield importlib.import_module(REG_MODULE)


def test_try_register_with_sdk_returns_none_when_probe_missing(reload_module, monkeypatch):
    rt = reload_module
    monkeypatch.setattr(rt, "probe_sdk", lambda: None)
    assert rt.try_register_with_sdk({"id": "tool"}, dry_run=False) is None


def test_try_register_with_sdk_success(monkeypatch, reload_module):
    rt = reload_module
    probe_calls = {}

    def fake_probe():
        probe_calls["called"] = True
        return object(), {}

    def fake_register(adk_mod, tool, *, entity_kind, name, dry_run):  # type: ignore[override]
        assert entity_kind == "tool"
        assert name == tool["id"]
        assert not dry_run
        payload = json.loads(json.dumps(tool))
        assert payload["id"] == "tool-success"
        return "sdk://tool-success"

    monkeypatch.setattr(rt, "probe_sdk", fake_probe)
    monkeypatch.setattr(rt, "register_entity_with_sdk", fake_register)

    result = rt.try_register_with_sdk({"id": "tool-success"}, dry_run=False)
    assert probe_calls["called"] is True
    assert result == "sdk://tool-success"


def test_register_with_cli_dry_run(reload_module):
    rt = reload_module
    tool = {"id": "demo"}
    result = rt.register_with_cli(tool, dry_run=True)
    assert result == "dry-run://cli/demo"


def test_register_with_cli_invokes_helper(monkeypatch, reload_module):
    rt = reload_module
    captured = {}

    def fake_register(cmd, *, dry_run):  # type: ignore[override]
        captured["cmd"] = cmd
        captured["dry_run"] = dry_run
        return "cli://demo"

    monkeypatch.setattr(rt, "register_entity_with_cli", fake_register)
    result = rt.register_with_cli({"id": "demo"}, dry_run=False)
    assert result == "cli://demo"
    assert captured["cmd"][0:4] == ["adk", "tools", "register", "--file"]
    assert captured["dry_run"] is False


def test_main_falls_back_to_cli_when_sdk_returns_none(monkeypatch, reload_module, tmp_path, capsys):
    rt = reload_module
    config_path = tmp_path / "tools.yaml"
    config_path.write_text("tools:\n  toolX:\n    description: t\n")

    monkeypatch.setattr(rt, "load_config", lambda path: {
        "tools": {
            "toolX": {"description": "test", "schemas": {}}
        }
    })

    sdk_calls = []
    cli_calls = []

    def fake_sdk(tool, dry_run):  # type: ignore[override]
        sdk_calls.append(tool["id"])
        return None

    def fake_cli(tool, dry_run):  # type: ignore[override]
        cli_calls.append(tool["id"])
        return f"cli://{tool['id']}"

    monkeypatch.setattr(rt, "try_register_with_sdk", fake_sdk)
    monkeypatch.setattr(rt, "register_with_cli", fake_cli)

    monkeypatch.setattr(sys, "argv", [
        "register_tools.py",
        "--config",
        str(config_path),
        "--confirm",
    ])

    rc = rt.main()
    capsys.readouterr()

    assert rc == 0
    assert sdk_calls == ["toolX"]
    assert cli_calls == ["toolX"]
