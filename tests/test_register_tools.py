import builtins
import types
import sys
import shutil
import subprocess

import pytest

import importlib


REG_MODULE = "scripts.register_tools"


@pytest.fixture(autouse=True)
def reload_module():
    # Ensure we import a fresh copy for each test to avoid cross-test sys.modules state
    if REG_MODULE in sys.modules:
        del sys.modules[REG_MODULE]
    mod = importlib.import_module(REG_MODULE)
    yield mod


def test_dry_run_prints(reload_module, monkeypatch, capsys):
    rt = reload_module

    # Provide a minimal registry via monkeypatching the loader
    monkeypatch.setattr(rt, "load_tool_registry", lambda path=None: {
        "tools": {
            "toolA": {
                "description": "A sample tool",
                "endpoints": {"local-colab": "http://localhost:9000"},
                "schemas": {"MoviePlan": {}},
            }
        }
    })

    monkeypatch.setattr(sys, "argv", ["register_tools.py", "--dry-run"])
    rc = rt.main()
    captured = capsys.readouterr()
    assert rc == 0
    assert "Dry run" in captured.out
    assert "toolA" in captured.out


def test_register_with_sdk_import_error(reload_module, monkeypatch):
    rt = reload_module
    # Force imports of google.adk to fail inside the function by patching __import__
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google.adk" or (name == "google" and "adk" in fromlist):
            raise ModuleNotFoundError("No module named google.adk")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        ok, msg = rt.register_with_sdk("x", {})
        assert not ok
        assert "SDK not importable" in msg
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)


def test_register_with_sdk_no_known_api(reload_module, monkeypatch):
    rt = reload_module
    # Inject a google.adk module with no registration entrypoints
    adk_mod = types.ModuleType("google.adk")
    # Ensure import finds our fake module
    monkeypatch.setitem(sys.modules, "google.adk", adk_mod)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))

    ok, msg = rt.register_with_sdk("t1", {})
    assert not ok
    assert "no known registration api" in msg.lower()


def test_register_with_sdk_success_probe(reload_module, monkeypatch):
    rt = reload_module
    # Build a fake adk module exposing tools.register
    adk_mod = types.ModuleType("google.adk")

    def fake_register(tool_id, meta):
        return {"registered": tool_id}

    tools_ns = types.SimpleNamespace(register=fake_register)
    setattr(adk_mod, "tools", tools_ns)
    monkeypatch.setitem(sys.modules, "google.adk", adk_mod)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))

    ok, msg = rt.register_with_sdk("tool-success", {"a": 1})
    assert ok
    assert "SDK: called tools.register" in msg


def test_register_with_cli_not_found(reload_module, monkeypatch):
    rt = reload_module
    monkeypatch.setattr(shutil, "which", lambda p: None)
    ok, msg = rt.register_with_cli("tcli", {})
    assert not ok
    assert "adk CLI not found" in msg


def test_register_with_cli_success(monkeypatch, reload_module):
    rt = reload_module
    # Pretend adk exists in PATH
    monkeypatch.setattr(shutil, "which", lambda p: "/usr/bin/adk")

    class FakeProc:
        def __init__(self, rc=0):
            self.returncode = rc

    def fake_run(cmd, check=False):
        return FakeProc(0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, msg = rt.register_with_cli("tcli_ok", {"x": 1})
    assert ok
    assert "registered" in msg


def test_main_falls_back_to_cli_when_sdk_missing(monkeypatch, reload_module, capsys):
    rt = reload_module
    # Provide a single tool in the registry
    monkeypatch.setattr(rt, "load_tool_registry", lambda path=None: {
        "tools": {
            "toolX": {"description": "x", "endpoints": {"local-colab": "http://x"}}
        }
    })

    # Force import failure for google.adk inside register_with_sdk
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google.adk" or (name == "google" and "adk" in fromlist):
            raise ModuleNotFoundError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Ensure CLI path is available and returns success
    monkeypatch.setattr(shutil, "which", lambda p: "/usr/bin/adk")

    class FakeProc:
        def __init__(self, rc=0):
            self.returncode = rc

    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: FakeProc(0))

    monkeypatch.setattr(sys, "argv", ["register_tools.py"])
    try:
        rc = rt.main()
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)

    out = capsys.readouterr().out
    assert rc == 0
    assert "CLI attempt" in out
    assert "registered" in out
