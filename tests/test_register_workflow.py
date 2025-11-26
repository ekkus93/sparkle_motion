import builtins
import types
import sys
import shutil
import subprocess

import pytest

REG_MODULE = "scripts.register_workflow"


@pytest.fixture(autouse=True)
def reload_module():
    if REG_MODULE in sys.modules:
        del sys.modules[REG_MODULE]
    mod = __import__(REG_MODULE, fromlist=["*"])
    yield mod


def test_dry_run_prints(reload_module, monkeypatch, capsys):
    rw = reload_module
    monkeypatch.setattr(rw, "load_workflow_registry", lambda path=None: {
        "workflows": {
            "wfA": {"description": "sample", "spec": {"steps": []}}
        }
    })
    monkeypatch.setattr(sys, "argv", ["register_workflow.py", "--dry-run", "--path", "dummy"])
    rc = rw.main()
    captured = capsys.readouterr()
    assert rc == 0
    assert "Dry run" in captured.out
    assert "wfA" in captured.out


def test_register_with_sdk_import_error(reload_module, monkeypatch):
    rw = reload_module
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google.adk" or (name == "google" and "adk" in fromlist):
            raise ModuleNotFoundError("No module named google.adk")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        ok, msg = rw.register_with_sdk("x", {})
        assert not ok
        assert "SDK not importable" in msg
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)


def test_register_with_sdk_no_known_api(reload_module, monkeypatch):
    rw = reload_module
    adk_mod = types.ModuleType("google.adk")
    monkeypatch.setitem(sys.modules, "google.adk", adk_mod)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    ok, msg = rw.register_with_sdk("w1", {})
    assert not ok
    assert "no known registration api" in msg.lower()


def test_register_with_sdk_success_probe(reload_module, monkeypatch):
    rw = reload_module
    adk_mod = types.ModuleType("google.adk")

    def fake_register(wid, meta):
        return {"registered": wid}

    workflows_ns = types.SimpleNamespace(register=fake_register)
    setattr(adk_mod, "workflows", workflows_ns)
    monkeypatch.setitem(sys.modules, "google.adk", adk_mod)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    ok, msg = rw.register_with_sdk("wf-success", {"a": 1})
    assert ok
    assert "SDK: called workflows.register" in msg


def test_register_with_cli_not_found(reload_module, monkeypatch):
    rw = reload_module
    monkeypatch.setattr(shutil, "which", lambda p: None)
    ok, msg = rw.register_with_cli("wcli", {})
    assert not ok
    assert "adk CLI not found" in msg


def test_register_with_cli_success(monkeypatch, reload_module):
    rw = reload_module
    monkeypatch.setattr(shutil, "which", lambda p: "/usr/bin/adk")

    class FakeProc:
        def __init__(self, rc=0):
            self.returncode = rc

    def fake_run(cmd, check=False):
        return FakeProc(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, msg = rw.register_with_cli("wcli_ok", {"x": 1})
    assert ok
    assert "registered" in msg


def test_main_falls_back_to_cli_when_sdk_missing(monkeypatch, reload_module, capsys):
    rw = reload_module
    monkeypatch.setattr(rw, "load_workflow_registry", lambda path=None: {
        "workflows": {"wfX": {"description": "x", "spec": {}}}
    })

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google.adk" or (name == "google" and "adk" in fromlist):
            raise ModuleNotFoundError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(shutil, "which", lambda p: "/usr/bin/adk")
    class FakeProc:
        def __init__(self, rc=0):
            self.returncode = rc

    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: FakeProc(0))
    monkeypatch.setattr(sys, "argv", ["register_workflow.py", "--path", "dummy"])
    try:
        rc = rw.main()
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)

    out = capsys.readouterr().out
    assert rc == 0
    assert "CLI attempt" in out
    assert "registered" in out
