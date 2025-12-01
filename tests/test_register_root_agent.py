import sys
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = "scripts.register_root_agent"


@pytest.fixture(autouse=True)
def reload_module():
    if MODULE in sys.modules:
        del sys.modules[MODULE]
    mod = __import__(MODULE, fromlist=["*"])
    yield mod


def test_dry_run_prints_summary(reload_module, monkeypatch, capsys):
    mod = reload_module
    monkeypatch.setattr(sys, "argv", ["register_root_agent.py", "--config", "configs/root_agent.yaml", "--dry-run"])
    rc = mod.main()
    captured = capsys.readouterr()
    assert rc == 0
    assert "Dry run" in captured.out
    assert "sparkle-motion/production_root_agent" in captured.out


def test_register_with_cli_not_found(reload_module, monkeypatch):
    mod = reload_module
    monkeypatch.setattr(shutil, "which", lambda _: None)
    ok, msg = mod.register_with_cli(Path("configs/root_agent.yaml"))
    assert not ok
    assert "adk CLI not found" in msg


def test_register_with_cli_success(reload_module, monkeypatch):
    mod = reload_module
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/adk")

    class FakeProc:
        def __init__(self, rc=0):
            self.returncode = rc

    monkeypatch.setattr(subprocess, "run", lambda cmd, check=False: FakeProc(0))
    ok, msg = mod.register_with_cli(Path("configs/root_agent.yaml"))
    assert ok
    assert "Registered" in msg
