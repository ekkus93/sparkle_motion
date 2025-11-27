from __future__ import annotations

import types


from sparkle_motion.adk_helpers import (
    run_adk_cli,
    register_entity_with_cli,
    register_entity_with_sdk,
)
from sparkle_motion.adk_helpers import publish_with_cli


def test_run_adk_cli_dry_run(capsys):
    rc, out, err = run_adk_cli(["adk", "version"], dry_run=True)
    assert rc == 0
    assert out == ""
    assert err == ""


def test_run_adk_cli_calls_subprocess(monkeypatch):
    class FakeProc:
        def __init__(self):
            self.returncode = 0
            self.stdout = "ok"
            self.stderr = ""

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        return FakeProc()

    monkeypatch.setattr("subprocess.run", fake_run)
    rc, out, err = run_adk_cli(["adk", "whoami"], dry_run=False)
    assert rc == 0
    assert out == "ok"
    assert err == ""


def test_register_entity_with_cli_parses_json_stdout(monkeypatch):
    def fake_run(cmd, dry_run=False):
        return 0, '{"uri": "artifact://proj/schemas/foo/v1"}', ""

    monkeypatch.setattr("sparkle_motion.adk_helpers.run_adk_cli", lambda cmd, dry_run=False: (0, '{"uri": "artifact://proj/schemas/foo/v1"}', ""))
    uri = register_entity_with_cli(["adk", "tools", "register"], dry_run=False)
    assert uri == "artifact://proj/schemas/foo/v1"


def test_register_entity_with_cli_token_fallback(monkeypatch):
    monkeypatch.setattr("sparkle_motion.adk_helpers.run_adk_cli", lambda cmd, dry_run=False: (0, "registered: artifact://proj/schemas/bar/v2", ""))
    uri = register_entity_with_cli(["adk", "tools", "register"], dry_run=False)
    assert uri is not None
    assert uri.startswith("artifact://proj/schemas/bar")


def test_register_entity_with_cli_nonzero(monkeypatch):
    monkeypatch.setattr("sparkle_motion.adk_helpers.run_adk_cli", lambda cmd, dry_run=False: (1, "", "error"))
    uri = register_entity_with_cli(["adk", "tools", "register"], dry_run=False)
    assert uri is None


def test_register_entity_with_sdk_success(monkeypatch):
    # Build a fake adk module with a `tools.register` that returns a dict
    adk_mod = types.ModuleType("google.adk")

    def fake_register(name, payload):
        return {"uri": f"tool://{name}"}

    tools_ns = types.SimpleNamespace(register=fake_register)
    setattr(adk_mod, "tools", tools_ns)

    # Call the helper with the fake module
    res = register_entity_with_sdk(adk_mod, {"name": "t1"}, entity_kind="tool", name="t1", dry_run=False)
    assert res == "tool://t1"


def test_register_entity_with_sdk_dry_run():
    # dry-run should return a dry-run:// URI when a plausible SDK method exists
    adk_mod = types.ModuleType("google.adk")
    # provide a plausible hub with a register method so the dry-run path triggers
    tools_ns = types.SimpleNamespace(register_tool=lambda name, payload: None)
    setattr(adk_mod, "ToolRegistry", tools_ns)
    res = register_entity_with_sdk(adk_mod, {}, entity_kind="tool", name="dryme", dry_run=True)
    assert res is not None
    assert res.startswith("dry-run://sdk/")


def test_run_adk_cli_fallback_and_register_cli_handles_typeerror(monkeypatch):
    # Simulate run_adk_cli raising TypeError, and subprocess.run being used as fallback
    def fake_run_simple(cmd, check=False):
        class P:
            returncode = 0
            stdout = '{"uri": "artifact://x/schemas/y/v1"}'
            stderr = ""

        return P()

    monkeypatch.setattr("sparkle_motion.adk_helpers.run_adk_cli", lambda cmd, dry_run=False: (_ for _ in ()).throw(TypeError("bad signature")))
    monkeypatch.setattr("sparkle_motion.adk_helpers.subprocess.run", fake_run_simple)
    uri = register_entity_with_cli(["adk", "something"], dry_run=False)
    assert uri == "artifact://x/schemas/y/v1"


def test_register_entity_with_sdk_object_with_id_and_uri():
    adk_mod = types.ModuleType("google.adk")

    class R:
        def __init__(self):
            self.id = "id-123"

    tools_ns = types.SimpleNamespace(register=lambda name, payload: R())
    setattr(adk_mod, "tools", tools_ns)
    res = register_entity_with_sdk(adk_mod, {}, entity_kind="tool", name="id-entity", dry_run=False)
    assert res == "id-123"


def test_publish_with_cli_parses_json_and_token(monkeypatch, tmp_path):
    # JSON stdout
    # publish_with_cli uses subprocess.run internally; patch that in the adk_helpers namespace
    def fake_proc_json(cmd, stdout=None, stderr=None, text=None):
        class P:
            returncode = 0
            stdout = '{"uri":"artifact://proj/schemas/z/v1"}'
            stderr = ""

        return P()

    monkeypatch.setattr("sparkle_motion.adk_helpers.subprocess.run", fake_proc_json)
    uri = publish_with_cli(str(tmp_path / "f.json"), "z", "proj", dry_run=False, artifact_map=None)
    assert uri == "artifact://proj/schemas/z/v1"

    # token fallback in stdout
    def fake_proc_token(cmd, stdout=None, stderr=None, text=None):
        class P:
            returncode = 0
            stdout = "registered artifact://proj/schemas/w/v2"
            stderr = ""

        return P()

    monkeypatch.setattr("sparkle_motion.adk_helpers.subprocess.run", fake_proc_token)
    uri2 = publish_with_cli(str(tmp_path / "g.json"), "w", "proj", dry_run=False, artifact_map=None)
    assert uri2.startswith("artifact://proj/schemas/w")
