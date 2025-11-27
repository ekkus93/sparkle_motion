import json


from sparkle_motion.adk_helpers import publish_with_cli


class Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_publish_with_cli_dry_run(tmp_path):
    uri = publish_with_cli(str(tmp_path / "x.schema.json"), "test_art", "proj", True, None)
    assert uri.startswith("artifact://")


def test_publish_with_cli_extracts_uri_from_stdout(monkeypatch):
    expected = "artifact://proj/schemas/test_art/v2"

    def fake_run(cmd, stdout, stderr, text):
        return Proc(stdout=expected + "\n", stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    uri = publish_with_cli("/tmp/x.schema.json", "test_art", "proj", False, None)
    assert uri == expected


def test_publish_with_cli_parses_json_stdout(monkeypatch):
    j = {"result": {"uri": "artifact://proj/schemas/test_art/v9"}}

    def fake_run(cmd, stdout, stderr, text):
        return Proc(stdout=json.dumps(j), stderr="", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    uri = publish_with_cli("/tmp/x.schema.json", "test_art", None, False, None)
    assert uri == "artifact://proj/schemas/test_art/v9"
