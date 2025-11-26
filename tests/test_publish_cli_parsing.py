from types import SimpleNamespace
from scripts.publish_schemas import publish_with_cli


def make_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_plaintext_stdout_contains_artifact_uri(monkeypatch, tmp_path):
    # stdout contains an artifact:// URI in plain text
    proc = make_proc(stdout="Uploaded: artifact://sparkle-motion/schemas/movie_plan/v1\nDone", stderr="", returncode=0)

    def fake_run(cmd, stdout, stderr, text):
        return proc

    monkeypatch.setattr("subprocess.run", fake_run)

    uri = publish_with_cli(str(tmp_path / "f.json"), "movie_plan", project="sparkle-motion", dry_run=False, artifact_map=None)
    assert uri == "artifact://sparkle-motion/schemas/movie_plan/v1"


def test_json_stdout_with_uri_field(monkeypatch, tmp_path):
    # stdout contains JSON with a nested 'uri' field
    json_out = '{"result": {"uri": "artifact://sparkle-motion/schemas/movie_plan/v2"}}'
    proc = make_proc(stdout=json_out, stderr="", returncode=0)

    def fake_run(cmd, stdout, stderr, text):
        return proc

    monkeypatch.setattr("subprocess.run", fake_run)

    uri = publish_with_cli(str(tmp_path / "f.json"), "movie_plan", project=None, dry_run=False, artifact_map=None)
    assert uri == "artifact://sparkle-motion/schemas/movie_plan/v2"


def test_stderr_only_uri(monkeypatch, tmp_path):
    # stderr contains the artifact URI
    proc = make_proc(stdout="", stderr="Error: created artifact://acme/schemas/a/v1\n", returncode=0)

    def fake_run(cmd, stdout, stderr, text):
        return proc

    monkeypatch.setattr("subprocess.run", fake_run)

    uri = publish_with_cli(str(tmp_path / "f.json"), "a", project="acme", dry_run=False, artifact_map=None)
    assert uri == "artifact://acme/schemas/a/v1"


def test_no_uri_constructs_best_effort(monkeypatch, tmp_path):
    # No artifact URI in output; should construct using provided project
    proc = make_proc(stdout="OK\n", stderr="", returncode=0)

    def fake_run(cmd, stdout, stderr, text):
        return proc

    monkeypatch.setattr("subprocess.run", fake_run)

    uri = publish_with_cli(str(tmp_path / "f.json"), "checkpoint", project="sparkle-motion", dry_run=False, artifact_map=None)
    assert uri == "artifact://sparkle-motion/schemas/checkpoint/v1"
