import json
from pathlib import Path

from fastapi.testclient import TestClient

from sparkle_motion.function_tools.script_agent import entrypoint


def test_invoke_persists_artifact(tmp_path, monkeypatch):
    """POST to /invoke with fixture-mode enabled and assert artifact saved."""
    # Ensure fixture-mode and deterministic artifact location
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("DETERMINISTIC", "1")

    app = entrypoint.app
    client = TestClient(app)

    payload = {"prompt": "unit-test-prompt"}
    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200, resp.text
    j = resp.json()
    assert j.get("status") == "success"
    artifact_uri = j.get("artifact_uri")
    assert artifact_uri and artifact_uri.startswith("file://")

    # Verify artifact file exists and contains the payload
    file_path = Path(artifact_uri.replace("file://", ""))
    assert file_path.exists()
    data = json.loads(file_path.read_text(encoding="utf-8"))
    request_payload = data.get("request") or {}
    assert request_payload.get("prompt") == "unit-test-prompt"
    plan = data.get("validated_plan") or {}
    assert isinstance(plan.get("shots"), list) and plan["shots"]
