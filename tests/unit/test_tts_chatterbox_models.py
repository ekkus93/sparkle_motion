from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from sparkle_motion.function_tools.tts_chatterbox.models import TTSChatterboxRequest, TTSChatterboxResponse


mod = import_module("sparkle_motion.function_tools.tts_chatterbox.entrypoint")
make_app = getattr(mod, "make_app")


def test_request_model_requires_prompt():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TTSChatterboxRequest()


def test_invoke_writes_artifact_and_returns_success(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    app = make_app()
    client = TestClient(app)

    resp = client.post("/invoke", json={"prompt": "smoke"})
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "success"
    assert j.get("artifact_uri", "").startswith("file://")

    envelope = TTSChatterboxResponse(**j)
    assert envelope.artifact_uri == j["artifact_uri"]
