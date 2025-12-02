import os
import pytest

from fastapi.testclient import TestClient

from tests.unit.utils import assert_backend_artifact_uri, expected_artifact_scheme

from src.sparkle_motion.function_tools.script_agent.entrypoint import (
    RequestModel,
    ResponseModel,
    make_app,
)


def test_request_model_requires_one_field():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel()


def test_request_model_accepts_prompt_title_or_shots():
    r1 = RequestModel(prompt="Hello")
    assert r1.prompt == "Hello"
    r2 = RequestModel(title="Title")
    assert r2.title == "Title"
    r3 = RequestModel(shots=[{"a": 1}])
    assert isinstance(r3.shots, list)


def test_response_model_fields_and_types():
    rm = ResponseModel(status="success", artifact_uri="file:///tmp/x.json", request_id="abc")
    d = rm.model_dump() if hasattr(rm, "model_dump") else rm.dict()
    assert d["status"] == "success"
    assert d["artifact_uri"].startswith("file://")
    assert d["request_id"] == "abc"


def test_invoke_writes_artifact_and_returns_success(monkeypatch, tmp_path):
    # ensure deterministic artifact naming and fixture mode
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

    app = make_app()
    client = TestClient(app)

    payload = {"prompt": "Test artifact content"}
    resp = client.post("/invoke", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    assert "artifact_uri" in data
    uri = data["artifact_uri"]
    assert_backend_artifact_uri(uri)

    backend = expected_artifact_scheme()
    local_path: str | None
    if backend == "filesystem":
        local_path = data.get("artifact_metadata", {}).get("local_path")
    else:
        local_path = uri[len("file://"):]
    if not local_path:
        return
    assert os.path.exists(local_path)
    # verify content matches
    with open(local_path, "r", encoding="utf-8") as fh:
        txt = fh.read()
    # JSON stored should contain prompt
    assert "Test artifact content" in txt
