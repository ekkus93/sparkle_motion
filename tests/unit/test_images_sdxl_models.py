import pytest

from fastapi.testclient import TestClient

from importlib import import_module


mod = import_module("sparkle_motion.function_tools.images_sdxl.entrypoint")
RequestModel = getattr(mod, "RequestModel")
ResponseModel = getattr(mod, "ResponseModel")
make_app = getattr(mod, "make_app")


def test_request_model_requires_prompt():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestModel()


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
