import pytest
from fastapi.testclient import TestClient

from sparkle_motion.function_tools.images_sdxl.entrypoint import make_app
from sparkle_motion.function_tools.images_sdxl.models import ImagesSDXLRequest


def test_request_model_requires_prompt() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ImagesSDXLRequest()


def test_request_model_enforces_dimensions() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ImagesSDXLRequest(prompt="ok", width=65, height=64)


def test_invoke_returns_artifact_list(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SPARKLE_LOCAL_RUNS_ROOT", str(tmp_path / "runs"))

    app = make_app()
    client = TestClient(app)

    resp = client.post("/invoke", json={"prompt": "smoke", "count": 1, "width": 64, "height": 64})
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "success"
    assert j.get("artifact_uri", "").startswith("file://")
    assert isinstance(j.get("artifacts"), list)
    assert j["artifacts"][0]["metadata"]["engine"] == "fixture"
