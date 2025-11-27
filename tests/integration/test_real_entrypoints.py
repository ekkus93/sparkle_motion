import importlib
from pathlib import Path

import pytest

from starlette.testclient import TestClient


ENTRYPOINT_MODULES = [
    "sparkle_motion.function_tools.images_sdxl.entrypoint",
    "sparkle_motion.function_tools.tts_chatterbox.entrypoint",
]


@pytest.fixture(autouse=True)
def fixture_env(monkeypatch, tmp_path):
    # deterministic fixture mode and isolated artifacts dir
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    yield


@pytest.mark.parametrize("modpath", ENTRYPOINT_MODULES)
def test_entrypoint_basic_contract(modpath):
    mod = importlib.import_module(modpath)
    # each module exposes a make_app() factory returning a FastAPI app
    app = getattr(mod, "make_app")()

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

        r2 = client.get("/ready")
        assert r2.status_code == 200
        assert r2.json().get("ready") is True

        # missing prompt -> validation error (400 or 422)
        r3 = client.post("/invoke", json={})
        assert r3.status_code in (400, 422)

        r4 = client.post("/invoke", json={"prompt": "hello world"})
        assert r4.status_code == 200
        body = r4.json()
        result = body.get("result") if isinstance(body, dict) and "result" in body else body
        assert isinstance(result, dict)
        assert result.get("status") == "success"
        uri = result.get("artifact_uri")
        assert isinstance(uri, str) and uri.startswith("file://")
        path = uri[len("file://") :]
        assert Path(path).exists()
