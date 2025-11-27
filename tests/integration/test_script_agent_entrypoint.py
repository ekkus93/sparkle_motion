import os
from pathlib import Path

import pytest

from starlette.testclient import TestClient

from sparkle_motion.function_tools.ScriptAgent.entrypoint import make_app


@pytest.fixture(autouse=True)
def fixture_env(monkeypatch, tmp_path):
    # ensure deterministic fixture mode and isolate artifacts
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    yield


def test_script_agent_health_ready_and_invoke(tmp_path):
    app = make_app()
    with TestClient(app) as client:
        # health
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json().get("status") in ("ok", "shutting_down")

        # ready should be true in fixture mode (app marks ready in lifespan)
        r2 = client.get("/ready")
        assert r2.status_code == 200
        assert r2.json().get("ready") is True

        # missing prompt -> validation error (400 or 422 depending on handlers)
        r3 = client.post("/invoke", json={})
        assert r3.status_code in (400, 422)

        # successful invoke
        prompt = "generate a short script"
        r4 = client.post("/invoke", json={"prompt": prompt})
        assert r4.status_code == 200
        body = r4.json()
        # Some entrypoints return a top-level model (like ScriptAgent) while
        # others are wrapped as {"tool": id, "result": {...}} by helpers.
        result = body.get("result") if isinstance(body, dict) and "result" in body else body
        assert isinstance(result, dict)
        assert result.get("status") == "success"
        uri = result.get("artifact_uri")
        assert isinstance(uri, str) and uri.startswith("file://")

        # artifact file should exist on disk
        path = uri[len("file://") :]
        assert Path(path).exists()
