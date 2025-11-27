from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sparkle_motion.function_tools.script_agent.entrypoint import app


@pytest.mark.smoke
def test_script_agent_invoke_creates_artifact(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    os.environ["ADK_USE_FIXTURE"] = "1"
    os.environ["ARTIFACTS_DIR"] = str(artifacts)
    # ensure artifact dir is used
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("status") == "ok"

    payload = {"prompt": "Write a short movie plan for a cat.", "title": "Cat Movie"}
    r = client.post("/invoke", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") == "success"
    uri = data.get("artifact_uri")
    assert uri and uri.startswith("file://")

    # check the file exists
    path = Path(uri[len("file://"):])
    assert path.exists()
    # clean up
    try:
        path.unlink()
    except Exception:
        pass
