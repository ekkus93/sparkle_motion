import importlib
from pathlib import Path

import pytest

from starlette.testclient import TestClient


MODULES = [
    "sparkle_motion.function_tools.videos_wan.entrypoint",
    "sparkle_motion.function_tools.assemble_ffmpeg.entrypoint",
    "sparkle_motion.function_tools.qa_qwen2vl.entrypoint",
    "sparkle_motion.function_tools.lipsync_wav2lip.entrypoint",
    "sparkle_motion.function_tools.script_agent.entrypoint",
]


@pytest.fixture(autouse=True)
def fixture_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    yield


@pytest.mark.parametrize("modpath", MODULES)
def test_entrypoint_contracts_are_sane(modpath):
    # skip if module import isn't available in this environment
    try:
        mod = importlib.import_module(modpath)
    except Exception as e:
        pytest.skip(f"Could not import {modpath}: {e}")

    make_app = getattr(mod, "make_app", None)
    if make_app is None:
        pytest.skip(f"{modpath} has no make_app() factory")

    app = make_app()

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

        r2 = client.get("/ready")
        assert r2.status_code == 200

        # validation error for missing body expected
        r3 = client.post("/invoke", json={})
        assert r3.status_code in (400, 422)

        # try a basic invoke payload; if the tool needs different fields it should
        # respond with a validation error instead of crashing
        r4 = client.post("/invoke", json={"prompt": "test"})
        if r4.status_code != 200:
            # accept common non-200 validation/unsupported responses
            assert r4.status_code in (400, 422, 501, 503)
            return

        body = r4.json()
        result = body.get("result") if isinstance(body, dict) and "result" in body else body
        # if result is a mapping, prefer artifact_uri check; otherwise ensure success-like shape
        if isinstance(result, dict):
            # some tools emit a status/result envelope
            uri = result.get("artifact_uri") or result.get("uri") or result.get("output_uri")
            if uri:
                assert isinstance(uri, str) and uri.startswith("file://")
                path = uri[len("file://") :]
                assert Path(path).exists()
            else:
                # if no URI, at least expect a success flag or text output
                assert result.get("status") in ("success", None) or "text" in result
        else:
            # allow string/text outputs
            assert isinstance(result, (str, list))
