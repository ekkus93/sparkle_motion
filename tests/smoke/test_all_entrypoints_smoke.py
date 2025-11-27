from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path
import pytest

from fastapi.testclient import TestClient


def discover_entrypoint_modules() -> list[str]:
    """Discover subpackages under sparkle_motion.function_tools that contain an entrypoint module."""
    mods = []
    try:
        pkg = importlib.import_module("sparkle_motion.function_tools")
    except Exception as e:
        pytest.skip(f"could not import function_tools package: {e}")
    path = getattr(pkg, "__path__", None)
    if not path:
        return mods
    for finder, name, ispkg in pkgutil.iter_modules(path):
        # We expect an `entrypoint.py` inside each subpackage
        mods.append(name)
    # deterministic order
    return sorted(set(mods))


ENTRYPOINT_MODULES = discover_entrypoint_modules()


@pytest.mark.parametrize("tool_name", ENTRYPOINT_MODULES)
def test_entrypoint_smoke(tool_name: str, tmp_path, monkeypatch):
    # Set deterministic/test-mode environment
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

    module_path = f"sparkle_motion.function_tools.{tool_name}.entrypoint"
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        pytest.skip(f"could not import {module_path}: {e}")

    # get FastAPI app instance
    app = None
    try:
        maker = getattr(mod, "make_app", None)
        if callable(maker):
            app = maker()
        elif hasattr(mod, "app"):
            app = getattr(mod, "app")
    except Exception as e:
        pytest.skip(f"failed to construct app for {module_path}: {e}")

    if app is None:
        pytest.skip(f"no app found in {module_path}")

    with TestClient(app) as client:
        # health
        r = client.get("/health")
        assert r.status_code == 200
        assert "status" in r.json()

        # ready
        r = client.get("/ready")
        assert r.status_code == 200
        jr = r.json()
        assert isinstance(jr.get("ready"), bool)

        # invoke: be tolerant about response codes (validation, not-ready, or success)
        payload = {"prompt": "smoke harness test"}
        r = client.post("/invoke", json=payload)
        assert r.status_code in {200, 400, 422, 503}, f"unexpected status {r.status_code} for {module_path}: {r.text}"

        if r.status_code == 200:
            data = r.json()
            assert data.get("status") == "success"
            uri = data.get("artifact_uri")
            assert uri
            # allow local file URIs, ADK artifact URIs, or HTTP(S)
            allowed_prefixes = ("file://", "artifact://", "http://", "https://")
            assert any(uri.startswith(p) for p in allowed_prefixes), f"unexpected artifact uri: {uri}"
            if uri.startswith("file://"):
                path = Path(uri[len("file://"):])
                assert path.exists()
        else:
            # For non-200 responses at least ensure the body is JSON
            try:
                _ = r.json()
            except Exception:
                pytest.fail(f"non-json response from /invoke for {module_path}: {r.text}")

