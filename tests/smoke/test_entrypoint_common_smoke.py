from fastapi.testclient import TestClient
from sparkle_motion.function_tools.entrypoint_common import entrypoint as entrypoint_mod


def test_entrypoint_common_app_health_ready_invoke():
    app = entrypoint_mod.app
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json().get("ready") is True

    r = client.post("/invoke", json={})
    assert r.status_code == 503
    body = r.json()
    assert body.get("status") == "not-implemented"
    assert "artifact_uri" in body
