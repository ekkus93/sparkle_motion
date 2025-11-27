
import pytest

from starlette.testclient import TestClient

from sparkle_motion.tool_entrypoint import create_app
from sparkle_motion import adk_helpers


@pytest.fixture(autouse=True)
def fixture_env(monkeypatch, tmp_path):
    # use in-memory fixture MemoryService for deterministic tests
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    yield


def make_tool_a_invoke():
    def invoke(body: dict):
        mem = adk_helpers.get_memory_service()
        session_id = body.get("session_id") or "session-a"
        seed = body.get("seed")
        mem.store_session_metadata(session_id, {"seed": seed, "from": "tool-a"})
        return {"stored": True, "session_id": session_id}

    return invoke


def make_tool_b_invoke():
    def invoke(body: dict):
        mem = adk_helpers.get_memory_service()
        session_id = body.get("session_id") or "session-a"
        meta = mem.get_session_metadata(session_id)
        decision = body.get("decision")
        if decision:
            mem.append_reviewer_decision(session_id, decision)
        return {"meta": meta, "appended": bool(decision)}

    return invoke


def test_entrypoint_memory_propagation():
    app_a = create_app("tool-a", make_tool_a_invoke())
    app_b = create_app("tool-b", make_tool_b_invoke())

    client_a = TestClient(app_a)
    client_b = TestClient(app_b)

    session_id = "e2e-session-1"

    resp_a = client_a.post("/invoke", json={"session_id": session_id, "seed": {"seed_value": 123}})
    assert resp_a.status_code == 200, resp_a.text
    assert resp_a.json()["result"]["stored"] is True

    # now tool-b reads what tool-a stored and appends a decision
    decision = {"approved": True, "notes": "ok"}
    resp_b = client_b.post("/invoke", json={"session_id": session_id, "decision": decision})
    assert resp_b.status_code == 200, resp_b.text
    resb = resp_b.json()["result"]
    assert resb["meta"] is not None and resb["meta"].get("seed") == {"seed_value": 123}
    assert resb["appended"] is True

    # verify via a fresh service instance that decision persisted in fixture mem
    mem = adk_helpers.get_memory_service()
    decs = mem.get_reviewer_decisions(session_id)
    assert any(d.get("approved") is True for d in decs)
