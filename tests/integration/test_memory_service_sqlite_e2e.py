import os


import pytest

from starlette.testclient import TestClient

from sparkle_motion.tool_entrypoint import create_app
from sparkle_motion import adk_helpers


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, tmp_path):
    # ensure fixture mode is off and point to a sqlite file for persistence
    monkeypatch.delenv("ADK_USE_FIXTURE", raising=False)
    db_path = tmp_path / "memory_shared.db"
    monkeypatch.setenv("ADK_MEMORY_SQLITE", str(db_path))
    monkeypatch.setenv("DETERMINISTIC", "1")
    yield


def make_tool_a_invoke():
    def invoke(body: dict):
        mem = adk_helpers.get_memory_service()
        session_id = body.get("session_id") or "session-sqlite"
        seed = body.get("seed")
        mem.store_session_metadata(session_id, {"seed": seed, "from": "tool-a"})
        return {"stored": True, "session_id": session_id}

    return invoke


def make_tool_b_invoke():
    def invoke(body: dict):
        mem = adk_helpers.get_memory_service()
        session_id = body.get("session_id") or "session-sqlite"
        meta = mem.get_session_metadata(session_id)
        decision = body.get("decision")
        if decision:
            mem.append_reviewer_decision(session_id, decision)
        return {"meta": meta, "appended": bool(decision)}

    return invoke


def test_sqlite_e2e_cross_instance_propagation():
    # create two separate app instances that will each create their own
    # SQLiteMemoryService (backed by the same DB file)
    app_a = create_app("tool-a", make_tool_a_invoke())
    app_b = create_app("tool-b", make_tool_b_invoke())

    client_a = TestClient(app_a)
    client_b = TestClient(app_b)

    session_id = "e2e-sqlite-session-1"

    resp_a = client_a.post("/invoke", json={"session_id": session_id, "seed": {"seed_value": 7}})
    assert resp_a.status_code == 200
    assert resp_a.json()["result"]["stored"] is True

    # tool-b should be able to read what tool-a stored from the shared sqlite file
    decision = {"approved": True, "notes": "ok-sqlite"}
    resp_b = client_b.post("/invoke", json={"session_id": session_id, "decision": decision})
    assert resp_b.status_code == 200
    resb = resp_b.json()["result"]
    assert resb["meta"] is not None and resb["meta"].get("seed") == {"seed_value": 7}
    assert resb["appended"] is True

    # verify via a fresh SQLiteMemoryService instance
    db_path = os.environ.get("ADK_MEMORY_SQLITE")
    mem = adk_helpers._SQLiteMemoryService(db_path)
    decs = mem.get_reviewer_decisions(session_id)
    assert any(d.get("notes") == "ok-sqlite" for d in decs)
