
import pytest

from sparkle_motion import adk_helpers


@pytest.fixture(autouse=True)
def unset_fixture_env(monkeypatch):
    # Ensure we don't pick the in-memory fixture implementation
    monkeypatch.setenv("ADK_USE_FIXTURE", "0")
    yield


def test_sqlite_memory_service_persistence(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("ADK_MEMORY_SQLITE", str(db_path))

    mem = adk_helpers.get_memory_service()
    assert mem is not None

    session_id = "sqlite-session-1"
    seed = {"seed_value": 99}

    mem.store_session_metadata(session_id, {"seed": seed})

    # create a fresh instance to validate persistence to disk
    mem2 = adk_helpers._SQLiteMemoryService(str(db_path))
    meta = mem2.get_session_metadata(session_id)
    assert meta is not None and meta.get("seed") == seed

    # decisions persist too
    mem.append_reviewer_decision(session_id, {"approved": True})
    mem3 = adk_helpers._SQLiteMemoryService(str(db_path))
    decs = mem3.get_reviewer_decisions(session_id)
    assert any(d.get("approved") is True for d in decs)
