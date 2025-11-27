import os
import tempfile
from pathlib import Path

import pytest

from sparkle_motion.adk_helpers import get_memory_service


@pytest.fixture(autouse=True)
def fixture_env(monkeypatch, tmp_path):
    # ensure deterministic fixture mode and isolated sqlite when used
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    yield


def test_inmemory_seed_propagation_singleton():
    # In fixture mode the in-memory MemoryService should be a process-shared singleton
    svc1 = get_memory_service()
    svc2 = get_memory_service()
    assert svc1 is svc2

    session = "session-seed-test"
    # store seed/metadata
    svc1.store_session_metadata(session, {"seed": 42})

    # read back via other handle
    m = svc2.get_session_metadata(session)
    assert isinstance(m, dict)
    assert m.get("seed") == 42


def test_sqlite_seed_persistence(tmp_path, monkeypatch):
    # Point MemoryService to a sqlite file and validate persistence across instances
    db = tmp_path / "mem.db"
    monkeypatch.setenv("ADK_MEMORY_SQLITE", str(db))

    svc = get_memory_service()
    session = "session-sqlite-test"
    svc.store_session_metadata(session, {"seed": 123, "note": "persist"})

    # force a new service instance by clearing any cached state if applicable
    # (get_memory_service should detect ADK_MEMORY_SQLITE and return a fresh SQLite-backed instance)
    svc2 = get_memory_service()
    m = svc2.get_session_metadata(session)
    assert isinstance(m, dict)
    assert m.get("seed") == 123
    assert m.get("note") == "persist"


def test_memoryservice_records_reviewer_decisions(tmp_path, monkeypatch):
    # validate MemoryService can append and retrieve reviewer decisions
    db = tmp_path / "mem2.db"
    monkeypatch.setenv("ADK_MEMORY_SQLITE", str(db))

    svc = get_memory_service()
    session = "session-reviewer-test"
    svc.append_reviewer_decision(session, {"reviewer": "alice", "decision": "approve"})
    svc.append_reviewer_decision(session, {"reviewer": "bob", "decision": "regenerate"})

    decisions = svc.get_reviewer_decisions(session)
    assert isinstance(decisions, list)
    assert any(d.get("reviewer") == "alice" and d.get("decision") == "approve" for d in decisions)
    assert any(d.get("reviewer") == "bob" and d.get("decision") == "regenerate" for d in decisions)
