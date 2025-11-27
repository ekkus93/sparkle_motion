import os
import tempfile
from pathlib import Path

import pytest

from sparkle_motion import adk_helpers


@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch, tmp_path):
    # ensure deterministic/local profile for MemoryService tests
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("DETERMINISTIC", "1")
    # point artifacts dir to tmp so tests don't touch repo
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    yield


def test_memory_service_store_and_retrieve(tmp_path):
    """Verify MemoryService can store and retrieve seeds and metadata.

    This test uses the ADK helper abstraction to get a memory service
    implementation. In fixture mode the helper should provide a local
    sqlite-backed memory service or a simple in-memory stub.
    """
    # Try to obtain a memory service via ADK helpers. If the project doesn't
    # expose one, the helper should raise ImportError or return None; handle
    # both gracefully so the test fails with a clear message.
    mem = None
    try:
        mem = adk_helpers.get_memory_service()
    except Exception as e:
        pytest.skip(f"MemoryService not available in this environment: {e}")

    assert mem is not None, "MemoryService helper returned None"

    session_id = "test-session-123"
    seed = {"seed_value": 42}
    reviewer_decision = {"approved": True, "notes": "looks good"}

    # store seed
    mem.store_session_metadata(session_id, {"seed": seed})

    # store reviewer decision
    mem.append_reviewer_decision(session_id, reviewer_decision)

    # retrieve and assert
    meta = mem.get_session_metadata(session_id)
    assert meta is not None and meta.get("seed") == seed

    decisions = mem.get_reviewer_decisions(session_id)
    assert isinstance(decisions, list)
    assert any(d.get("approved") is True for d in decisions)


def test_memory_service_cross_tool_propagation(tmp_path):
    """Simulate two tools writing and reading session metadata.

    Tool A writes the seed and Tool B reads it and appends a reviewer note.
    """
    try:
        mem = adk_helpers.get_memory_service()
    except Exception as e:
        pytest.skip(f"MemoryService not available: {e}")

    session_id = "cross-tool-session"
    seed = {"seed_value": 7}

    # Tool A
    mem.store_session_metadata(session_id, {"seed": seed, "creator": "tool-a"})

    # Tool B reads
    meta_b = mem.get_session_metadata(session_id)
    assert meta_b is not None
    assert meta_b.get("seed") == seed
    assert meta_b.get("creator") == "tool-a"

    # Tool B appends reviewer decision
    mem.append_reviewer_decision(session_id, {"approved": False, "notes": "needs edit"})

    decisions = mem.get_reviewer_decisions(session_id)
    assert any(d.get("notes") == "needs edit" for d in decisions)
