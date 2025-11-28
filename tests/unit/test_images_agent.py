from __future__ import annotations
import pytest

from sparkle_motion import images_agent


def test_batch_split_and_ordering():
    prompt = "make a test image"
    opts = {"count": 20, "max_images_per_call": 8, "seed": 42}
    artifacts = images_agent.render(prompt, opts)
    assert len(artifacts) == 20

    # global_index metadata should be ascending
    indices = [a["metadata"]["global_index"] for a in artifacts]
    assert indices == list(range(20))


def test_deduplicate_within_plan():
    # prompt contains 'duplicate' to trigger adapter duplicate behavior
    prompt = "duplicate test"
    opts = {"count": 4, "max_images_per_call": 4, "seed": 1, "dedupe": True}
    artifacts = images_agent.render(prompt, opts)
    assert len(artifacts) == 4

    uris = [a.get("uri") for a in artifacts]
    # duplicates created by adapter are canonicalized to same inmem:// hash
    assert uris[0] == uris[2]


def test_qa_called(monkeypatch):
    called = {"count": 0}

    def fake_inspect(frames, prompts):
        called["count"] += 1
        return {"ok": True}

    monkeypatch.setattr("function_tools.qa_qwen2vl.entrypoint.inspect_frames", fake_inspect)
    prompt = "qa test"
    opts = {"count": 2, "qa": True}
    artifacts = images_agent.render(prompt, opts)
    assert called["count"] == 1
    assert len(artifacts) == 2
