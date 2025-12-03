from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from sparkle_motion import images_stage
from sparkle_motion.ratelimit import RateLimitDecision, RateLimiter
from sparkle_motion.utils.dedupe import RecentIndex, compute_hash
from sparkle_motion.utils.recent_index_sqlite import RecentIndexSqlite


class _ScriptedLimiter(RateLimiter):
    def __init__(self, decisions: List[RateLimitDecision]) -> None:
        self._decisions = decisions
        self.calls: List[Dict[str, Any]] = []

    def allow(self, tokens: int = 1) -> bool:
        if not self._decisions:
            return True
        return self._decisions[0].allowed

    def request(self, tokens: int = 1, *, queue_allowed: bool = False, ttl_s: float = 600.0) -> RateLimitDecision:
        self.calls.append({"tokens": tokens, "queue_allowed": queue_allowed, "ttl_s": ttl_s})
        if not self._decisions:
            return RateLimitDecision(status="allowed", tokens=tokens)
        return self._decisions.pop(0)


def _install_fake_renderer(monkeypatch: pytest.MonkeyPatch, responses: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def fake_render(prompt: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:  # pragma: no cover - helper
        call_index = len(calls)
        calls.append({"count": opts["count"], "batch_start": opts.get("batch_start", 0), "opts": dict(opts)})
        return responses[call_index]

    monkeypatch.setattr(images_stage, "render_images", fake_render)
    return calls


def test_render_batches_preserve_order(monkeypatch: pytest.MonkeyPatch) -> None:
    responses: List[List[Dict[str, Any]]] = []
    counts = [3, 3, 3, 1]
    start = 0
    for batch_size in counts:
        batch: List[Dict[str, Any]] = []
        for offset in range(batch_size):
            absolute = start + offset
            batch.append({"data": f"item-{absolute}".encode(), "metadata": {"value": absolute}})
        responses.append(batch)
        start += batch_size

    calls = _install_fake_renderer(monkeypatch, responses)

    result = images_stage.render("prompt", {"count": 10, "max_images_per_call": 3})

    assert [call["count"] for call in calls] == counts
    assert len(result) == 10
    for idx, artifact in enumerate(result):
        meta = artifact["metadata"]
        assert meta["global_index"] == idx
        assert meta["batch_index"] == idx // 3
        expected_item_index = idx % 3 if idx < 9 else 0
        assert meta["item_index"] == expected_item_index
        assert meta["batch_start"] in {0, 3, 6, 9}
        assert artifact.get("data") is not None


def test_render_dedupe_within_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    duplicate_payload = b"duplicate-0"
    duplicate_payload_variant = b"duplicate-1"
    dup_phash = "abcdabcdabcdabcd"
    responses = [[
        {"data": b"unique-0", "metadata": {"phash": "1111111111111111"}},
        {"data": duplicate_payload, "metadata": {"phash": dup_phash}},
        {"data": duplicate_payload_variant, "metadata": {"phash": dup_phash}},
        {"data": b"unique-3", "metadata": {"phash": "2222222222222222"}},
    ]]

    _install_fake_renderer(monkeypatch, responses)
    recent = RecentIndex()

    result = images_stage.render(
        "prompt",
        {"count": 4, "max_images_per_call": 8, "dedupe": True, "recent_index": recent},
    )

    assert len(result) == 4
    # First duplicate establishes canonical entry (not marked deduped)
    canonical = result[1]["uri"]
    assert result[1]["metadata"].get("deduped") is None
    assert result[2]["metadata"].get("deduped") is True
    assert result[2]["duplicate_of"] == canonical
    assert "data" not in result[2]
    assert recent.get(dup_phash) == canonical


def test_render_dedupe_threshold_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    near_match = "0000000000000000"
    slight_variant = "000000000000000f"
    responses = [
        [
            {"data": b"first", "metadata": {"phash": near_match}},
            {"data": b"second", "metadata": {"phash": slight_variant}},
        ],
        [
            {"data": b"first", "metadata": {"phash": near_match}},
            {"data": b"second", "metadata": {"phash": slight_variant}},
        ],
    ]
    _install_fake_renderer(monkeypatch, responses)

    strict = images_stage.render(
        "prompt",
        {
            "count": 2,
            "dedupe": True,
            "dedupe_phash_threshold": 0,
        },
    )
    assert strict[1]["metadata"].get("deduped") is None
    assert "duplicate_of" not in strict[1]

    tolerant = images_stage.render(
        "prompt",
        {
            "count": 2,
            "dedupe": True,
            "dedupe_phash_threshold": 16,
        },
    )
    assert tolerant[1]["metadata"].get("deduped") is True
    assert tolerant[1]["duplicate_of"] == tolerant[0]["uri"]


def test_render_invalid_dedupe_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_render(prompt: str, opts: Dict[str, Any]) -> List[Dict[str, Any]]:  # pragma: no cover - should not run
        raise AssertionError("adapter should not be called")

    monkeypatch.setattr(images_stage, "render_images", fake_render)

    with pytest.raises(ValueError):
        images_stage.render(
            "prompt",
            {
                "count": 1,
                "dedupe": True,
                "dedupe_phash_threshold": -1,
            },
        )


def test_render_without_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [[
        {"data": b"blob", "metadata": {}},
        {"data": b"blob", "metadata": {}},
    ]]
    _install_fake_renderer(monkeypatch, responses)

    result = images_stage.render("prompt", {"count": 2, "max_images_per_call": 2, "dedupe": False})

    assert len(result) == 2
    for artifact in result:
        assert artifact["data"] == b"blob"
        assert "duplicate_of" not in artifact
        assert artifact["metadata"].get("deduped") is None


def test_render_with_sqlite_recent_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    responses = [[
        {"data": b"dup", "metadata": {}},
        {"data": b"dup", "metadata": {}},
    ]]
    _install_fake_renderer(monkeypatch, responses)
    db_path = tmp_path / "recent.db"
    recent = RecentIndexSqlite(str(db_path))

    try:
        first = images_stage.render(
            "prompt",
            {"count": 2, "max_images_per_call": 2, "dedupe": True, "recent_index": recent},
        )
        assert first[1]["metadata"].get("deduped") is True
        assert first[1]["duplicate_of"] == first[0]["uri"]
        digest = compute_hash(b"dup")
        assert recent.get(digest) == first[0]["uri"]
    finally:
        recent.close()


def test_qa_arguments_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [[{"data": b"blob", "metadata": {}}]]
    calls = _install_fake_renderer(monkeypatch, responses)

    result = images_stage.render(
        "prompt",
        {
            "count": 1,
            "qa": True,
            "reference_images": [b"ignored"],
        },
    )

    assert len(result) == 1
    assert "qa_status" not in result[0]["metadata"]
    assert "qa" not in calls[0]["opts"]
    assert "reference_images" not in calls[0]["opts"]


def test_rate_limit_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [[{"data": b"blob", "metadata": {}}]]
    _install_fake_renderer(monkeypatch, responses)
    decision = RateLimitDecision(status="queued", tokens=1, retry_after_s=1.0, eta_epoch_s=42.0)
    limiter = _ScriptedLimiter([decision])

    with pytest.raises(images_stage.RateLimitQueued) as exc:
        images_stage.render("prompt", {"count": 1, "rate_limiter": limiter})

    assert exc.value.decision == decision
    assert limiter.calls[0]["queue_allowed"] is True


def test_rate_limit_reject_without_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [[{"data": b"blob", "metadata": {}}]]
    _install_fake_renderer(monkeypatch, responses)
    decision = RateLimitDecision(status="rejected", tokens=1, retry_after_s=0.5)
    limiter = _ScriptedLimiter([decision])

    with pytest.raises(images_stage.RateLimitExceeded):
        images_stage.render(
            "prompt",
            {
                "count": 1,
                "rate_limiter": limiter,
                "queue_allowed": False,
            },
        )

    assert limiter.calls[0]["queue_allowed"] is False
