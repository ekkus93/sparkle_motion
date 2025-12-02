from __future__ import annotations

import asyncio
import threading
import time

import pytest

import sparkle_motion.run_registry as run_registry_module
from sparkle_motion.run_registry import ArtifactEntry, AsyncControlGate, RunRegistry


def test_async_control_gate_supports_sync_and_async_waiters() -> None:
    gate = AsyncControlGate()
    gate.clear()
    results: list[str] = []

    def _sync_wait() -> None:
        gate.wait_sync()
        results.append("sync")

    sync_thread = threading.Thread(target=_sync_wait, daemon=True)
    sync_thread.start()

    async def _async_wait() -> None:
        await gate.wait_async()
        results.append("async")

    async def _exercise() -> None:
        task = asyncio.create_task(_async_wait())
        await asyncio.sleep(0)
        assert results == []
        gate.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(_exercise())
    sync_thread.join(timeout=1)
    assert sorted(results) == ["async", "sync"]


def test_pre_step_hook_pauses_until_resume() -> None:
    registry = RunRegistry()
    run_state = registry.start_run(
        run_id="run-control-test",
        plan_id="plan-control-test",
        plan_title="Control Test",
        mode="run",
    )
    run_id = run_state.run_id
    registry.request_pause(run_id)

    gate_hook = registry.pre_step_hook(run_id)
    completed: list[str] = []

    def _run_step() -> None:
        gate_hook("step-control")
        completed.append("done")

    worker = threading.Thread(target=_run_step, daemon=True)
    worker.start()
    time.sleep(0.05)
    assert completed == []

    registry.request_resume(run_id)
    worker.join(timeout=1)
    assert completed == ["done"]

    status = registry.get_status(run_id)
    assert status["status"] == "running"
    assert status["current_stage"] == "step-control"


def test_get_artifacts_merges_filesystem_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = RunRegistry()
    run_state = registry.start_run(
        run_id="run-fs-merge",
        plan_id="plan-fs-merge",
        plan_title="FS Merge",
        mode="run",
    )
    fs_entry = ArtifactEntry(
        stage="plan_intake",
        artifact_type="movie_plan",
        name="movie_plan.json",
        artifact_uri="artifact+fs://run-fs-merge/plan_intake/movie_plan/1",
        media_type="application/json",
        local_path="/tmp/movie_plan.json",
        download_url=None,
        storage_hint="filesystem",
        mime_type="application/json",
        metadata={},
    )
    monkeypatch.setattr(run_registry_module, "filesystem_backend_enabled", lambda env=None: True)
    monkeypatch.setattr(registry, "_load_filesystem_artifacts", lambda run_id, stage_filter: [fs_entry])
    with registry._lock:  # type: ignore[attr-defined]
        run_state.artifacts.clear()

    artifacts = registry.get_artifacts(run_state.run_id)
    assert artifacts and artifacts[0]["artifact_uri"] == fs_entry.artifact_uri

    status = registry.get_status(run_state.run_id)
    assert status["artifact_counts"].get("plan_intake") == 1
