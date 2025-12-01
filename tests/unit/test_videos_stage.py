from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from sparkle_motion import adk_helpers, videos_stage
from sparkle_motion.gpu_utils import ModelOOMError
from sparkle_motion.utils.dedupe import RecentIndex


class _StubRenderer:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.failures: Dict[int, BaseException] = {}

    def queue_failure(self, index: int, exc: BaseException) -> None:
        self.failures[index] = exc

    def __call__(self, chunk: videos_stage.ChunkSpec, context: videos_stage.VideoAdapterContext) -> videos_stage.ChunkRenderResult:
        call_index = len(self.calls)
        failure = self.failures.get(call_index)
        self.calls.append({"chunk": chunk, "context": context})
        if failure is not None:
            raise failure
        if context.progress_callback:
            context.progress_callback({"progress": 0.5, "phase": "rendering"})
        frames = list(range(context.opts["render_start"], context.opts["render_end"] + 1))
        return videos_stage.ChunkRenderResult(chunk=chunk, frames=frames, metadata={"call_index": call_index})


def _publish_backend():
    def _publisher(**kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - helper
        return {
            "uri": f"file://{kwargs['local_path']}",
            "storage": "local",
            "metadata": kwargs["metadata"],
            "run_id": kwargs.get("run_id"),
        }

    return adk_helpers.HelperBackend(publish=_publisher)


@pytest.fixture(autouse=True)
def _isolate_publish_backend():
    with adk_helpers.set_backend(_publish_backend()):
        yield


def _oom() -> ModelOOMError:
    return ModelOOMError(model_key="wan", stage="render", message="OOM", device_map=None, memory_snapshot={})


def test_chunk_split_and_reassembly(tmp_path: Path) -> None:
    renderer = _StubRenderer()
    artifact = videos_stage.render_video(
        start_frames=[],
        end_frames=[],
        prompt="Calm city skyline",
        opts={
            "num_frames": 150,
            "chunk_length_frames": 64,
            "chunk_overlap_frames": 4,
            "output_path": tmp_path / "clip.json",
            "debug_frames": True,
        },
        adapter=renderer,
    )

    metadata = artifact["metadata"]
    assert metadata["num_frames"] == 150
    assert metadata["assembled_frames"] == list(range(150))
    chunks = metadata["chunks"]
    assert len(chunks) == 3
    assert chunks[0]["chunk_start_frame"] == 0
    assert chunks[0]["chunk_end_frame"] == 63
    assert chunks[1]["chunk_start_frame"] == 64
    assert chunks[-1]["chunk_end_frame"] == 149


def test_adaptive_oom_retry_shrinks_chunk(tmp_path: Path) -> None:
    renderer = _StubRenderer()
    renderer.queue_failure(0, _oom())

    artifact = videos_stage.render_video(
        start_frames=[],
        end_frames=[],
        prompt="Flight over canyon",
        opts={
            "num_frames": 80,
            "chunk_length_frames": 40,
            "chunk_overlap_frames": 0,
            "output_path": tmp_path / "oom.json",
        },
        adapter=renderer,
    )

    assert artifact["metadata"]["attempts"][0]["attempts"] >= 2
    second_call_opts = renderer.calls[1]["context"].opts
    assert second_call_opts["chunk_length_frames"] < renderer.calls[0]["context"].opts["chunk_length_frames"]


def test_cpu_fallback_on_oom(tmp_path: Path) -> None:
    renderer = _StubRenderer()
    renderer.queue_failure(0, _oom())
    renderer.queue_failure(1, _oom())

    artifact = videos_stage.render_video(
        start_frames=[],
        end_frames=[],
        prompt="Stormy sea",
        opts={
            "num_frames": 32,
            "chunk_length_frames": 32,
            "chunk_overlap_frames": 0,
            "output_path": tmp_path / "cpu.json",
            "max_retries_per_chunk": 1,
        },
        adapter=renderer,
    )

    meta = artifact["metadata"]["chunks"][0]
    assert meta["cpu_fallback"] is True
    assert renderer.calls[-1]["context"].opts.get("device") == "cpu"


def test_progress_events_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    renderer = _StubRenderer()
    events: List[videos_stage.CallbackEvent] = []
    memory_events: List[Dict[str, Any]] = []

    def fake_write_memory_event(*, run_id: Optional[str], event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "videos_stage.progress":
            memory_events.append(payload)

    monkeypatch.setattr(adk_helpers, "write_memory_event", fake_write_memory_event)

    videos_stage.render_video(
        start_frames=[],
        end_frames=[],
        prompt="Aurora fade",
        opts={
            "num_frames": 16,
            "output_path": tmp_path / "progress.json",
            "plan_id": "plan-123",
            "step_id": "shot-1",
        },
        adapter=renderer,
        on_progress=events.append,
    )

    assert any(evt.get("phase") == "rendering" for evt in events)
    assert any(evt.get("phase") == "rendering" for evt in memory_events)


def test_dedupe_skips_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    publish_calls: List[Path] = []

    def fake_publish_artifact(*, local_path: Path, artifact_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        publish_calls.append(Path(local_path))
        return {
            "uri": f"file://{local_path}",
            "storage": "local",
            "artifact_type": artifact_type,
            "metadata": metadata,
            "run_id": "run-1",
        }

    monkeypatch.setattr(adk_helpers, "publish_artifact", fake_publish_artifact)

    opts = {"num_frames": 16, "output_path": tmp_path / "clip.json", "dedupe": True}
    recent = RecentIndex()

    renderer = _StubRenderer()
    first = videos_stage.render_video([], [], "City", opts={**opts, "recent_index": recent}, adapter=renderer)

    renderer = _StubRenderer()
    second = videos_stage.render_video([], [], "City", opts={**opts, "recent_index": recent}, adapter=renderer)

    assert len(publish_calls) == 1
    assert second["metadata"]["deduped"] is True
    assert second["metadata"]["duplicate_of"] == first["uri"]
    assert second["uri"] == first["uri"]
