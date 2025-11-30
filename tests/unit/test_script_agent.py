from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sparkle_motion import script_agent


class _StubAgent:
    def __init__(self, payload):
        self._payload = payload

    def generate(self, prompt: str):
        return self._payload


def _sample_plan_payload() -> dict:
    return {
        "title": "Demo",
        "base_images": [
            {"id": "frame_000", "prompt": "hero start"},
            {"id": "frame_001", "prompt": "hero end"},
        ],
        "shots": [
            {
                "id": "shot-1",
                "duration_sec": 5,
                "visual_description": "A calm beach",
                "start_base_image_id": "frame_000",
                "end_base_image_id": "frame_001",
            }
        ],
        "dialogue_timeline": [
            {"type": "silence", "start_time_sec": 0.0, "duration_sec": 5.0},
        ],
        "render_profile": {"video": {"model_id": "wan-2.1"}, "metadata": {}},
    }


@pytest.fixture(autouse=True)
def _stub_run_id(monkeypatch):
    monkeypatch.setattr(script_agent.observability, "get_session_id", lambda: "test-run")


@pytest.fixture(autouse=True)
def _stub_memory_events(monkeypatch):
    events: list[dict] = []

    def fake_write_memory_event(*, run_id, event_type, payload, ts=None):  # noqa: D401 - test helper
        events.append({"run_id": run_id, "event_type": event_type, "payload": payload})

    monkeypatch.setattr(script_agent.adk_helpers, "write_memory_event", fake_write_memory_event)
    return events


def _install_agent(monkeypatch, payload):
    stub = _StubAgent(payload)
    monkeypatch.setattr(script_agent.adk_factory, "get_agent", lambda *_, **__: stub)
    return stub


def test_generate_plan_validates_and_publishes(monkeypatch, tmp_path):
    plan_payload = _sample_plan_payload()
    _install_agent(monkeypatch, json.dumps(plan_payload))

    captured: dict = {}

    def fake_publish_artifact(**kwargs):
        captured.update(kwargs)
        payload = json.loads(Path(kwargs["local_path"]).read_text(encoding="utf-8"))
        captured["persisted"] = payload
        return {
            "uri": "file:///tmp/plan.json",
            "storage": "local",
            "artifact_type": kwargs["artifact_type"],
            "media_type": kwargs.get("media_type"),
            "metadata": kwargs.get("metadata") or {},
            "run_id": kwargs.get("run_id"),
        }

    monkeypatch.setattr(script_agent.adk_helpers, "publish_artifact", fake_publish_artifact)

    plan = script_agent.generate_plan("Create a tranquil short.", model_spec="stub-model", run_id="run-123")

    assert plan.title == plan_payload["title"]
    assert captured["artifact_type"] == "script_agent_movie_plan"
    assert captured["metadata"]["model_spec"] == "stub-model"
    assert captured["persisted"]["validated_plan"]["title"] == plan.title


def test_generate_plan_raises_on_invalid_json(monkeypatch):
    _install_agent(monkeypatch, "not-json")

    with pytest.raises(script_agent.PlanParseError):
        script_agent.generate_plan("bad response", model_spec="stub-model")


def test_generate_plan_raises_on_schema_failure(monkeypatch):
    bad_payload = _sample_plan_payload()
    bad_payload["base_images"] = [{"id": "frame_000", "prompt": "start"}]  # insufficient entries
    _install_agent(monkeypatch, json.dumps(bad_payload))

    with pytest.raises(script_agent.PlanSchemaError):
        script_agent.generate_plan("needs fields", model_spec="stub-model")


def test_generate_plan_respects_shot_limit(monkeypatch):
    payload = _sample_plan_payload()
    payload["shots"].append(
        {
            "id": "shot-2",
            "duration_sec": 5,
            "visual_description": "Scene",
            "start_base_image_id": "frame_001",
            "end_base_image_id": "frame_002",
        }
    )
    payload["base_images"].append({"id": "frame_002", "prompt": "hero close"})
    payload["dialogue_timeline"] = [
        {"type": "silence", "start_time_sec": 0.0, "duration_sec": 10.0},
    ]
    _install_agent(monkeypatch, json.dumps(payload))
    monkeypatch.setenv("SCRIPT_AGENT_MAX_SHOTS", "1")
    monkeypatch.setattr(
        script_agent.adk_helpers,
        "publish_artifact",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not publish")),
    )

    with pytest.raises(script_agent.PlanResourceError):
        script_agent.generate_plan("too many shots", model_spec="stub-model")


def test_generate_plan_requires_non_empty_render_model(monkeypatch):
    payload = _sample_plan_payload()
    payload["render_profile"]["video"]["model_id"] = "   "
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError):
        script_agent.generate_plan("blank model", model_spec="stub-model")


def test_generate_plan_rejects_duplicate_base_images(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload["base_images"][1]["id"] = payload["base_images"][0]["id"]
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError):
        script_agent.generate_plan("duplicate base ids", model_spec="stub-model")


def test_generate_plan_rejects_timeline_not_starting_at_zero(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload["dialogue_timeline"] = [
        {"type": "silence", "start_time_sec": 0.5, "duration_sec": 4.5},
    ]
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError):
        script_agent.generate_plan("timeline must start at zero", model_spec="stub-model")


def test_generate_plan_rejects_timeline_with_gaps(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload["dialogue_timeline"] = [
        {"type": "silence", "start_time_sec": 0.0, "duration_sec": 1.0},
        {
            "type": "dialogue",
            "character_id": "c1",
            "text": "Hello",
            "start_time_sec": 2.0,
            "duration_sec": 3.0,
        },
    ]
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError):
        script_agent.generate_plan("timeline gaps", model_spec="stub-model")


def test_generate_plan_rejects_short_dialogue_timeline(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload["dialogue_timeline"] = [
        {"type": "silence", "start_time_sec": 0.0, "duration_sec": payload["shots"][0]["duration_sec"] - 0.25},
    ]
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError) as excinfo:
        script_agent.generate_plan("timeline too short", model_spec="stub-model")

    assert "ends at" in str(excinfo.value)


def test_generate_plan_rejects_overlong_dialogue_timeline(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload["dialogue_timeline"] = [
        {"type": "silence", "start_time_sec": 0.0, "duration_sec": payload["shots"][0]["duration_sec"] + 0.5},
    ]
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError) as excinfo:
        script_agent.generate_plan("timeline too long", model_spec="stub-model")

    assert "exceeds total shot duration" in str(excinfo.value)


def test_generate_plan_requires_dialogue_timeline(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload.pop("dialogue_timeline", None)
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError):
        script_agent.generate_plan("missing timeline", model_spec="stub-model")


def test_generate_plan_requires_render_profile(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload.pop("render_profile", None)
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError):
        script_agent.generate_plan("missing render profile", model_spec="stub-model")


def test_generate_plan_rejects_extra_base_images(monkeypatch):
    payload = deepcopy(_sample_plan_payload())
    payload["base_images"].append({"id": "frame_002", "prompt": "extra continuity"})
    _install_agent(monkeypatch, json.dumps(payload))

    with pytest.raises(script_agent.PlanSchemaError) as excinfo:
        script_agent.generate_plan("too many base images", model_spec="stub-model")

    assert "len(shots)+1" in str(excinfo.value)
