from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sparkle_motion import script_agent
from sparkle_motion.schemas import MoviePlan

from . import helpers


def _configure_real_script_agent_env(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: Path,
    *,
    run_id: str,
) -> None:
    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_ADAPTERS", "SMOKE_ADK"],
        disable_keys=["ADK_USE_FIXTURE"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "ADK_USE_FIXTURE": "0",
            "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
            "SPARKLE_LOCAL_RUNS_ROOT": str(tmp_path / "runs"),
            "SPARKLE_RUN_ID": run_id,
        },
    )


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _plan_signature(plan: MoviePlan) -> dict[str, Any]:
    shots_signature = [
        (
            _normalize_text(shot.visual_description),
            round(float(shot.duration_sec), 3),
            _normalize_text(getattr(shot, "motion_prompt", "")),
            bool(shot.is_talking_closeup),
            shot.start_base_image_id,
            shot.end_base_image_id,
        )
        for shot in plan.shots
    ]
    base_signature = [
        (
            base.id,
            _normalize_text(base.prompt),
        )
        for base in plan.base_images
    ]
    timeline_signature = [
        (
            entry.type,
            _normalize_text(getattr(entry, "text", None)),
            getattr(entry, "character_id", None),
            round(float(entry.start_time_sec), 3),
            round(float(entry.duration_sec), 3),
        )
        for entry in plan.dialogue_timeline
    ]
    return {
        "title": _normalize_text(plan.title),
        "shots": shots_signature,
        "base_images": base_signature,
        "timeline": timeline_signature,
    }


@pytest.mark.gpu
def test_script_agent_generate_plan_real_llm(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    _configure_real_script_agent_env(monkeypatch, tmp_path, run_id="gpu-script-agent")

    prompt = "Generate a 2-shot plan about a robot exploring Mars."
    plan = script_agent.generate_plan(prompt, seed=42)

    assert isinstance(plan, MoviePlan)
    assert len(plan.shots) >= 2

    assert len(plan.base_images) == len(plan.shots) + 1
    assert all(base.prompt.strip() for base in plan.base_images)

    for shot in plan.shots:
        assert shot.visual_description.strip()
        assert shot.start_base_image_id
        assert shot.end_base_image_id

    timeline = plan.dialogue_timeline
    assert timeline, "dialogue timeline should cover runtime"
    assert timeline[0].start_time_sec == pytest.approx(0.0, abs=0.05)

    final_timeline_time = timeline[-1].start_time_sec + timeline[-1].duration_sec
    total_duration = sum(shot.duration_sec for shot in plan.shots)
    assert final_timeline_time == pytest.approx(total_duration, rel=0.05, abs=0.25)

    metadata_source = (plan.metadata or {}).get("source")
    assert metadata_source != "script_agent.entrypoint.synthetic"


@pytest.mark.gpu
def test_script_agent_determinism(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    _configure_real_script_agent_env(monkeypatch, tmp_path, run_id="gpu-script-agent-determinism")

    prompt = "Describe a cinematic two-shot scene featuring explorers on Titan."
    seed = 1337

    plan_a = script_agent.generate_plan(prompt, seed=seed, run_id="gpu-script-agent-determinism-a")
    plan_b = script_agent.generate_plan(prompt, seed=seed, run_id="gpu-script-agent-determinism-b")

    assert isinstance(plan_a, MoviePlan)
    assert isinstance(plan_b, MoviePlan)
    assert len(plan_a.shots) >= 2, "Expected multi-shot plan for determinism validation"

    signature_a = _plan_signature(plan_a)
    signature_b = _plan_signature(plan_b)
    assert signature_a == signature_b, "Plan signature should remain stable for identical seeds"

    for plan in (plan_a, plan_b):
        metadata_source = (plan.metadata or {}).get("source")
        assert metadata_source != "script_agent.entrypoint.synthetic"


@pytest.mark.gpu
def test_script_agent_resource_limits(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()

    _configure_real_script_agent_env(monkeypatch, tmp_path, run_id="gpu-script-agent-resource-limit")
    monkeypatch.setenv("SCRIPT_AGENT_MAX_SHOTS", "0")

    prompt = "Generate a detailed four-shot montage about rebuilding a lunar outpost."

    with pytest.raises(script_agent.PlanResourceError) as excinfo:
        script_agent.generate_plan(prompt, seed=256, run_id="gpu-script-agent-resource-limit-run")

    assert "shot" in str(excinfo.value).lower()


