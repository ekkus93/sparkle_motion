from __future__ import annotations

from pathlib import Path

import pytest

from sparkle_motion import tts_agent


def _make_request(tmp_path: Path, text: str = "Sample line") -> tts_agent.AdapterRequest:
    provider = tts_agent.ProviderConfig(
        provider_id="fixture-local",
        display_name="Fixture Local",
        tier="fixture",
        adapter="tts_fixture",
        fixture_alias="fixture",
        default_voice="fixture-voice",
        estimated_latency_s=0.1,
        estimated_cost_usd_per_1k_chars=0.0,
        quality_score=0.5,
        features=("fixture",),
        languages=("en",),
        rate_limits=tts_agent.RateLimitPolicy(per_minute=120, burst=20),
        retry_policy=tts_agent.RetryPolicy(max_retries=0, backoff_s=0.05),
        watermarking=False,
    )
    voice = tts_agent.VoiceMetadata(
        voice_id="narrator",
        provider_id="fixture-local",
        provider_voice_id="fixture-voice",
        description="Narrator",
        display_name="Narrator",
        language_codes=("en",),
        features=("fixture",),
        sample_rate=24000,
        bit_depth=16,
        watermarking=False,
        estimated_cost_usd_per_1k_chars=0.0,
        estimated_latency_s=0.1,
        quality_score=0.5,
    )
    return tts_agent.AdapterRequest(
        text=text,
        voice=voice,
        provider=provider,
        plan_id="plan-123",
        step_id="step-tts",
        run_id="run-456",
        output_dir=tmp_path,
        options={},
    )


def test_fixture_adapter_deterministic_duration(tmp_path: Path) -> None:
    request = _make_request(tmp_path, text="Consistent audio output")
    res1 = tts_agent._fixture_adapter(request)
    res2 = tts_agent._fixture_adapter(request)

    assert res1.path.exists()
    assert res2.path.exists()
    assert res1.sample_rate == request.voice.sample_rate
    assert res1.bit_depth == request.voice.bit_depth

    text_len = len(request.text.strip())
    duration = max(text_len / 18.0, 0.3)
    frames = max(int(request.voice.sample_rate * duration), request.voice.sample_rate)
    frames = min(frames, request.voice.sample_rate * 10)
    expected_duration = frames / request.voice.sample_rate

    assert res1.duration_s == pytest.approx(expected_duration, rel=1e-5)
    assert res2.duration_s == pytest.approx(expected_duration, rel=1e-5)


def test_artifact_metadata_builder_populates_context(tmp_path: Path) -> None:
    request = _make_request(tmp_path)
    selection = tts_agent.SelectionDecision(
        provider_id=request.provider.provider_id,
        score=0.87,
        breakdown={"quality": 0.8, "latency": 0.15, "cost": 0.05},
        reason="weighted",
    )
    result = tts_agent.AdapterResult(
        path=tmp_path / "result.wav",
        duration_s=1.25,
        sample_rate=request.voice.sample_rate,
        bit_depth=request.voice.bit_depth,
        watermarking=request.provider.watermarking,
        metadata={"engine": "fixture"},
    )

    metadata = tts_agent._build_artifact_metadata(
        result,
        request,
        selection,
        attempt=2,
        total_text=request.text,
    )

    assert metadata["plan_id"] == request.plan_id
    assert metadata["provider_id"] == request.provider.provider_id
    assert metadata["adapter_metadata"]["engine"] == "fixture"
    assert metadata["score_breakdown"]["quality"] == pytest.approx(0.8)
    assert metadata["model_id"] == request.provider.display_name
    assert metadata["attempt"] == 2
