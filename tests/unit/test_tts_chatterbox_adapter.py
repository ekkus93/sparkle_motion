from __future__ import annotations

from pathlib import Path

import pytest

from sparkle_motion.function_tools.tts_chatterbox import adapter
from sparkle_motion import tts_stage


def _make_request(tmp_path: Path, *, text: str = "Hello fixture") -> tts_stage.AdapterRequest:
    provider = tts_stage.ProviderConfig(
        provider_id="fixture-local",
        display_name="Fixture Local",
        tier="fixture",
        adapter="tts_chatterbox",
        fixture_alias="fixture",
        default_voice="fixture-voice",
        estimated_latency_s=0.1,
        estimated_cost_usd_per_1k_chars=0.0,
        quality_score=0.5,
        features=("fixture",),
        languages=("en",),
        rate_limits=tts_stage.RateLimitPolicy(per_minute=60, burst=10),
        retry_policy=tts_stage.RetryPolicy(max_retries=0, backoff_s=0.05),
        watermarking=False,
    )
    voice = tts_stage.VoiceMetadata(
        voice_id="narrator",
        provider_id=provider.provider_id,
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
    return tts_stage.AdapterRequest(
        text=text,
        voice=voice,
        provider=provider,
        plan_id="plan",
        step_id="step",
        run_id="run",
        output_dir=tmp_path,
        options={"seed": 42},
    )


def test_fixture_synthesis_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMOKE_TTS", raising=False)
    req_kwargs = dict(
        text="Deterministic audio",
        voice_id="emma",
        sample_rate=24000,
        bit_depth=16,
        language="en",
        seed=123,
        watermarking=False,
        output_dir=tmp_path,
        metadata={"test": True},
        force_fixture=True,
    )
    result_a = adapter.synthesize_text(**req_kwargs)
    result_b = adapter.synthesize_text(**req_kwargs)

    assert result_a.duration_s == pytest.approx(result_b.duration_s)
    assert result_a.sample_rate == result_b.sample_rate == 24000
    assert result_a.path.read_bytes() == result_b.path.read_bytes()
    assert result_a.metadata["mode"] == "fixture"


def test_stage_adapter_returns_adapter_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMOKE_TTS", raising=False)
    stage_adapter = adapter.create_tts_stage_adapter(tts_stage.AdapterResult)
    request = _make_request(tmp_path)

    result = stage_adapter(request)

    assert result.path.exists()
    assert result.sample_rate == request.voice.sample_rate
    assert result.metadata["provider_voice_id"] == request.voice.provider_voice_id
    assert result.metadata["mode"] == "fixture"