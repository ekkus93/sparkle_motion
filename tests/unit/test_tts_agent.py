from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List

import pytest
import yaml

from sparkle_motion import adk_helpers, tts_agent

if TYPE_CHECKING:
    from tests.conftest import MediaAssets


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_agent, "_CONFIG_CACHE", None)
    monkeypatch.setattr(tts_agent, "_ADAPTERS", {})
    monkeypatch.setattr(tts_agent, "_fixture_only_mode", lambda: False)
    monkeypatch.setattr(tts_agent.observability, "get_session_id", lambda: "session-test")
    monkeypatch.setattr(tts_agent, "_random_uniform", lambda a, b: 0.0)


@pytest.fixture(autouse=True)
def _silence_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_agent.telemetry, "emit_event", lambda *_, **__: None)

    def _noop_memory_event(*, run_id: str | None, event_type: str, payload: Dict[str, Any]) -> None:  # pragma: no cover - helper
        return None

    monkeypatch.setattr(adk_helpers, "write_memory_event", _noop_memory_event)


@pytest.fixture()
def publish_backend() -> Iterator[List[Dict[str, Any]]]:
    published: List[Dict[str, Any]] = []

    def _publisher(**kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - helper
        published.append(kwargs)
        return {
            "uri": f"file://{kwargs['local_path']}",
            "storage": "local",
            "metadata": kwargs["metadata"],
            "run_id": kwargs.get("run_id"),
        }

    backend = adk_helpers.HelperBackend(publish=_publisher)
    with adk_helpers.set_backend(backend):
        yield published


def _write_config(tmp_path: Path, providers: Dict[str, Dict[str, Any]]) -> Path:
    preferences = []
    for pid, entry in providers.items():
        preferences.append({"provider": pid, "provider_voice_id": entry.get("default_voice", f"{pid}-voice")})

    cfg = {
        "providers": providers,
        "voices": {
            "narrator": {
                "description": "Narrator",
                "default_sample_rate": 24000,
                "default_bit_depth": 16,
                "provider_preferences": preferences,
            }
        },
        "priority_profiles": {
            "balanced": {
                "weights": {
                    "quality": 0.6,
                    "latency": 0.3,
                    "cost": 0.1,
                }
            }
        },
        "rate_caps": {},
    }
    path = tmp_path / "tts_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _provider_entry(
    *,
    adapter: str,
    tier: str = "tier1",
    fixture_alias: str,
    default_voice: str,
    quality_score: float,
    estimated_latency_s: float,
    estimated_cost_usd_per_1k_chars: float,
    features: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "display_name": adapter,
        "tier": tier,
        "adapter": adapter,
        "fixture_alias": fixture_alias,
        "default_voice": default_voice,
        "estimated_latency_s": estimated_latency_s,
        "estimated_cost_usd_per_1k_chars": estimated_cost_usd_per_1k_chars,
        "quality_score": quality_score,
        "features": features or ["fixture"],
        "languages": ["en"],
        "rate_limits": {"per_minute": 120, "burst": 20},
        "retry_policy": {"max_retries": 1, "backoff_s": 0.05},
        "watermarking": False,
    }


def test_provider_fallback_on_quota(
    tmp_path: Path, publish_backend: List[Dict[str, Any]], deterministic_media_assets: MediaAssets
) -> None:
    providers = {
        "fixture-local": _provider_entry(
            adapter="fixture_adapter",
            tier="fixture",
            fixture_alias="fixture",
            default_voice="fixture-voice",
            quality_score=0.4,
            estimated_latency_s=0.1,
            estimated_cost_usd_per_1k_chars=0.0,
        ),
        "pro-cloud": _provider_entry(
            adapter="pro_adapter",
            fixture_alias="pro",
            default_voice="pro-voice",
            quality_score=0.95,
            estimated_latency_s=0.4,
            estimated_cost_usd_per_1k_chars=0.08,
        ),
    }
    config_path = _write_config(tmp_path, providers)

    call_sequence: List[str] = []

    def _quota_adapter(request: tts_agent.AdapterRequest) -> tts_agent.AdapterResult:
        call_sequence.append(request.provider.provider_id)
        raise tts_agent.TTSQuotaExceeded("quota exceeded")

    def _fixture_adapter(request: tts_agent.AdapterRequest) -> tts_agent.AdapterResult:
        call_sequence.append(request.provider.provider_id)
        dest = request.output_dir / "fixture.wav"
        shutil.copyfile(deterministic_media_assets.audio, dest)
        return tts_agent.AdapterResult(
            path=dest,
            duration_s=0.5,
            sample_rate=request.voice.sample_rate,
            bit_depth=request.voice.bit_depth,
            watermarking=request.provider.watermarking,
            metadata={"adapter": request.provider.adapter},
        )

    tts_agent.register_adapter("pro_adapter", _quota_adapter)
    tts_agent.register_adapter("fixture_adapter", _fixture_adapter)

    artifact = tts_agent.synthesize(
        "Hello world",
        voice_config={"voice_id": "narrator"},
        config_path=config_path,
        plan_id="plan-1",
        step_id="step-tts",
        run_id="run-123",
    )

    assert artifact["uri"].startswith("file://")
    assert call_sequence == ["pro-cloud", "fixture-local"]
    published_meta = publish_backend[0]["metadata"]
    assert published_meta["provider_id"] == "fixture-local"
    assert published_meta["adapter_metadata"]["adapter"] == "fixture_adapter"
    assert published_meta["score_breakdown"]
    assert published_meta["selection_reason"] == "weighted"
    assert published_meta["estimated_cost_usd"] == pytest.approx(0.0)
    voice_meta = published_meta["voice_metadata"]
    assert voice_meta["voice_id"] == "narrator"
    assert voice_meta["provider_id"] == "fixture-local"


def test_retryable_error_retries_before_success(
    tmp_path: Path, publish_backend: List[Dict[str, Any]], deterministic_media_assets: MediaAssets
) -> None:
    providers = {
        "fixture-local": _provider_entry(
            adapter="retry_adapter",
            tier="tier1",
            fixture_alias="fixture",
            default_voice="retry-voice",
            quality_score=0.8,
            estimated_latency_s=0.2,
            estimated_cost_usd_per_1k_chars=0.02,
        ),
    }
    config_path = _write_config(tmp_path, providers)

    attempts = {"count": 0}

    def _sometimes_fails(request: tts_agent.AdapterRequest) -> tts_agent.AdapterResult:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise tts_agent.TTSRetryableError("transient")
        dest = request.output_dir / "retry.wav"
        shutil.copyfile(deterministic_media_assets.audio, dest)
        return tts_agent.AdapterResult(
            path=dest,
            duration_s=0.25,
            sample_rate=request.voice.sample_rate,
            bit_depth=request.voice.bit_depth,
            watermarking=request.provider.watermarking,
            metadata={"attempt": attempts["count"]},
        )

    tts_agent.register_adapter("retry_adapter", _sometimes_fails)

    sleeps: List[float] = []
    artifact = tts_agent.synthesize(
        "Hello retry",
        voice_config={"voice_id": "narrator"},
        config_path=config_path,
        plan_id="plan-2",
        step_id="step-tts",
        run_id="run-456",
        sleep_fn=lambda delay: sleeps.append(delay),
    )

    assert attempts["count"] == 2
    assert sleeps == pytest.approx([0.05])
    published_meta = publish_backend[0]["metadata"]
    assert published_meta["attempt"] == 2
    assert published_meta["provider_id"] == "fixture-local"
    assert published_meta["adapter_metadata"]["attempt"] == 2
    assert artifact["metadata"]["attempt"] == 2
    assert published_meta["estimated_cost_usd"] >= 0
    assert published_meta["voice_metadata"]["provider_voice_id"] == "retry-voice"


def test_retryable_error_honors_retry_after(
    tmp_path: Path, publish_backend: List[Dict[str, Any]], deterministic_media_assets: MediaAssets
) -> None:
    providers = {
        "fixture-local": _provider_entry(
            adapter="retry_after_adapter",
            tier="tier1",
            fixture_alias="fixture",
            default_voice="retry-voice",
            quality_score=0.8,
            estimated_latency_s=0.2,
            estimated_cost_usd_per_1k_chars=0.02,
        ),
    }
    config_path = _write_config(tmp_path, providers)

    attempts = {"count": 0}

    def _adapter(request: tts_agent.AdapterRequest) -> tts_agent.AdapterResult:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise tts_agent.TTSRetryableError("backoff", retry_after_s=1.75)
        dest = request.output_dir / "retry-after.wav"
        shutil.copyfile(deterministic_media_assets.audio, dest)
        return tts_agent.AdapterResult(
            path=dest,
            duration_s=0.3,
            sample_rate=request.voice.sample_rate,
            bit_depth=request.voice.bit_depth,
            watermarking=request.provider.watermarking,
            metadata={"attempt": attempts["count"]},
        )

    tts_agent.register_adapter("retry_after_adapter", _adapter)

    sleeps: List[float] = []
    artifact = tts_agent.synthesize(
        "Hello retry",
        voice_config={"voice_id": "narrator"},
        config_path=config_path,
        plan_id="plan-3",
        step_id="step-tts",
        run_id="run-789",
        sleep_fn=lambda delay: sleeps.append(delay),
    )

    assert attempts["count"] == 2
    assert sleeps == pytest.approx([1.75])
    assert artifact["metadata"]["attempt"] == 2
    assert artifact["metadata"]["voice_metadata"]["provider_voice_id"] == "retry-voice"


def test_max_cost_filters_providers(
    tmp_path: Path, publish_backend: List[Dict[str, Any]], deterministic_media_assets: MediaAssets
) -> None:
    providers = {
        "expensive": _provider_entry(
            adapter="expensive_adapter",
            tier="tier1",
            fixture_alias="exp",
            default_voice="exp-voice",
            quality_score=0.99,
            estimated_latency_s=0.4,
            estimated_cost_usd_per_1k_chars=5.0,
        ),
        "economy": _provider_entry(
            adapter="economy_adapter",
            tier="tier1",
            fixture_alias="eco",
            default_voice="eco-voice",
            quality_score=0.6,
            estimated_latency_s=0.3,
            estimated_cost_usd_per_1k_chars=0.01,
        ),
    }
    config_path = _write_config(tmp_path, providers)

    call_sequence: List[str] = []

    def _recording_adapter(request: tts_agent.AdapterRequest) -> tts_agent.AdapterResult:
        call_sequence.append(request.provider.provider_id)
        dest = request.output_dir / f"{request.provider.provider_id}.wav"
        shutil.copyfile(deterministic_media_assets.audio, dest)
        return tts_agent.AdapterResult(
            path=dest,
            duration_s=0.25,
            sample_rate=request.voice.sample_rate,
            bit_depth=request.voice.bit_depth,
            watermarking=request.provider.watermarking,
            metadata={"provider": request.provider.provider_id},
        )

    tts_agent.register_adapter("expensive_adapter", _recording_adapter)
    tts_agent.register_adapter("economy_adapter", _recording_adapter)

    artifact = tts_agent.synthesize(
        "Cost constrained line",
        voice_config={"voice_id": "narrator"},
        config_path=config_path,
        plan_id="plan-4",
        step_id="step-tts",
        run_id="run-101",
        max_cost_usd=0.001,
    )

    assert call_sequence == ["economy"]
    assert artifact["metadata"]["provider_id"] == "economy"
    assert artifact["metadata"]["estimated_cost_usd"] <= 0.001
    assert artifact["metadata"]["voice_metadata"]["provider_id"] == "economy"