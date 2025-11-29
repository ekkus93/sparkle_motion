from __future__ import annotations

import os
import random
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from . import adk_helpers, observability, telemetry

__all__ = [
    "synthesize",
    "get_voice_metadata",
    "list_available_voices",
    "register_adapter",
    "TTSError",
    "ProviderSelectionError",
    "TTSRetryableError",
    "TTSQuotaExceeded",
    "TTSInvalidInputError",
    "TTSPolicyViolation",
    "VoiceMetadata",
]


class TTSError(RuntimeError):
    """Base error raised by tts_agent."""


class ProviderSelectionError(TTSError):
    """Raised when no provider can satisfy the request."""


class TTSRetryableError(TTSError):
    """Raised by adapters to request a retry with the same provider."""

    def __init__(self, message: str, retry_after_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class TTSQuotaExceeded(TTSError):
    """Raised when a provider reports quota exhaustion and suggests failover."""


class TTSInvalidInputError(TTSError):
    """Raised when the request payload cannot be fulfilled by any provider."""


class TTSPolicyViolation(TTSError):
    """Raised when the text violates the safety/policy guardrails."""


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    backoff_s: float


@dataclass(frozen=True)
class RateLimitPolicy:
    per_minute: int
    burst: int


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    display_name: str
    tier: str
    adapter: str
    fixture_alias: str
    default_voice: str
    estimated_latency_s: float
    estimated_cost_usd_per_1k_chars: float
    quality_score: float
    features: tuple[str, ...]
    languages: tuple[str, ...]
    rate_limits: RateLimitPolicy
    retry_policy: RetryPolicy
    watermarking: bool


@dataclass(frozen=True)
class VoicePreference:
    provider: str
    provider_voice_id: str


@dataclass(frozen=True)
class VoiceProfile:
    voice_id: str
    description: str
    default_sample_rate: int
    default_bit_depth: int
    preferences: tuple[VoicePreference, ...]


@dataclass(frozen=True)
class RateCap:
    daily_requests: int
    concurrent_jobs: int


@dataclass(frozen=True)
class PriorityProfile:
    weights: Mapping[str, float]


@dataclass(frozen=True)
class VoiceMetadata:
    voice_id: str
    provider_id: str
    provider_voice_id: str
    description: str
    display_name: str
    language_codes: tuple[str, ...]
    features: tuple[str, ...]
    sample_rate: int
    bit_depth: int
    watermarking: bool
    estimated_cost_usd_per_1k_chars: float
    estimated_latency_s: float
    quality_score: float


@dataclass(frozen=True)
class AdapterRequest:
    text: str
    voice: VoiceMetadata
    provider: ProviderConfig
    plan_id: str
    step_id: str
    run_id: str
    output_dir: Path
    options: Mapping[str, Any]


@dataclass(frozen=True)
class AdapterResult:
    path: Path
    duration_s: float
    sample_rate: int
    bit_depth: int
    watermarking: bool
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SelectionDecision:
    provider_id: str
    score: float
    breakdown: Mapping[str, float]
    reason: str
    estimated_cost_usd: float


@dataclass(frozen=True)
class TTSConfig:
    providers: Mapping[str, ProviderConfig]
    voices: Mapping[str, VoiceProfile]
    priority_profiles: Mapping[str, PriorityProfile]
    rate_caps: Mapping[str, RateCap]


AdapterCallable = Callable[[AdapterRequest], AdapterResult]


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tts_providers.yaml"
_CONFIG_CACHE: Optional[Tuple[Path, float, TTSConfig]] = None
_ADAPTERS: Dict[str, AdapterCallable] = {}
_TRUTHY = {"1", "true", "yes", "on"}
_ARTIFACT_TYPE = "tts_audio"
_POLICY_BLOCKLIST = {"weaponized", "forbidden", "terror", "bomb"}
_RETRY_JITTER_FRACTION = 0.2


def _random_uniform(a: float, b: float) -> float:
    return random.uniform(a, b)


def _compute_retry_delay(policy: RetryPolicy, attempt: int, override: Optional[float]) -> float:
    attempt = max(attempt, 1)
    base = override if override is not None else max(policy.backoff_s, 0.0) * attempt
    if base <= 0:
        return 0.0
    jitter = base * _RETRY_JITTER_FRACTION
    if jitter <= 0:
        return base
    delta = _random_uniform(-jitter, jitter)
    return max(base + delta, 0.0)


def _voice_metadata_payload(voice: VoiceMetadata) -> Dict[str, Any]:
    return {
        "voice_id": voice.voice_id,
        "provider_id": voice.provider_id,
        "provider_voice_id": voice.provider_voice_id,
        "description": voice.description,
        "display_name": voice.display_name,
        "language_codes": list(voice.language_codes),
        "features": list(voice.features),
        "sample_rate": voice.sample_rate,
        "bit_depth": voice.bit_depth,
        "watermarking": voice.watermarking,
        "estimated_cost_usd_per_1k_chars": voice.estimated_cost_usd_per_1k_chars,
        "estimated_latency_s": voice.estimated_latency_s,
        "quality_score": voice.quality_score,
    }


def register_adapter(name: str, adapter: AdapterCallable) -> None:
    """Register or replace an adapter callable for a provider adapter id."""

    _ADAPTERS[name] = adapter


def list_available_voices(provider: Optional[str] = None) -> list[VoiceMetadata]:
    """Return VoiceMetadata entries for every voice/provider pair."""

    cfg = _load_config()
    voices: list[VoiceMetadata] = []
    for voice in cfg.voices.values():
        voices.extend(_expand_voice_metadata(voice, cfg.providers, provider_filter=provider))
    return voices


def get_voice_metadata(voice_id: str, provider: Optional[str] = None) -> VoiceMetadata:
    """Resolve a single VoiceMetadata entry, optionally pinning to a provider."""

    cfg = _load_config()
    if voice_id not in cfg.voices:
        raise TTSError(f"Unknown voice_id '{voice_id}'")
    candidates = _expand_voice_metadata(cfg.voices[voice_id], cfg.providers, provider_filter=provider)
    if not candidates:
        raise TTSError(f"No providers configured for voice_id '{voice_id}'")
    return candidates[0]


def synthesize(
    text: str,
    voice_config: Optional[Mapping[str, Any]] = None,
    *,
    priority_profile: str = "balanced",
    required_features: Optional[Sequence[str]] = None,
    max_latency_s: Optional[float] = None,
    max_cost_usd: Optional[float] = None,
    provider_override: Optional[str] = None,
    adapter_overrides: Optional[Mapping[str, AdapterCallable]] = None,
    output_dir: Optional[Path] = None,
    plan_id: Optional[str] = None,
    step_id: Optional[str] = None,
    run_id: Optional[str] = None,
    config_path: Optional[Path] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> adk_helpers.ArtifactRef:
    """Synthesize speech using the configured provider selection rules."""

    if not text or not text.strip():
        raise TTSInvalidInputError("text must be a non-empty string")
    _ensure_policy(text)

    cfg = _load_config(config_path)
    resolved_run = run_id or observability.get_session_id()
    resolved_plan = plan_id or "plan-unknown"
    resolved_step = step_id or "tts"
    sleep = sleep_fn or time.sleep
    base_output_dir = output_dir or Path(tempfile.mkdtemp(prefix="tts_agent_"))
    base_output_dir.mkdir(parents=True, exist_ok=True)
    text_chars = len(text)

    voice_id = _resolve_voice_id(voice_config, cfg)
    voice_profile = cfg.voices[voice_id]
    priority = cfg.priority_profiles.get(priority_profile) or cfg.priority_profiles.get("balanced")
    if priority is None:
        raise TTSError(f"Priority profile '{priority_profile}' missing from config")

    enforced_fixture = _fixture_only_mode()
    candidates = _build_candidate_list(
        cfg,
        voice_profile,
        priority,
        required_features=required_features,
        max_latency_s=max_latency_s,
        max_cost_usd=max_cost_usd,
        provider_override=provider_override,
        enforce_fixture=enforced_fixture,
        text_length=text_chars,
    )
    if not candidates:
        raise ProviderSelectionError("No providers available after applying constraints")

    telemetry.emit_event(
        "tts_agent.synthesize.start",
        {
            "plan_id": resolved_plan,
            "step_id": resolved_step,
            "run_id": resolved_run,
            "text_chars": text_chars,
            "voice_id": voice_id,
            "voice_description": voice_profile.description,
            "voice_default_sample_rate": voice_profile.default_sample_rate,
            "voice_default_bit_depth": voice_profile.default_bit_depth,
            "candidate_count": len(candidates),
        },
    )

    errors: list[str] = []
    last_exception: Optional[Exception] = None

    for provider, selection in candidates:
        resolved_voice = _resolve_voice_metadata(voice_profile, provider)
        adapter = _resolve_adapter(provider, adapter_overrides)
        policy = provider.retry_policy
        attempts = max(1, 1 + policy.max_retries)
        attempt = 0
        while attempt < attempts:
            attempt += 1
            request = AdapterRequest(
                text=text,
                voice=resolved_voice,
                provider=provider,
                plan_id=resolved_plan,
                step_id=resolved_step,
                run_id=resolved_run,
                output_dir=base_output_dir,
                options=dict(voice_config or {}),
            )
            telemetry.emit_event(
                "tts_agent.provider.attempt",
                {
                    "provider_id": provider.provider_id,
                    "attempt": attempt,
                    "plan_id": resolved_plan,
                    "step_id": resolved_step,
                    "score": selection.score,
                },
            )
            try:
                result = adapter(request)
                metadata = _build_artifact_metadata(
                    result,
                    request,
                    selection,
                    attempt=attempt,
                    total_text=text,
                )
                voice_payload = metadata.get("voice_metadata") or {}
                artifact = adk_helpers.publish_artifact(
                    local_path=result.path,
                    artifact_type=_ARTIFACT_TYPE,
                    metadata=metadata,
                )
                _record_memory_event(
                    resolved_run,
                    {
                        "plan_id": resolved_plan,
                        "step_id": resolved_step,
                        "provider_id": provider.provider_id,
                        "voice_id": resolved_voice.voice_id,
                        "artifact_uri": artifact.get("uri"),
                        "score_breakdown": dict(selection.breakdown),
                        "voice_metadata": dict(voice_payload),
                    },
                )
                telemetry.emit_event(
                    "tts_agent.synthesize.completed",
                    {
                        "provider_id": provider.provider_id,
                        "attempt": attempt,
                        "plan_id": resolved_plan,
                        "step_id": resolved_step,
                        "artifact_uri": artifact.get("uri"),
                        "voice_metadata": dict(voice_payload),
                    },
                )
                return artifact
            except TTSInvalidInputError:
                raise
            except TTSQuotaExceeded as exc:
                msg = f"provider {provider.provider_id} quota exceeded: {exc}"
                errors.append(msg)
                last_exception = exc
                break
            except TTSRetryableError as exc:
                last_exception = exc
                if attempt >= attempts:
                    errors.append(f"provider {provider.provider_id} retry limit reached: {exc}")
                    break
                delay = _compute_retry_delay(policy, attempt, exc.retry_after_s)
                sleep(delay)
                continue
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_exception = exc
                errors.append(f"provider {provider.provider_id} failed: {exc}")
                break

    message = "; ".join(errors) if errors else "no provider could satisfy the request"
    raise ProviderSelectionError(message) from last_exception


def _build_candidate_list(
    cfg: TTSConfig,
    voice: VoiceProfile,
    priority: PriorityProfile,
    *,
    required_features: Optional[Sequence[str]],
    max_latency_s: Optional[float],
    max_cost_usd: Optional[float],
    provider_override: Optional[str],
    enforce_fixture: bool,
    text_length: int,
) -> list[tuple[ProviderConfig, SelectionDecision]]:
    candidates: list[tuple[ProviderConfig, SelectionDecision]] = []
    weights = priority.weights
    override = provider_override.lower().strip() if provider_override else None
    required = {feat.lower() for feat in (required_features or [])}
    text_length = max(text_length, 0)

    ordered_ids: list[str] = []
    if override:
        for provider in cfg.providers.values():
            aliases = {provider.provider_id.lower(), provider.fixture_alias.lower()}
            if override in aliases:
                ordered_ids.append(provider.provider_id)
                break
    if not ordered_ids:
        ordered_ids.extend([pref.provider for pref in voice.preferences if pref.provider in cfg.providers])
        for provider_id in cfg.providers:
            if provider_id not in ordered_ids:
                ordered_ids.append(provider_id)

    seen: set[str] = set()
    for provider_id in ordered_ids:
        if provider_id in seen:
            continue
        provider = cfg.providers.get(provider_id)
        if provider is None:
            continue
        seen.add(provider_id)
        if enforce_fixture and provider.tier != "fixture" and not override:
            continue
        if required and not required.issubset({f.lower() for f in provider.features}):
            continue
        if max_latency_s is not None and provider.estimated_latency_s > max_latency_s:
            continue
        per_request_cost = provider.estimated_cost_usd_per_1k_chars * (text_length / 1000.0)
        if max_cost_usd is not None and per_request_cost > max_cost_usd:
            continue
        score, breakdown = _score_provider(provider, weights)
        override_hit = override in {provider.provider_id.lower(), provider.fixture_alias.lower()} if override else False
        reason = "override" if override_hit else "weighted"
        decision = SelectionDecision(
            provider_id=provider.provider_id,
            score=score,
            breakdown=breakdown,
            reason=reason,
            estimated_cost_usd=round(per_request_cost, 4),
        )
        candidates.append((provider, decision))

    candidates.sort(key=lambda item: item[1].score, reverse=True)
    return candidates


def _score_provider(provider: ProviderConfig, weights: Mapping[str, float]) -> tuple[float, Dict[str, float]]:
    latency_score = 1.0 / (1.0 + max(provider.estimated_latency_s, 0.01))
    cost_score = 1.0 / (1.0 + max(provider.estimated_cost_usd_per_1k_chars, 0.0001))
    quality_score = max(0.0, min(provider.quality_score, 1.0))
    breakdown = {
        "quality": quality_score,
        "latency": latency_score,
        "cost": cost_score,
    }
    score = sum(weights.get(metric, 0.0) * breakdown.get(metric, 0.0) for metric in ("quality", "latency", "cost"))
    return score, breakdown


def _resolve_adapter(provider: ProviderConfig, adapter_overrides: Optional[Mapping[str, AdapterCallable]]) -> AdapterCallable:
    if adapter_overrides:
        if provider.provider_id in adapter_overrides:
            return adapter_overrides[provider.provider_id]
        if provider.adapter in adapter_overrides:
            return adapter_overrides[provider.adapter]
    if provider.adapter in _ADAPTERS:
        return _ADAPTERS[provider.adapter]
    if "tts_fixture" in _ADAPTERS:
        return _ADAPTERS["tts_fixture"]
    return _fixture_adapter


def _resolve_voice_metadata(voice: VoiceProfile, provider: ProviderConfig) -> VoiceMetadata:
    provider_voice_id = provider.default_voice
    for pref in voice.preferences:
        if pref.provider == provider.provider_id:
            provider_voice_id = pref.provider_voice_id
            break
    display_name = voice.voice_id.replace("_", " ").title()
    return VoiceMetadata(
        voice_id=voice.voice_id,
        provider_id=provider.provider_id,
        provider_voice_id=provider_voice_id,
        description=voice.description,
        display_name=display_name,
        language_codes=provider.languages,
        features=provider.features,
        sample_rate=voice.default_sample_rate,
        bit_depth=voice.default_bit_depth,
        watermarking=provider.watermarking,
        estimated_cost_usd_per_1k_chars=provider.estimated_cost_usd_per_1k_chars,
        estimated_latency_s=provider.estimated_latency_s,
        quality_score=provider.quality_score,
    )


def _expand_voice_metadata(
    voice: VoiceProfile,
    providers: Mapping[str, ProviderConfig],
    *,
    provider_filter: Optional[str] = None,
) -> list[VoiceMetadata]:
    selected: list[VoiceMetadata] = []
    lower_filter = provider_filter.lower() if provider_filter else None
    for pref in voice.preferences:
        provider = providers.get(pref.provider)
        if provider is None:
            continue
        if lower_filter and provider.provider_id.lower() != lower_filter:
            continue
        selected.append(_resolve_voice_metadata(voice, provider))
    if not selected:
        for provider in providers.values():
            if lower_filter and provider.provider_id.lower() != lower_filter:
                continue
            selected.append(_resolve_voice_metadata(voice, provider))
            break
    return selected


def _resolve_voice_id(voice_config: Optional[Mapping[str, Any]], cfg: TTSConfig) -> str:
    if voice_config is None:
        return next(iter(cfg.voices))
    if isinstance(voice_config, str):
        if voice_config in cfg.voices:
            return voice_config
        raise TTSError(f"Unknown voice_id '{voice_config}'")
    for key in ("voice_id", "voice", "id"):
        value = voice_config.get(key)
        if isinstance(value, str) and value in cfg.voices:
            return value
    default_voice = voice_config.get("default")
    if isinstance(default_voice, str) and default_voice in cfg.voices:
        return default_voice
    if cfg.voices:
        return next(iter(cfg.voices))
    raise TTSError("No voices configured")


def _build_artifact_metadata(
    result: AdapterResult,
    request: AdapterRequest,
    selection: SelectionDecision,
    *,
    attempt: int,
    total_text: str,
) -> Dict[str, Any]:
    voice_payload = _voice_metadata_payload(request.voice)
    metadata: Dict[str, Any] = {
        "plan_id": request.plan_id,
        "step_id": request.step_id,
        "voice_id": request.voice.voice_id,
        "provider_id": request.provider.provider_id,
        "provider_voice_id": request.voice.provider_voice_id,
        "duration_s": round(result.duration_s, 3),
        "sample_rate": result.sample_rate,
        "bit_depth": result.bit_depth,
        "watermarked": bool(result.watermarking),
        "text_characters": len(total_text),
        "attempt": attempt,
        "score_breakdown": dict(selection.breakdown),
        "selection_reason": selection.reason,
        "selection_score": selection.score,
        "estimated_cost_usd": selection.estimated_cost_usd,
        "adapter_metadata": dict(result.metadata),
        "voice_metadata": voice_payload,
    }
    metadata.setdefault("model_id", request.provider.display_name)
    return metadata


def _record_memory_event(run_id: str, payload: Mapping[str, Any]) -> None:
    try:
        adk_helpers.write_memory_event(
            run_id=run_id,
            event_type="tts_agent.synthesize",
            payload=dict(payload),
        )
    except adk_helpers.MemoryWriteError:
        pass


def _load_config(path: Optional[Path] = None) -> TTSConfig:
    global _CONFIG_CACHE
    resolved = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise TTSError(f"Config file not found: {resolved}")
    stamp = resolved.stat().st_mtime
    if _CONFIG_CACHE and _CONFIG_CACHE[0] == resolved and _CONFIG_CACHE[1] == stamp:
        return _CONFIG_CACHE[2]

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    providers = {
        pid: ProviderConfig(
            provider_id=pid,
            display_name=entry.get("display_name", pid),
            tier=entry.get("tier", "tier2"),
            adapter=entry.get("adapter", "tts_fixture"),
            fixture_alias=entry.get("fixture_alias", pid),
            default_voice=entry.get("default_voice", ""),
            estimated_latency_s=float(entry.get("estimated_latency_s", 1.0)),
            estimated_cost_usd_per_1k_chars=float(entry.get("estimated_cost_usd_per_1k_chars", 0.0)),
            quality_score=float(entry.get("quality_score", 0.5)),
            features=tuple(entry.get("features", []) or ()),
            languages=tuple(entry.get("languages", []) or ()),
            rate_limits=RateLimitPolicy(
                per_minute=int(entry.get("rate_limits", {}).get("per_minute", 60)),
                burst=int(entry.get("rate_limits", {}).get("burst", 10)),
            ),
            retry_policy=RetryPolicy(
                max_retries=int(entry.get("retry_policy", {}).get("max_retries", 1)),
                backoff_s=float(entry.get("retry_policy", {}).get("backoff_s", 1.0)),
            ),
            watermarking=bool(entry.get("watermarking", False)),
        )
        for pid, entry in (raw.get("providers") or {}).items()
    }
    voices = {
        vid: VoiceProfile(
            voice_id=vid,
            description=entry.get("description", vid),
            default_sample_rate=int(entry.get("default_sample_rate", 24000)),
            default_bit_depth=int(entry.get("default_bit_depth", 16)),
            preferences=tuple(
                VoicePreference(provider=pref.get("provider"), provider_voice_id=pref.get("provider_voice_id", vid))
                for pref in (entry.get("provider_preferences") or [])
                if pref.get("provider") in providers
            ),
        )
        for vid, entry in (raw.get("voices") or {}).items()
    }
    priority_profiles = {
        name: PriorityProfile(weights=_normalize_weights(entry.get("weights") or {}))
        for name, entry in (raw.get("priority_profiles") or {}).items()
    }
    rate_caps = {
        tier: RateCap(
            daily_requests=int(entry.get("daily_requests", 0)),
            concurrent_jobs=int(entry.get("concurrent_jobs", 0)),
        )
        for tier, entry in (raw.get("rate_caps") or {}).items()
    }
    config = TTSConfig(providers=providers, voices=voices, priority_profiles=priority_profiles, rate_caps=rate_caps)
    _CONFIG_CACHE = (resolved, stamp, config)
    return config


def _normalize_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    floats = {k: float(v) for k, v in weights.items()}
    total = sum(floats.values())
    if total <= 0:
        return {"quality": 0.5, "latency": 0.25, "cost": 0.25}
    return {k: v / total for k, v in floats.items()}


def _ensure_policy(text: str) -> None:
    lower = text.lower()
    for token in _POLICY_BLOCKLIST:
        if token in lower:
            raise TTSPolicyViolation(f"Text contains disallowed content: '{token}'")


def _fixture_only_mode() -> bool:
    smoke_tts = os.environ.get("SMOKE_TTS", "0").strip().lower() in _TRUTHY
    smoke_adapters = os.environ.get("SMOKE_ADAPTERS", "0").strip().lower() in _TRUTHY
    return not (smoke_tts or smoke_adapters)


def _fixture_adapter(request: AdapterRequest) -> AdapterResult:
    duration = max(len(request.text.strip()) / 18.0, 0.3)
    frames = max(int(request.voice.sample_rate * duration), request.voice.sample_rate)
    frames = min(frames, request.voice.sample_rate * 10)
    dest = request.output_dir / f"{request.provider.provider_id}-{uuid.uuid4().hex}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(dest.as_posix(), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(request.voice.bit_depth // 8)
        wav_file.setframerate(request.voice.sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)
    metadata = {
        "fixture": True,
        "provider_voice_id": request.voice.provider_voice_id,
    }
    return AdapterResult(
        path=dest,
        duration_s=frames / request.voice.sample_rate,
        sample_rate=request.voice.sample_rate,
        bit_depth=request.voice.bit_depth,
        watermarking=request.provider.watermarking,
        metadata=metadata,
    )


def _fixture_adapter_once() -> None:
    if "tts_fixture" not in _ADAPTERS:
        register_adapter("tts_fixture", _fixture_adapter)


_fixture_adapter_once()


def _register_chatterbox_adapter() -> None:
    try:
        from sparkle_motion.function_tools.tts_chatterbox import adapter as _cb_adapter
    except Exception:
        return
    try:
        register_adapter("tts_chatterbox", _cb_adapter.create_tts_agent_adapter(AdapterResult))
    except Exception:
        pass


_register_chatterbox_adapter()
