from __future__ import annotations

import hashlib
import logging
import math
import os
import struct
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from sparkle_motion.utils.env import fixture_mode_enabled

LOG = logging.getLogger(__name__)
TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SynthesisResult:
    path: Path
    duration_s: float
    sample_rate: int
    bit_depth: int
    watermarking: bool
    metadata: Mapping[str, Any]


def should_use_real_engine(env: Mapping[str, str] | None = None) -> bool:
    data = env or os.environ
    if fixture_mode_enabled(data, default=False):
        return False
    for key in ("SMOKE_TTS", "SMOKE_ADAPTERS"):
        if data.get(key, "0").strip().lower() in TRUTHY:
            return True
    return False


def _cache_ttl_seconds() -> Optional[float]:
    raw = os.environ.get("TTS_CHATTERBOX_CACHE_TTL_S")
    if raw is None:
        return 900.0
    try:
        value = float(raw)
    except ValueError:
        return 900.0
    return max(value, 0.0)


def synthesize_text(
    *,
    text: str,
    voice_id: str,
    sample_rate: int,
    bit_depth: int,
    language: Optional[str],
    seed: Optional[int],
    watermarking: bool,
    output_dir: Path,
    metadata: Optional[Mapping[str, Any]] = None,
    force_fixture: Optional[bool] = None,
) -> SynthesisResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    use_real = should_use_real_engine() if force_fixture is None else not force_fixture
    if use_real:
        try:
            return _synthesize_real(
                text=text,
                voice_id=voice_id,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                language=language,
                seed=seed,
                watermarking=watermarking,
                output_dir=output_dir,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - exercised only in smoke mode
            LOG.warning("Real chatterbox synthesis failed, falling back to fixture: %s", exc)
    return _synthesize_fixture(
        text=text,
        voice_id=voice_id,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        language=language,
        seed=seed,
        watermarking=watermarking,
        output_dir=output_dir,
        metadata=metadata,
    )


def _synthesize_fixture(
    *,
    text: str,
    voice_id: str,
    sample_rate: int,
    bit_depth: int,
    language: Optional[str],
    seed: Optional[int],
    watermarking: bool,
    output_dir: Path,
    metadata: Optional[Mapping[str, Any]],
) -> SynthesisResult:
    cleaned = text.strip() or " "
    duration = max(len(cleaned) / 18.0, 0.3)
    frames = max(int(sample_rate * duration), sample_rate)
    frames = min(frames, sample_rate * 10)
    if bit_depth != 16:
        raise ValueError("fixture synthesis currently supports 16-bit output only")
    base_seed = seed if seed is not None else int(hashlib.sha1(cleaned.encode("utf-8")).hexdigest(), 16)
    freq = 180 + (base_seed % 200)
    max_val = (1 << (bit_depth - 1)) - 1
    data = bytearray()
    angle_step = 2.0 * math.pi * freq / sample_rate
    angle = 0.0
    for _ in range(frames):
        value = int(math.sin(angle) * 0.25 * max_val)
        data.extend(struct.pack("<h", value))
        angle += angle_step
    dest = output_dir / f"fixture-{voice_id}-{uuid.uuid4().hex}.wav"
    with wave.open(dest.as_posix(), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(data))
    meta = {
        "engine": "fixture",
        "voice_id": voice_id,
        "language": language or "auto",
        "seed": base_seed,
        "text_length": len(cleaned),
        "mode": "fixture",
    }
    if metadata:
        meta.update(metadata)
    return SynthesisResult(
        path=dest,
        duration_s=frames / sample_rate,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        watermarking=watermarking,
        metadata=meta,
    )


def _synthesize_real(
    *,
    text: str,
    voice_id: str,
    sample_rate: int,
    bit_depth: int,
    language: Optional[str],
    seed: Optional[int],
    watermarking: bool,
    output_dir: Path,
    metadata: Optional[Mapping[str, Any]],
) -> SynthesisResult:
    try:  # pragma: no cover - requires heavy deps
        from chatterbox.tts import ChatterboxTTS  # type: ignore
        import torchaudio  # type: ignore
    except Exception as exc:  # pragma: no cover - executed only when deps missing
        raise RuntimeError(
            "chatterbox-tts and torchaudio must be installed to enable SMOKE_TTS runs"
        ) from exc

    from sparkle_motion import gpu_utils  # local import to avoid heavy deps unless needed

    model_id = os.environ.get("TTS_CHATTERBOX_MODEL", "ResembleAI/chatterbox")
    device = os.environ.get("TTS_CHATTERBOX_DEVICE", "cuda")
    cache_ttl = _cache_ttl_seconds()

    def _loader() -> Any:  # pragma: no cover - heavy path
        return ChatterboxTTS.from_pretrained(device=device)

    frame_count = 0
    with gpu_utils.model_context(
        f"tts_chatterbox:{model_id}",
        loader=_loader,
        weights=model_id,
        offload=False,
        xformers=False,
        keep_warm=True,
        warm_ttl_s=cache_ttl,
        block_until_gpu_free=False,
    ) as ctx:  # pragma: no cover - smoke path only
        generator_kwargs: Dict[str, Any] = {}
        if language:
            generator_kwargs["language_id"] = language
        filtered = {k: v for k, v in (metadata or {}).items() if k in {"cfg_weight", "exaggeration", "audio_prompt_path"}}
        generator_kwargs.update(filtered)
        wav = ctx.pipeline.generate(text, **generator_kwargs)
        active_sample_rate = getattr(ctx.pipeline, "sr", sample_rate)
        dest = output_dir / f"chatterbox-{voice_id}-{uuid.uuid4().hex}.wav"
        torchaudio.save(dest.as_posix(), wav, active_sample_rate)
        shape = getattr(wav, "shape", None)
        if shape:
            try:
                frame_count = int(shape[-1])
            except Exception:  # pragma: no cover - defensive
                frame_count = 0

    meta = {
        "engine": "chatterbox",
        "model_id": model_id,
        "device": device,
        "voice_id": voice_id,
        "language": language or "auto",
        "seed": seed,
        "mode": "real",
    }
    if metadata:
        meta.update(metadata)
    duration = frame_count / active_sample_rate if frame_count and active_sample_rate else 0.0
    return SynthesisResult(
        path=dest,
        duration_s=duration,
        sample_rate=active_sample_rate,
        bit_depth=bit_depth,
        watermarking=watermarking,
        metadata=meta,
    )


def create_tts_stage_adapter(adapter_result_cls: Any) -> Any:
    def _adapter(request: Any) -> Any:
        opts = getattr(request, "options", {}) or {}
        result = synthesize_text(
            text=request.text,
            voice_id=request.voice.voice_id,
            sample_rate=request.voice.sample_rate,
            bit_depth=request.voice.bit_depth,
            language=opts.get("language"),
            seed=opts.get("seed"),
            watermarking=request.provider.watermarking,
            output_dir=request.output_dir,
            metadata={"provider_voice_id": request.voice.provider_voice_id},
        )
        return adapter_result_cls(
            path=result.path,
            duration_s=result.duration_s,
            sample_rate=result.sample_rate,
            bit_depth=result.bit_depth,
            watermarking=result.watermarking,
            metadata=dict(result.metadata),
        )

    return _adapter
