from __future__ import annotations

import shutil
import uuid
import wave
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from sparkle_motion import adk_helpers, tts_stage

from . import helpers


def _require_chatterbox_stack() -> None:
    pytest.importorskip("chatterbox.tts", reason="Chatterbox TTS dependency missing")
    pytest.importorskip("torchaudio", reason="torchaudio dependency missing")


def _stub_publish_artifact(monkeypatch: "pytest.MonkeyPatch", dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _fake_publish_artifact(
        *,
        local_path: str | Path,
        artifact_type: str,
        metadata: Optional[Mapping[str, Any]] = None,
        media_type: Optional[str] = None,
        run_id: Optional[str] = None,
        dry_run: bool = False,
        project: Optional[str] = None,
    ) -> adk_helpers.ArtifactRef:
        source = Path(local_path)
        copied = dest_dir / source.name
        shutil.copy2(source, copied)
        meta = dict(metadata or {})
        meta.setdefault("source_path", str(source))
        meta.setdefault("local_path", str(copied))
        meta.setdefault("media_type", media_type or meta.get("media_type") or "audio/wav")
        return {
            "uri": copied.as_uri(),
            "storage": "local",
            "artifact_type": artifact_type,
            "media_type": meta["media_type"],
            "metadata": meta,
            "run_id": run_id or "gpu-tts-test",
        }

    monkeypatch.setattr(adk_helpers, "publish_artifact", _fake_publish_artifact)


def _prepare_real_tts_env(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_TTS", "SMOKE_ADAPTERS"],
        disable_keys=["ADK_USE_FIXTURE"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "TTS_CHATTERBOX_DEVICE": "cuda",
            "SPARKLE_LOCAL_RUNS_ROOT": str(tmp_path / "runs"),
        },
    )
    _stub_publish_artifact(monkeypatch, tmp_path / "published")


def _read_wav_duration(path: Path) -> float:
    with wave.open(path.as_posix(), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    if rate == 0:
        raise AssertionError("WAV file has zero sample rate")
    return frames / float(rate)


def _copy_sample_audio(dest: Path) -> tuple[float, int]:
    shutil.copy2(helpers.asset_path("sample_audio.wav"), dest)
    with wave.open(dest.as_posix(), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / float(rate), rate


def _synthesize_lines(
    *,
    lines: Sequence[str],
    output_root: Path,
    voice_config: Mapping[str, Any],
    plan_id: str,
    step_prefix: str,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for idx, text in enumerate(lines):
        artifact = tts_stage.synthesize(
            text=text,
            voice_config=dict(voice_config),
            output_dir=output_root / f"line_{idx}",
            plan_id=plan_id,
            step_id=f"{step_prefix}-{idx}",
            run_id="gpu-tts-run",
        )
        metadata = artifact["metadata"]
        source_path = metadata.get("source_path")
        assert isinstance(source_path, str) and source_path, "metadata must include source_path"
        wav_path = Path(source_path)
        artifacts.append(
            {
                "artifact": artifact,
                "metadata": metadata,
                "wav_path": wav_path,
                "sha": helpers.file_sha256(wav_path),
            }
        )
    return artifacts


@pytest.mark.gpu
def test_tts_synthesize_single_line(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()
    _require_chatterbox_stack()
    _prepare_real_tts_env(monkeypatch, tmp_path)

    output_dir = tmp_path / "tts_outputs"
    artifact = tts_stage.synthesize(
        text="Hello, world.",
        voice_config={"voice_id": "emma", "seed": 42},
        provider_override="chatterbox-local",
        output_dir=output_dir,
        plan_id="gpu-test-plan",
        step_id="tts-single-line",
        run_id="gpu-tts-run",
    )

    metadata = artifact["metadata"]
    source_path = metadata.get("source_path")
    assert isinstance(source_path, str) and source_path, "Artifact metadata must include source_path"
    wav_path = Path(source_path)
    assert wav_path.exists(), "Chatterbox synthesis must create a WAV artifact"
    assert wav_path.suffix == ".wav", "TTS artifact should be a WAV file"

    duration = _read_wav_duration(wav_path)
    assert duration > 0.5, "Single line should exceed half a second"

    assert metadata.get("provider_id") == "chatterbox-local"
    assert metadata.get("voice_id") == "emma"
    assert metadata.get("sample_rate", 0) >= 16000
    assert metadata.get("watermarked") is True

    adapter_meta = metadata.get("adapter_metadata") or {}
    assert adapter_meta.get("engine") == "chatterbox"
    assert adapter_meta.get("mode") == "real"


@pytest.mark.gpu
def test_tts_voice_profile_routing(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()
    _require_chatterbox_stack()
    _prepare_real_tts_env(monkeypatch, tmp_path)

    output_dir = tmp_path / "tts_voice_route"
    artifact = tts_stage.synthesize(
        text="Route this voice through adk-edge",
        voice_config={"voice_id": "emma", "seed": 314},
        provider_override="adk-edge",
        output_dir=output_dir,
        plan_id="gpu-test-plan",
        step_id="tts-voice-routing",
        run_id="gpu-tts-run",
    )

    metadata = artifact["metadata"]
    assert metadata.get("provider_id") == "adk-edge"
    assert metadata.get("provider_voice_id") == "adk_voice_en_f1"
    assert metadata.get("watermarked") is False

    voice_meta = metadata.get("voice_metadata") or {}
    assert voice_meta.get("provider_id") == "adk-edge"
    assert voice_meta.get("provider_voice_id") == "adk_voice_en_f1"

    adapter_meta = metadata.get("adapter_metadata") or {}
    assert adapter_meta.get("engine") == "chatterbox"
    assert adapter_meta.get("mode") == "real"


@pytest.mark.gpu
def test_tts_quota_handling(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()
    _prepare_real_tts_env(monkeypatch, tmp_path)

    def _quota_adapter(request: Any) -> tts_stage.AdapterResult:
        if request.provider.provider_id == "chatterbox-local":
            raise tts_stage.TTSQuotaExceeded("primary provider quota exhausted")

        dest = request.output_dir / f"{request.provider.provider_id}-{uuid.uuid4().hex}.wav"
        duration, rate = _copy_sample_audio(dest)
        return tts_stage.AdapterResult(
            path=dest,
            duration_s=duration,
            sample_rate=rate,
            bit_depth=request.voice.bit_depth,
            watermarking=request.provider.watermarking,
            metadata={
                "engine": f"stub-{request.provider.provider_id}",
                "provider_voice_id": request.voice.provider_voice_id,
            },
        )

    output_dir = tmp_path / "tts_quota"
    artifact = tts_stage.synthesize(
        text="Handle quota exhaustion by falling back.",
        voice_config={"voice_id": "emma", "seed": 21},
        output_dir=output_dir,
        plan_id="gpu-test-plan",
        step_id="tts-quota",
        run_id="gpu-tts-run",
        adapter_overrides={"tts_chatterbox": _quota_adapter},
    )

    metadata = artifact["metadata"]
    assert metadata.get("provider_id") == "adk-edge"
    assert metadata.get("provider_voice_id") == "adk_voice_en_f1"

    adapter_meta = metadata.get("adapter_metadata") or {}
    assert adapter_meta.get("engine") == "stub-adk-edge"
    assert metadata.get("watermarked") is False


@pytest.mark.gpu
def test_tts_multiple_lines_deterministic(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    helpers.require_gpu_available()
    _require_chatterbox_stack()
    _prepare_real_tts_env(monkeypatch, tmp_path)

    lines = ["Line one.", "Line two.", "Line three."]
    voice_cfg = {"voice_id": "emma", "seed": 100}

    first = _synthesize_lines(
        lines=lines,
        output_root=tmp_path / "tts_lines_a",
        voice_config=voice_cfg,
        plan_id="gpu-test-plan",
        step_prefix="tts-multi-a",
    )
    second = _synthesize_lines(
        lines=lines,
        output_root=tmp_path / "tts_lines_b",
        voice_config=voice_cfg,
        plan_id="gpu-test-plan",
        step_prefix="tts-multi-b",
    )

    assert len(first) == len(second) == len(lines)
    for text, run_a, run_b in zip(lines, first, second):
        wav_a = run_a["wav_path"]
        wav_b = run_b["wav_path"]
        assert wav_a.exists()
        assert wav_b.exists()
        assert run_a["sha"] == run_b["sha"], f"Audio mismatch for line: {text}"

        meta_a = run_a["metadata"]
        meta_b = run_b["metadata"]
        assert meta_a.get("voice_id") == meta_b.get("voice_id") == "emma"
        assert meta_a.get("provider_id") == meta_b.get("provider_id") == "chatterbox-local"

        artifact_uri_a = run_a["artifact"].get("uri")
        artifact_uri_b = run_b["artifact"].get("uri")
        assert isinstance(artifact_uri_a, str) and artifact_uri_a
        assert isinstance(artifact_uri_b, str) and artifact_uri_b

        duration_a = _read_wav_duration(wav_a)
        duration_b = _read_wav_duration(wav_b)
        assert duration_a > 0.3 and pytest.approx(duration_a, rel=0.01) == duration_b