from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Protocol, Sequence

from sparkle_motion.schemas import MoviePlan


class DialogueTimelineError(RuntimeError):
    """Error raised when dialogue timeline synthesis fails."""


class DialogueSynthesizer(Protocol):
    """Lightweight protocol describing the TTS agent surface needed here."""

    def synthesize(
        self,
        text: str,
        *,
        voice_config: Mapping[str, Any],
        plan_id: str,
        step_id: str,
        run_id: str,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        ...


VoiceResolver = Callable[[Optional[str]], Mapping[str, Any]]


@dataclass(frozen=True)
class DialogueLineSpec:
    index: int
    entry_type: Literal["dialogue", "silence"]
    start_time_sec: float
    duration_sec: float
    character_id: Optional[str]
    text: Optional[str]


@dataclass(frozen=True)
class DialogueLineArtifact:
    spec: DialogueLineSpec
    local_path: Path
    artifact_uri: Optional[str]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DialogueTimelineBuild:
    line_entries: List[Dict[str, Any]]
    line_paths: List[Path]
    summary_path: Path
    summary_payload: Dict[str, Any]
    timeline_audio_path: Path
    total_duration_s: float
    sample_rate: int
    channels: int
    sample_width: int
    timeline_offsets: Dict[int, Dict[str, Any]]


@dataclass(frozen=True)
class _TimelineSegment:
    entry_index: int
    kind: Literal["dialogue", "silence"]
    target_duration: float
    path: Optional[Path]


def build_dialogue_timeline(
    plan: MoviePlan,
    *,
    plan_id: str,
    run_id: str,
    output_dir: Path,
    synthesizer: DialogueSynthesizer,
    voice_resolver: VoiceResolver,
    timeline_subdir: str = "audio/timeline",
    timeline_audio_filename: str = "tts_timeline.wav",
    summary_filename: str = "dialogue_timeline_audio.json",
) -> DialogueTimelineBuild:
    """Synthesize dialogue audio per timeline entry and stitch a single WAV."""

    timeline_dir = output_dir / timeline_subdir
    timeline_dir.mkdir(parents=True, exist_ok=True)
    timeline_audio_path = timeline_dir / timeline_audio_filename
    summary_path = timeline_dir / summary_filename

    line_entries: List[Dict[str, Any]] = []
    line_paths: List[Path] = []
    segments: List[_TimelineSegment] = []

    for index, entry in enumerate(plan.dialogue_timeline):
        entry_type = getattr(entry, "type", "dialogue")
        start_time = _positive_float(getattr(entry, "start_time_sec", 0.0))
        duration = _positive_float(getattr(entry, "duration_sec", 0.0))
        base_payload: Dict[str, Any] = {
            "index": index,
            "type": entry_type,
            "start_time_sec": start_time,
            "duration_sec": duration,
            "character_id": getattr(entry, "character_id", None),
        }
        if entry_type == "silence":
            segments.append(
                _TimelineSegment(
                    entry_index=index,
                    kind="silence",
                    target_duration=duration,
                    path=None,
                )
            )
            line_entries.append({**base_payload, "text": None, "artifact_uri": None, "local_path": None})
            continue

        text = getattr(entry, "text", "")
        if not text or not text.strip():
            raise DialogueTimelineError(f"Dialogue timeline entry {index} must include text")

        voice_config = voice_resolver(getattr(entry, "character_id", None))
        step_label = f"dialogue_timeline:{index:04d}"
        artifact = synthesizer.synthesize(
            text,
            voice_config=voice_config,
            plan_id=plan_id,
            step_id=step_label,
            run_id=run_id,
            output_dir=timeline_dir,
        )

        metadata = dict(artifact.get("metadata") or {})
        source_path = metadata.get("source_path")
        local_path = Path(source_path) if source_path else timeline_dir / f"timeline_{index:04d}.wav"
        if not local_path.exists():
            local_path.write_bytes(b"")
        line_paths.append(local_path)

        raw_duration_hint = _positive_float(metadata.get("duration_s"))
        target_duration = duration if duration > 0 else raw_duration_hint
        segments.append(
            _TimelineSegment(
                entry_index=index,
                kind="dialogue",
                target_duration=target_duration,
                path=local_path,
            )
        )

        entry_payload: Dict[str, Any] = {
            **base_payload,
            "text": text,
            "artifact_uri": artifact.get("uri"),
            "local_path": local_path.as_posix(),
            "voice_id": metadata.get("voice_id"),
            "provider_id": metadata.get("provider_id"),
            "duration_audio_s": metadata.get("duration_s"),
            "sample_rate": metadata.get("sample_rate"),
            "bit_depth": metadata.get("bit_depth"),
            "watermarked": metadata.get("watermarked"),
        }
        voice_meta = metadata.get("voice_metadata")
        if isinstance(voice_meta, Mapping):
            entry_payload["voice_metadata"] = dict(voice_meta)
        adapter_meta = metadata.get("adapter_metadata")
        if isinstance(adapter_meta, Mapping):
            entry_payload["adapter_metadata"] = dict(adapter_meta)
        line_entries.append(entry_payload)

    if not segments:
        raise DialogueTimelineError("dialogue_timeline must contain at least one entry to synthesize")

    total_duration, sample_rate, sample_width, channels, offsets = _stitch_timeline_audio(segments, timeline_audio_path)

    for entry in line_entries:
        offset_meta = offsets.get(entry["index"])
        if offset_meta:
            entry["start_time_actual_s"] = offset_meta["start_time_s"]
            entry["end_time_actual_s"] = offset_meta["end_time_s"]
            entry["duration_actual_s"] = offset_meta["written_duration_s"]
            entry["duration_audio_raw_s"] = offset_meta.get("source_duration_s")
            entry["timeline_padding_s"] = offset_meta.get("padding_applied_s")
            entry["timeline_trimmed_s"] = offset_meta.get("trimmed_s")
        else:
            entry["start_time_actual_s"] = entry.get("start_time_sec", 0.0)
            planned_duration = entry.get("duration_sec") or 0.0
            entry["end_time_actual_s"] = entry["start_time_actual_s"] + planned_duration
            entry["duration_actual_s"] = planned_duration
            entry["duration_audio_raw_s"] = None
            entry["timeline_padding_s"] = 0.0
            entry["timeline_trimmed_s"] = 0.0

    summary_payload = {
        "plan_id": plan_id,
        "run_id": run_id,
        "entry_count": len(line_entries),
        "lines": line_entries,
        "timeline_audio": {
            "path": timeline_audio_path.as_posix(),
            "duration_s": total_duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
        },
        "timeline_offsets": offsets,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return DialogueTimelineBuild(
        line_entries=line_entries,
        line_paths=line_paths,
        summary_path=summary_path,
        summary_payload=summary_payload,
        timeline_audio_path=timeline_audio_path,
        total_duration_s=total_duration,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        timeline_offsets=offsets,
    )


def _positive_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return 0.0


def _stitch_timeline_audio(
    segments: Sequence[_TimelineSegment],
    timeline_path: Path,
) -> tuple[float, int, int, int, Dict[int, Dict[str, Any]]]:
    sample_rate: Optional[int] = None
    sample_width: Optional[int] = None
    channels: Optional[int] = None
    total_duration = 0.0
    offsets: Dict[int, Dict[str, float]] = {}
    timeline_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_writer_defaults(writer: wave.Wave_write) -> None:
        nonlocal sample_rate, sample_width, channels
        if sample_rate is None or sample_width is None or channels is None:
            sample_rate, sample_width, channels = 22050, 2, 1
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)

    with wave.open(str(timeline_path), "wb") as writer:
        for segment in segments:
            start_time = total_duration
            if segment.kind == "dialogue":
                if segment.path is None or not segment.path.exists():
                    raise DialogueTimelineError("dialogue segment missing audio payload")
                with wave.open(str(segment.path), "rb") as reader:
                    sr = reader.getframerate()
                    sw = reader.getsampwidth()
                    ch = reader.getnchannels()
                    frame_count = reader.getnframes()
                    data = reader.readframes(frame_count)
                if sample_rate is None:
                    sample_rate, sample_width, channels = sr, sw, ch
                    writer.setnchannels(channels)
                    writer.setsampwidth(sample_width)
                    writer.setframerate(sample_rate)
                elif sr != sample_rate or sw != sample_width or ch != channels:
                    raise DialogueTimelineError("dialogue audio segments must share sample parameters")
                if sample_rate is None or sample_width is None or channels is None:
                    raise DialogueTimelineError("audio parameters unavailable for dialogue segment")
                bytes_per_frame = sample_width * channels
                actual_duration = frame_count / sample_rate if sample_rate else 0.0
                target_duration = segment.target_duration if segment.target_duration > 0 else actual_duration
                if target_duration <= 0:
                    target_duration = actual_duration or 1.0 / sample_rate
                desired_frames = max(1, int(round(target_duration * sample_rate)))
                if desired_frames <= frame_count:
                    writer.writeframes(data[: desired_frames * bytes_per_frame])
                else:
                    writer.writeframes(data)
                    missing_frames = desired_frames - frame_count
                    writer.writeframes(b"\x00" * bytes_per_frame * missing_frames)
                written_duration = desired_frames / sample_rate
                padding = max(0.0, written_duration - actual_duration)
                trimmed = max(0.0, actual_duration - written_duration)
                offsets[segment.entry_index] = {
                    "kind": segment.kind,
                    "start_time_s": start_time,
                    "end_time_s": start_time + written_duration,
                    "written_duration_s": written_duration,
                    "target_duration_s": target_duration,
                    "source_duration_s": actual_duration,
                    "padding_applied_s": padding,
                    "trimmed_s": trimmed,
                }
                total_duration += written_duration
            else:
                _ensure_writer_defaults(writer)
                if sample_rate is None or sample_width is None or channels is None:
                    raise DialogueTimelineError("audio parameters unavailable for silence segment")
                bytes_per_frame = sample_width * channels
                target_duration = max(0.0, segment.target_duration)
                desired_frames = int(round(target_duration * sample_rate))
                if desired_frames > 0:
                    silence_frame = b"\x00" * bytes_per_frame
                    writer.writeframes(silence_frame * desired_frames)
                written_duration = desired_frames / sample_rate if sample_rate and desired_frames > 0 else 0.0
                offsets[segment.entry_index] = {
                    "kind": "silence",
                    "start_time_s": start_time,
                    "end_time_s": start_time + written_duration,
                    "written_duration_s": written_duration,
                    "target_duration_s": target_duration,
                    "source_duration_s": 0.0,
                    "padding_applied_s": 0.0,
                    "trimmed_s": 0.0,
                }
                total_duration += written_duration

    if sample_rate is None or sample_width is None or channels is None:
        raise DialogueTimelineError("Failed to synthesize dialogue timeline audio")
    return total_duration, sample_rate, sample_width, channels, offsets
