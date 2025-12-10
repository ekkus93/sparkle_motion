from __future__ import annotations

from array import array
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence
import wave

import pytest

from sparkle_motion.dialogue_timeline import (
    DialogueTimelineBuilder,
    DialogueSynthesizer,
    DialogueTimelineBuild,
    DialogueTimelineError,
    build_dialogue_timeline,
)
from sparkle_motion.schemas import (
    BaseImageSpec,
    CharacterSpec,
    DialogueLine,
    DialogueTimelineDialogue,
    DialogueTimelineSilence,
    MoviePlan,
    RenderProfile,
    RenderProfileVideo,
    ShotSpec,
)


class FakeSynthesizer(DialogueSynthesizer):
    """Deterministic synthesizer that writes tiny WAV files per call."""

    def __init__(self, durations: Sequence[float], sample_rate: int = 22050) -> None:
        self._durations = list(durations)
        self.sample_rate = sample_rate
        self.calls: List[Dict[str, Any]] = []

    def _next_duration(self) -> float:
        if not self._durations:
            raise AssertionError("No durations remaining for synthesizer call")
        return self._durations.pop(0)

    def synthesize(
        self,
        text: str,
        *,
        voice_config: Optional[Dict[str, Any]] = None,
        plan_id: str,
        step_id: str,
        run_id: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        duration = self._next_duration()
        local_path = output_dir / f"{step_id.replace(':', '_')}.wav"
        _write_silence_wav(local_path, duration, sample_rate=self.sample_rate)
        metadata = {
            "source_path": str(local_path),
            "voice_id": (voice_config or {}).get("voice_id"),
            "provider_id": "fake",
            "duration_s": duration,
            "sample_rate": self.sample_rate,
            "bit_depth": 16,
            "watermarked": False,
        }
        self.calls.append({
            "text": text,
            "voice_config": voice_config,
            "plan_id": plan_id,
            "step_id": step_id,
            "run_id": run_id,
        })
        return {"uri": local_path.as_uri(), "metadata": metadata}


def _write_silence_wav(path: Path, duration_s: float, *, sample_rate: int) -> None:
    frame_count = max(1, int(round(duration_s * sample_rate)))
    frames = array("h", [0] * frame_count)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames.tobytes())


def _make_plan(entries: Sequence[tuple[Literal["dialogue", "silence"], float, Optional[str], Optional[str]]]) -> MoviePlan:
    base_images = [BaseImageSpec(id=f"frame_{idx:03d}", prompt=f"Frame {idx}") for idx in range(len(entries) + 1)]
    shots: List[ShotSpec] = []
    timeline: List[DialogueTimelineDialogue | DialogueTimelineSilence] = []
    characters: Dict[str, CharacterSpec] = {}
    current = 0.0
    for idx, (entry_type, duration, text, character_id) in enumerate(entries):
        char_id = character_id or "character"
        if entry_type == "dialogue":
            characters.setdefault(char_id, CharacterSpec(id=char_id, name=char_id.title()))
            dialogue_lines = [DialogueLine(character_id=char_id, text=text or "")] \
                if text is not None else [DialogueLine(character_id=char_id, text="")]
            timeline.append(
                DialogueTimelineDialogue(
                    character_id=char_id,
                    text=text or "",
                    start_time_sec=current,
                    duration_sec=duration,
                )
            )
        else:
            dialogue_lines = []
            timeline.append(
                DialogueTimelineSilence(
                    start_time_sec=current,
                    duration_sec=duration,
                )
            )
        shots.append(
            ShotSpec(
                id=f"shot_{idx:03d}",
                visual_description=f"Shot {idx}",
                duration_sec=duration,
                dialogue=dialogue_lines,
                start_base_image_id=base_images[idx].id,
                end_base_image_id=base_images[idx + 1].id,
            )
        )
        current += duration
    if not characters:
        characters["character"] = CharacterSpec(id="character", name="Character")
    return MoviePlan(
        title="Dialogue Timeline Test",
        metadata={"plan_id": "plan-timeline"},
        characters=list(characters.values()),
        base_images=base_images,
        shots=shots,
        dialogue_timeline=timeline,
        render_profile=RenderProfile(video=RenderProfileVideo(model_id="wan-fixture"), metadata={}),
    )


def _resolver(character_id: Optional[str]) -> Dict[str, Any]:
    return {"voice_id": (character_id or "character")}


def _build(entries: Sequence[tuple[Literal["dialogue", "silence"], float, Optional[str], Optional[str]]], *,
           tmp_path: Path, durations: Sequence[float]) -> DialogueTimelineBuild:
    plan = _make_plan(entries)
    synth = FakeSynthesizer(durations)
    return build_dialogue_timeline(
        plan,
        plan_id="plan-id",
        run_id="run-id",
        output_dir=tmp_path,
        synthesizer=synth,
        voice_resolver=_resolver,
    )


def test_build_dialogue_timeline_padding_and_trim(tmp_path: Path) -> None:
    entries = [
        ("dialogue", 0.25, "First line", "hero"),
        ("dialogue", 0.75, "Second line", "hero"),
    ]
    result = _build(entries, tmp_path=tmp_path, durations=[0.4, 0.5])
    assert result.timeline_audio_path.exists()
    assert result.summary_path.exists()
    offsets = result.timeline_offsets
    assert pytest.approx(result.total_duration_s, rel=1e-3) == 1.0
    first = offsets[0]
    second = offsets[1]
    assert pytest.approx(first["written_duration_s"], rel=1e-3) == 0.25
    assert pytest.approx(first["trimmed_s"], rel=1e-3) == pytest.approx(0.15, rel=1e-3)
    assert pytest.approx(first["padding_applied_s"], rel=1e-6) == 0.0
    assert pytest.approx(second["written_duration_s"], rel=1e-3) == 0.75
    assert pytest.approx(second["padding_applied_s"], rel=1e-3) == 0.25
    assert pytest.approx(second["trimmed_s"], rel=1e-6) == 0.0
    line0 = result.line_entries[0]
    line1 = result.line_entries[1]
    assert pytest.approx(line0["duration_audio_raw_s"], rel=1e-3) == 0.4
    assert pytest.approx(line1["duration_audio_raw_s"], rel=1e-3) == 0.5


def test_build_dialogue_timeline_includes_silence(tmp_path: Path) -> None:
    entries = [
        ("dialogue", 0.6, "Hello", "hero"),
        ("silence", 0.4, None, None),
    ]
    result = _build(entries, tmp_path=tmp_path, durations=[0.3])
    assert len(result.line_entries) == 2
    assert len(result.line_paths) == 1  # silence entries should not produce files
    silence_entry = result.line_entries[1]
    assert silence_entry["text"] is None
    assert silence_entry["artifact_uri"] is None
    offset = result.timeline_offsets[1]
    assert offset["kind"] == "silence"
    assert pytest.approx(offset["target_duration_s"], rel=1e-3) == 0.4
    assert pytest.approx(result.total_duration_s, rel=1e-3) == 1.0


def test_build_dialogue_timeline_requires_text(tmp_path: Path) -> None:
    entries = [("dialogue", 0.5, "   ", "hero")]
    plan = _make_plan(entries)
    synth = FakeSynthesizer([0.3])
    with pytest.raises(DialogueTimelineError):
        build_dialogue_timeline(
            plan,
            plan_id="plan",
            run_id="run",
            output_dir=tmp_path,
            synthesizer=synth,
            voice_resolver=_resolver,
        )

    def test_dialogue_timeline_builder_uses_dependencies(tmp_path: Path) -> None:
        entries = [
            ("dialogue", 0.5, "Builder line", "hero"),
        ]
        plan = _make_plan(entries)
        synth = FakeSynthesizer([0.5])
        resolved_ids: List[Optional[str]] = []

        def _recording_resolver(character_id: Optional[str]) -> Dict[str, Any]:
            resolved_ids.append(character_id)
            return {"voice_id": f"voice-{character_id or 'default'}"}

        builder = DialogueTimelineBuilder(
            synthesizer=synth,
            voice_resolver=_recording_resolver,
            timeline_subdir="builder",
            timeline_audio_filename="custom_timeline.wav",
            summary_filename="custom_summary.json",
        )

        result = builder.build(
            plan,
            plan_id="plan-builder",
            run_id="run-builder",
            output_dir=tmp_path,
        )

        assert result.summary_path.name == "custom_summary.json"
        assert result.timeline_audio_path.name == "custom_timeline.wav"
        assert result.summary_path.parent == tmp_path / "builder"
        assert synth.calls and synth.calls[0]["plan_id"] == "plan-builder"
        assert resolved_ids == [plan.dialogue_timeline[0].character_id]
