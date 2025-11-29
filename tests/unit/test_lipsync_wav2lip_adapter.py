from __future__ import annotations

from pathlib import Path

from sparkle_motion.function_tools.lipsync_wav2lip import adapter


def test_run_wav2lip_fixture_produces_file(tmp_path: Path) -> None:
    face = tmp_path / "face.mp4"
    audio = tmp_path / "audio.wav"
    face.write_bytes(b"face-bytes")
    audio.write_bytes(b"audio-bytes")
    out_path = tmp_path / "out.mp4"

    result = adapter.run_wav2lip(face, audio, out_path)

    assert result.path == out_path
    assert out_path.exists()
    assert result.metadata["engine"] == "wav2lip_fixture"
    assert "artifact_digest" in result.metadata
    assert result.logs["stdout"].startswith("fixture")


def test_build_subprocess_command_includes_flags(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wav2lip.pth"
    checkpoint.write_bytes(b"ckpt")
    face = tmp_path / "face.mp4"
    audio = tmp_path / "audio.wav"
    face.write_bytes(b"face")
    audio.write_bytes(b"audio")
    opts = {
        "pads": (0, 10, 0, 0),
        "resize_factor": 2,
        "crop": (10, 10, 50, 50),
        "nosmooth": True,
        "fps": 25,
        "face_det_checkpoint": str(checkpoint),
    }

    cmd = adapter.build_subprocess_command(
        python_bin="python",
        script_path="inference.py",
        checkpoint_path=checkpoint,
        face_path=face,
        audio_path=audio,
        out_path=tmp_path / "out.mp4",
        opts=opts,
    )

    assert "--pads" in cmd
    assert "--crop" in cmd
    assert "--nosmooth" in cmd
    assert "--fps" in cmd
    assert "--face_det_checkpoint" in cmd