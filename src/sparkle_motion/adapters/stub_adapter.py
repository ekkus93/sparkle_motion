"""Deterministic stub adapter for smoke and unit tests."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Minimal valid 1x1 PNG (base64) to use as deterministic image bytes.
_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class StubAssetRef:
    """Describe a generated asset for smoke tests."""

    id: str
    data: bytes
    meta: Dict[str, Any]


@dataclass(frozen=True)
class AssemblyResult:
    """Result returned by :meth:`StubAdapter.assemble`."""

    path: Path
    audio_included: bool
    duration: Optional[float] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None

    def __iter__(self):
        """Allow tuple-unpacking ``path, audio = AssemblyResult(...)``."""

        yield self.path
        yield self.audio_included


def _write_minimal_mp4(path: Path) -> None:
    """Write a tiny MP4-like structure so signature checks succeed."""

    ftyp = (
        (24).to_bytes(4, "big")
        + b"ftyp"
        + b"isom"
        + (0).to_bytes(4, "big")
        + b"isom"
        + b"mp41"
    )
    payload = b"\x00\x00\x00\x00"
    mdat = (8 + len(payload)).to_bytes(4, "big") + b"mdat" + payload
    path.write_bytes(ftyp + mdat)


def _probe_media(path: Path) -> tuple[Optional[float], Optional[str], Optional[str]]:
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None, None, None

    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None, None, None

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, None, None

    duration_str = info.get("format", {}).get("duration")
    duration = float(duration_str) if duration_str else None
    vcodec = None
    acodec = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and vcodec is None:
            vcodec = stream.get("codec_name")
        if stream.get("codec_type") == "audio" and acodec is None:
            acodec = stream.get("codec_name")
    return duration, vcodec, acodec


def _pick_audio_encoder(ffmpeg_bin: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError:
        return None

    encoders = proc.stdout
    for candidate in ("aac", "libmp3lame", "libopus"):
        if candidate in encoders:
            return candidate
    return None


def _normalize_sequence(images: Sequence[bytes]) -> List[bytes]:
    return [bytes(img) for img in images]


class StubAdapter:
    """Adapter shim returning deterministic, lightweight artifacts."""

    def generate_images(self, prompt: str, count: int = 1) -> List[StubAssetRef]:
        """Return ``count`` deterministic PNGs with metadata."""

        count = max(0, int(count))
        return [
            StubAssetRef(id=str(uuid.uuid4()), data=_ONE_PIXEL_PNG, meta={"prompt": prompt})
            for _ in range(count)
        ]

    def tts(self, text: str) -> bytes:
        """Return UTF-8 bytes for the supplied text."""

        return text.encode("utf-8")

    def lipsync(self, audio: bytes, video_ref: Any) -> Dict[str, Any]:
        """Return a small metadata dict describing a lipsync invocation."""

        return {"status": "ok", "video_ref": video_ref, "audio_len": len(audio)}

    @staticmethod
    def _want_audio(add_audio: Optional[bool], detected_codec: Optional[str]) -> bool:
        if add_audio is True:
            return detected_codec is not None
        if add_audio is False:
            return False
        return detected_codec is not None

    def assemble(
        self,
        shots: List[Dict[str, Any]],
        out_path: Optional[Path] = None,
        add_audio: Optional[bool] = None,
    ) -> AssemblyResult:
        """Create a tiny MP4 placeholder using ``ffmpeg`` when available."""

        if out_path is None:
            out_path = Path(tempfile.gettempdir()) / f"stub_assemble_{uuid.uuid4().hex}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            detected = _pick_audio_encoder(ffmpeg_bin)
            want_audio = self._want_audio(add_audio, detected)
            with tempfile.TemporaryDirectory() as td:
                img_path = Path(td) / "frame.png"
                img_path.write_bytes(_ONE_PIXEL_PNG)
                cmd = [ffmpeg_bin, "-y", "-loop", "1", "-i", str(img_path)]
                if want_audio and detected:
                    cmd += [
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-c:a",
                        detected,
                        "-shortest",
                    ]
                cmd += [
                    "-c:v",
                    "mpeg4",
                    "-t",
                    "0.6",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_path),
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    duration, vcodec, acodec = _probe_media(out_path)
                    return AssemblyResult(
                        path=out_path,
                        audio_included=bool(want_audio and detected),
                        duration=duration,
                        video_codec=vcodec,
                        audio_codec=acodec,
                    )
                except subprocess.CalledProcessError:
                    if add_audio is True:
                        raise RuntimeError("ffmpeg failed to encode requested audio/video") from None
                except Exception:
                    pass

        _write_minimal_mp4(out_path)
        return AssemblyResult(path=out_path, audio_included=False)

    def render_sequence(
        self,
        images: Sequence[bytes],
        out_path: Optional[Path] = None,
        fps: int = 1,
        add_audio: Optional[bool] = None,
    ) -> AssemblyResult:
        """Render a sequence of images into a short MP4."""

        frames = _normalize_sequence(images)
        if not frames:
            raise ValueError("images list must be non-empty")

        if out_path is None:
            out_path = Path(tempfile.gettempdir()) / f"stub_render_{uuid.uuid4().hex}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return self.assemble(shots=[{"frames": len(frames)}], out_path=out_path, add_audio=add_audio)

        detected = _pick_audio_encoder(ffmpeg_bin)
        want_audio = self._want_audio(add_audio, detected)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for idx, img in enumerate(frames):
                (td_path / f"frame_{idx:03d}.png").write_bytes(img)

            cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(max(1, int(fps))),
                "-i",
                str(td_path / "frame_%03d.png"),
            ]
            if want_audio and detected:
                cmd += [
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-c:a",
                    detected,
                    "-shortest",
                ]
            cmd += ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]

            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                duration, vcodec, acodec = _probe_media(out_path)
                return AssemblyResult(
                    path=out_path,
                    audio_included=bool(want_audio and detected),
                    duration=duration,
                    video_codec=vcodec,
                    audio_codec=acodec,
                )
            except subprocess.CalledProcessError:
                if add_audio is True:
                    raise RuntimeError("ffmpeg failed to encode requested audio/video for sequence") from None
            except Exception:
                pass

        return self.assemble(shots=[{"frames": len(frames)}], out_path=out_path, add_audio=False)


def get_stub_adapter() -> StubAdapter:
    """Convenience factory used by tests and examples."""

    return StubAdapter()


__all__ = ["StubAdapter", "get_stub_adapter", "StubAssetRef", "AssemblyResult"]
