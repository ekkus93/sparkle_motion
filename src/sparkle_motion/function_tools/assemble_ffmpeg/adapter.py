from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

LOG = logging.getLogger(__name__)
TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ClipSpec:
    uri: Path
    start_s: float = 0.0
    end_s: float | None = None
    metadata: Mapping[str, Any] | None = None
    transition: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AudioSpec:
    uri: Path
    start_s: float = 0.0
    end_s: float | None = None
    metadata: Mapping[str, Any] | None = None
    gain_db: float | None = None


@dataclass(frozen=True)
class AssemblyResult:
    path: Path
    metadata: MutableMapping[str, Any]
    engine: str
    duration_s: float
    logs_path: Path | None = None


@dataclass(frozen=True)
class SubprocessResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    attempts: int


class AssemblyError(RuntimeError):
    def __init__(self, message: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


class CommandError(RuntimeError):
    def __init__(self, message: str, *, stdout: str, stderr: str, exit_code: int, attempts: int) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.attempts = attempts


class CommandTimeoutError(CommandError):
    pass


def should_use_real_engine(env: Mapping[str, str] | None = None) -> bool:
    data = env or os.environ
    if _is_truthy(data.get("ADK_USE_FIXTURE", "0")):
        return False
    if _is_truthy(data.get("ASSEMBLE_FFMPEG_FIXTURE_ONLY", "0")):
        return False
    for key in ("SMOKE_ASSEMBLE", "SMOKE_ADAPTERS"):
        if _is_truthy(data.get(key, "0")):
            return True
    return False


def assemble_movie(
    *,
    clips: Sequence[ClipSpec],
    audio: AudioSpec | None = None,
    plan_id: str | None = None,
    options: Mapping[str, Any] | None = None,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> AssemblyResult:
    if not clips:
        raise AssemblyError("At least one clip is required", metadata={"reason": "no_clips"})
    for clip in clips:
        if not clip.uri.exists():
            raise AssemblyError("Clip path does not exist", metadata={"clip": str(clip.uri)})
    if audio and not audio.uri.exists():
        raise AssemblyError("Audio path does not exist", metadata={"audio": str(audio.uri)})

    target_dir = output_dir or _default_output_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    opts = dict(options or {})

    if opts.get("fixture_only"):
        return _assemble_fixture(
            clips=clips,
            audio=audio,
            out_dir=target_dir,
            plan_id=plan_id,
            options=opts,
            seed=seed,
        )

    if should_use_real_engine() and not _is_truthy(os.environ.get("ASSEMBLE_FFMPEG_FIXTURE_ONLY", "0")):
        try:
            return _assemble_with_ffmpeg(
                clips=clips,
                audio=audio,
                out_dir=target_dir,
                plan_id=plan_id,
                options=opts,
            )
        except (AssemblyError, CommandError) as exc:
            LOG.warning("ffmpeg assembly failed, falling back to fixture: %s", exc)

    return _assemble_fixture(
        clips=clips,
        audio=audio,
        out_dir=target_dir,
        plan_id=plan_id,
        options=opts,
        seed=seed,
    )


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_s: float | None = 600.0,
    retries: int = 0,
    env: Mapping[str, str] | None = None,
    sleep_between_retries_s: float = 0.2,
) -> SubprocessResult:
    attempts = 0
    last_error: CommandError | None = None
    while attempts <= retries:
        attempts += 1
        start = time.monotonic()
        proc = subprocess.Popen(  # noqa: S603
            list(cmd),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(env or os.environ),
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            duration = time.monotonic() - start
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            stdout, stderr = proc.communicate()
            duration = time.monotonic() - start
            last_error = CommandTimeoutError(
                f"Command timed out after {timeout_s}s",
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode if proc.returncode is not None else -9,
                attempts=attempts,
            )
        else:
            if proc.returncode == 0:
                return SubprocessResult(exit_code=0, stdout=stdout, stderr=stderr, duration_s=duration, attempts=attempts)
            last_error = CommandError(
                f"Command exited with {proc.returncode}",
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or -1,
                attempts=attempts,
            )
        if attempts <= retries:
            time.sleep(sleep_between_retries_s)
    assert last_error is not None
    raise last_error


def _assemble_fixture(
    *,
    clips: Sequence[ClipSpec],
    audio: AudioSpec | None,
    out_dir: Path,
    plan_id: str | None,
    options: Mapping[str, Any],
    seed: int | None,
) -> AssemblyResult:
    digest = _calculate_digest(clips=clips, audio=audio, plan_id=plan_id, options=options, seed=seed)
    filename = f"{_slug(plan_id or 'assemble')}-{digest}.mp4"
    target = out_dir / filename
    payload = _fixture_bytes()
    target.write_bytes(payload)
    metadata: MutableMapping[str, Any] = {
        "engine": "fixture",
        "engine_version": "assemble_ffmpeg_fixture_v1",
        "clip_count": len(clips),
        "audio_attached": audio is not None,
        "digest": digest,
        "options": dict(options),
    }
    if plan_id:
        metadata["plan_id"] = plan_id
    return AssemblyResult(path=target, metadata=metadata, engine="fixture", duration_s=0.0)


def _assemble_with_ffmpeg(
    *,
    clips: Sequence[ClipSpec],
    audio: AudioSpec | None,
    out_dir: Path,
    plan_id: str | None,
    options: Mapping[str, Any],
) -> AssemblyResult:
    ffmpeg_bin = os.environ.get("FFMPEG_PATH", "ffmpeg")
    timeout_s = float(options.get("timeout_s", 120.0))
    retries = int(options.get("retries", 0))
    video_codec = str(options.get("video_codec", "libx264"))
    audio_codec = str(options.get("audio_codec", "aac"))
    pix_fmt = str(options.get("pix_fmt", "yuv420p"))
    crf = str(options.get("crf", "18"))
    preset = str(options.get("preset", "veryslow"))
    audio_bitrate = str(options.get("audio_bitrate", "192k"))

    filename = f"{_slug(plan_id or 'assemble')}-{int(time.time())}.mp4"
    target = out_dir / filename

    with tempfile.TemporaryDirectory() as tmp_dir:
        list_file = Path(tmp_dir) / "concat.txt"
        list_file.write_text(_build_concat_manifest(clips), encoding="utf-8")
        cmd: list[str] = [
            ffmpeg_bin,
            "-y",
            "-safe",
            "0",
            "-f",
            "concat",
            "-i",
            str(list_file),
        ]
        if audio is not None:
            cmd += ["-i", str(audio.uri)]
        cmd += [
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            pix_fmt,
        ]
        if audio is not None:
            cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate]
        cmd += ["-movflags", "+faststart", str(target)]

        result = run_command(cmd, cwd=out_dir, timeout_s=timeout_s, retries=retries)
        if result.exit_code != 0:
            raise AssemblyError(
                "ffmpeg exited with non-zero status",
                metadata={"stderr": result.stderr, "stdout": result.stdout},
            )

    metadata: MutableMapping[str, Any] = {
        "engine": "ffmpeg",
        "engine_version": _probe_ffmpeg_version(ffmpeg_bin),
        "clip_count": len(clips),
        "audio_attached": audio is not None,
        "ffmpeg_command": cmd,
        "stdout_tail": result.stdout[-1024:],
        "stderr_tail": result.stderr[-1024:],
        "options": dict(options),
    }
    if plan_id:
        metadata["plan_id"] = plan_id
    return AssemblyResult(path=target, metadata=metadata, engine="ffmpeg", duration_s=result.duration_s)


def _default_output_dir() -> Path:
    root = Path(os.environ.get("ARTIFACTS_DIR", Path.cwd() / "artifacts"))
    return root / "assemble_ffmpeg"


def _fixture_bytes() -> bytes:
    return (
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
        b"\x00\x00\x00\x08free\x00\x00\x00\x1cmdat" + b"\x00" * 64
    )


def _calculate_digest(
    *,
    clips: Sequence[ClipSpec],
    audio: AudioSpec | None,
    plan_id: str | None,
    options: Mapping[str, Any],
    seed: int | None,
) -> str:
    payload = {
        "clips": [str(clip.uri) for clip in clips],
        "audio": str(audio.uri) if audio else None,
        "plan_id": plan_id,
        "options": dict(options),
        "seed": seed,
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:10]


def _build_concat_manifest(clips: Sequence[ClipSpec]) -> str:
    lines: list[str] = []
    for clip in clips:
        lines.append(f"file '{clip.uri}'")
    return "\n".join(lines)


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]", "-", value)
    value = re.sub(r"-+", "-", value)
    return value[:48] or "assemble"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in TRUTHY


def _probe_ffmpeg_version(ffmpeg_bin: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.splitlines()[0].strip()
    except Exception:
        pass
    return "unknown"


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    try:
        if hasattr(os, "killpg") and getattr(proc, "pid", None):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # type: ignore[arg-type]
        else:
            proc.kill()
    except Exception:
        proc.kill()


__all__ = [
    "ClipSpec",
    "AudioSpec",
    "AssemblyResult",
    "SubprocessResult",
    "AssemblyError",
    "CommandError",
    "CommandTimeoutError",
    "assemble_movie",
    "run_command",
    "should_use_real_engine",
]
