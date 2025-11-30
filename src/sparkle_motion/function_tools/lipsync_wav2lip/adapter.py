from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from sparkle_motion.utils.env import fixture_mode_enabled

LOG = logging.getLogger(__name__)
TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_TIMEOUT_S = 900


class LipsyncError(RuntimeError):
    """Raised when the lipsync adapter cannot complete a request."""


class CommandError(LipsyncError):
    """Raised when a subprocess invocation fails."""

    def __init__(self, message: str, result: CommandResult) -> None:  # noqa: F821 - forward ref
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CommandResult:
    cmd: Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


@dataclass(frozen=True)
class LipsyncResult:
    path: Path
    metadata: MutableMapping[str, Any]
    logs: MutableMapping[str, str]
    duration_s: float


def run_wav2lip(
    face_video: Path | str,
    audio: Path | str,
    out_path: Path | str,
    *,
    opts: Mapping[str, Any] | None = None,
    force_fixture: Optional[bool] = None,
) -> LipsyncResult:
    """Lip-sync ``face_video`` to ``audio`` and write the merged clip."""

    face_path = _validated_path(face_video, "face_video")
    audio_path = _validated_path(audio, "audio")
    target_path = Path(out_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    options: MutableMapping[str, Any] = dict(opts or {})

    fixture_hint = options.get("fixture_only")
    if isinstance(fixture_hint, str):
        fixture_hint = fixture_hint.lower() in TRUTHY
    should_use_real = should_use_real_engine(force_fixture=force_fixture or fixture_hint)

    start = time.perf_counter()
    if should_use_real:
        try:
            result = _run_real_wav2lip(face_path, audio_path, target_path, options)
            result.metadata.setdefault("mode", "real")
            return result
        except Exception as exc:  # pragma: no cover - exercised in smoke envs
            if not bool(options.get("allow_fixture_fallback", True)):
                raise
            LOG.warning("real wav2lip execution failed; falling back to fixture", exc_info=exc)

    result = _run_fixture(face_path, audio_path, target_path, options)
    elapsed = time.perf_counter() - start
    result.metadata.setdefault("mode", "fixture")
    result.metadata.setdefault("duration_s", round(elapsed, 4))
    result.logs.setdefault("mode", "fixture")
    result.logs.setdefault("note", "fixture output; enable SMOKE_LIPSYNC for real inference")
    return result


def should_use_real_engine(*, force_fixture: Optional[bool] = None, env: Optional[Mapping[str, str]] = None) -> bool:
    data = env or os.environ
    if _is_truthy(data.get("LIPSYNC_WAV2LIP_FIXTURE_ONLY")):
        return False
    if fixture_mode_enabled(data, default=False):
        return False
    if force_fixture is True:
        return False
    flags = ("SMOKE_LIPSYNC", "SMOKE_ADAPTERS", "SMOKE_ADK")
    return any(_is_truthy(data.get(flag)) for flag in flags)


def build_subprocess_command(
    *,
    python_bin: str,
    script_path: str,
    checkpoint_path: Path,
    face_path: Path,
    audio_path: Path,
    out_path: Path,
    opts: Mapping[str, Any],
) -> list[str]:
    """Build the canonical Wav2Lip subprocess command."""

    cmd = [python_bin, script_path, "--checkpoint_path", str(checkpoint_path), "--face", str(face_path), "--audio", str(audio_path), "--outfile", str(out_path)]
    pads = opts.get("pads")
    if pads:
        pad_values = _coerce_int_tuple(pads, expected=4, field_name="pads")
        cmd.extend(["--pads", *[str(v) for v in pad_values]])
    resize_factor = opts.get("resize_factor")
    if resize_factor:
        cmd.extend(["--resize_factor", str(int(resize_factor))])
    crop = opts.get("crop")
    if crop:
        crop_values = _coerce_int_tuple(crop, expected=4, field_name="crop")
        cmd.extend(["--crop", *[str(v) for v in crop_values]])
    if opts.get("nosmooth"):
        cmd.append("--nosmooth")
    fps = opts.get("fps")
    if fps:
        cmd.extend(["--fps", str(fps)])
    face_det = opts.get("face_det_checkpoint")
    if face_det:
        cmd.extend(["--face_det_checkpoint", str(face_det)])
    return cmd


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    timeout_s: Optional[int] = None,
    retries: int = 0,
    env: Optional[Mapping[str, str]] = None,
) -> CommandResult:
    attempt = 0
    last_error: Optional[CommandError] = None
    timeout = timeout_s or DEFAULT_TIMEOUT_S
    while attempt <= retries:
        start = time.perf_counter()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=dict(env or os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            stdout, stderr = proc.communicate()
            duration = time.perf_counter() - start
            result = CommandResult(cmd=list(cmd), returncode=-1, stdout=stdout, stderr=stderr, duration_s=duration)
            raise CommandError(f"command timed out after {timeout}s", result)
        duration = time.perf_counter() - start
        result = CommandResult(cmd=list(cmd), returncode=proc.returncode, stdout=stdout, stderr=stderr, duration_s=duration)
        if proc.returncode == 0:
            return result
        last_error = CommandError(f"command exited with {proc.returncode}", result)
        attempt += 1
    assert last_error is not None
    raise last_error


def _run_real_wav2lip(
    face_path: Path,
    audio_path: Path,
    out_path: Path,
    opts: Mapping[str, Any],
) -> LipsyncResult:
    checkpoint = _resolve_checkpoint(opts)
    repo = _resolve_repo(opts)
    python_bin = str(opts.get("python_bin") or sys.executable)
    script_path = str(opts.get("script_path") or (repo / "inference.py" if repo else "inference.py"))
    cmd = build_subprocess_command(
        python_bin=python_bin,
        script_path=script_path,
        checkpoint_path=checkpoint,
        face_path=face_path,
        audio_path=audio_path,
        out_path=out_path,
        opts=opts,
    )
    cwd = repo if repo else None
    timeout = int(opts.get("timeout_s" or DEFAULT_TIMEOUT_S))
    retries = int(opts.get("retries" or 0))
    result = run_command(cmd, cwd=cwd, timeout_s=timeout, retries=retries)
    if not out_path.exists():
        raise LipsyncError("wav2lip execution completed without producing output")

    metadata: MutableMapping[str, Any] = {
        "engine": "wav2lip_subprocess",
        "checkpoint_path": str(checkpoint),
        "face_video": str(face_path),
        "audio": str(audio_path),
        "options": dict(opts),
    }
    logs: MutableMapping[str, str] = {
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "command": json.dumps(cmd),
    }
    return LipsyncResult(path=out_path, metadata=metadata, logs=logs, duration_s=round(result.duration_s, 4))


def _run_fixture(
    face_path: Path,
    audio_path: Path,
    out_path: Path,
    opts: Mapping[str, Any],
) -> LipsyncResult:
    seed = opts.get("fixture_seed")
    digest = _fixture_digest(face_path, audio_path, seed=seed)
    payload = _fixture_payload(digest)
    out_path.write_bytes(payload)
    metadata: MutableMapping[str, Any] = {
        "engine": "wav2lip_fixture",
        "artifact_digest": digest,
        "face_video": str(face_path),
        "audio": str(audio_path),
        "options": dict(opts),
    }
    logs: MutableMapping[str, str] = {
        "stdout": "fixture output",
        "stderr": "",
    }
    return LipsyncResult(path=out_path, metadata=metadata, logs=logs, duration_s=0.0)


def _fixture_payload(digest: str) -> bytes:
    rng = random.Random(int(digest[:8], 16))
    header = bytearray(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2\x00\x00\x00\x08free\x00\x00\x00\x1cmdat")
    payload = bytearray(rng.getrandbits(8) for _ in range(2048))
    size = len(header) + len(payload)
    header[0:4] = size.to_bytes(4, "big", signed=False)
    return bytes(header + payload)


def _fixture_digest(face_path: Path, audio_path: Path, *, seed: Any = None) -> str:
    payload = {
        "face": face_path.read_bytes()[:4096].hex(),
        "audio": audio_path.read_bytes()[:4096].hex(),
        "seed": seed,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _validated_path(value: Path | str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in TRUTHY


def _coerce_int_tuple(value: Any, *, expected: int, field_name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = value.strip().split()
    elif isinstance(value, Iterable):
        parts = list(value)
    else:
        raise ValueError(f"{field_name} must be an iterable of ints")
    if len(parts) != expected:
        raise ValueError(f"{field_name} must have {expected} elements")
    return tuple(int(p) for p in parts)


def _resolve_checkpoint(opts: Mapping[str, Any]) -> Path:
    checkpoint = opts.get("checkpoint_path") or os.environ.get("LIPSYNC_WAV2LIP_MODEL")
    if not checkpoint:
        raise LipsyncError("checkpoint_path must be provided via opts or LIPSYNC_WAV2LIP_MODEL")
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise LipsyncError(f"checkpoint not found: {path}")
    return path


def _resolve_repo(opts: Mapping[str, Any]) -> Optional[Path]:
    repo = opts.get("repo_path") or os.environ.get("WAV2LIP_REPO")
    if not repo:
        return None
    path = Path(repo).expanduser()
    if not path.exists():
        raise LipsyncError(f"repo_path not found: {path}")
    return path


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return