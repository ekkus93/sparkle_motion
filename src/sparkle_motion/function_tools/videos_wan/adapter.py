from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from sparkle_motion import gpu_utils

LOG = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"


class VideoRenderError(RuntimeError):
	"""Raised when the Wan adapter cannot render a clip."""


@dataclass(frozen=True)
class VideoRenderResult:
	path: Path
	metadata: MutableMapping[str, Any]
	engine: str
	duration_s: float
	frame_count: int
	fps: int


def render_clip(
	*,
	prompt: str,
	num_frames: int,
	fps: int,
	width: int,
	height: int,
	seed: Optional[int] = None,
	plan_id: Optional[str] = None,
	chunk_index: Optional[int] = None,
	chunk_count: Optional[int] = None,
	metadata: Optional[Mapping[str, Any]] = None,
	options: Optional[Mapping[str, Any]] = None,
	start_frame: Optional[bytes] = None,
	end_frame: Optional[bytes] = None,
	output_dir: Optional[Path] = None,
	force_fixture: Optional[bool] = None,
) -> VideoRenderResult:
	"""Render a video clip via Wan2.1 or the deterministic fixture."""

	_validate_dimensions(num_frames=num_frames, fps=fps, width=width, height=height)
	opts: MutableMapping[str, Any] = dict(options or {})
	extra_metadata: MutableMapping[str, Any] = dict(metadata or {})
	target_dir = (output_dir or _default_output_dir()).resolve()
	target_dir.mkdir(parents=True, exist_ok=True)
	chunk_label = chunk_index if chunk_index is not None else 0
	derived_seed = _derive_seed(seed, prompt=prompt, chunk_index=chunk_label)

	fixture_flag = force_fixture or opts.get("fixture_only")
	if isinstance(fixture_flag, str):
		fixture_flag = _is_truthy(fixture_flag)
	engine_pref = should_use_real_engine(force_fixture=bool(fixture_flag))
	if engine_pref:
		try:
			return _render_with_wan(
				prompt=prompt,
				num_frames=num_frames,
				fps=fps,
				width=width,
				height=height,
				seed=derived_seed,
				plan_id=plan_id,
				chunk_index=chunk_index,
				chunk_count=chunk_count,
				metadata=extra_metadata,
				options=opts,
				start_frame=start_frame,
				end_frame=end_frame,
				output_dir=target_dir,
			)
		except Exception as exc:  # pragma: no cover - real path exercised in smoke
			LOG.warning("Wan pipeline failed; falling back to fixture", exc_info=exc)

	return _render_fixture(
		prompt=prompt,
		num_frames=num_frames,
		fps=fps,
		width=width,
		height=height,
		seed=derived_seed,
		plan_id=plan_id,
		chunk_index=chunk_index,
		chunk_count=chunk_count,
		metadata=extra_metadata,
		options=opts,
		start_frame=start_frame,
		end_frame=end_frame,
		output_dir=target_dir,
	)


def should_use_real_engine(
	*,
	force_fixture: Optional[bool] = None,
	env: Optional[Mapping[str, str]] = None,
) -> bool:
	data = env or os.environ
	if _is_truthy(data.get("VIDEOS_WAN_FIXTURE_ONLY")):
		return False
	if force_fixture is True:
		return False
	if _is_truthy(data.get("ADK_USE_FIXTURE")):
		return False
	flags = ("SMOKE_VIDEOS", "SMOKE_ADAPTERS", "SMOKE_ADK")
	return any(_is_truthy(data.get(flag)) for flag in flags)


def _render_fixture(
	*,
	prompt: str,
	num_frames: int,
	fps: int,
	width: int,
	height: int,
	seed: int,
	plan_id: Optional[str],
	chunk_index: Optional[int],
	chunk_count: Optional[int],
	metadata: MutableMapping[str, Any],
	options: Mapping[str, Any],
	start_frame: Optional[bytes],
	end_frame: Optional[bytes],
	output_dir: Path,
) -> VideoRenderResult:
	digest = _fixture_digest(
		prompt=prompt,
		num_frames=num_frames,
		fps=fps,
		width=width,
		height=height,
		seed=seed,
		chunk_index=chunk_index,
		chunk_count=chunk_count,
	)
	filename = f"{_slug(prompt)}-{digest}.mp4"
	target = output_dir / filename
	payload = _fixture_payload(seed=seed, width=width, height=height, num_frames=num_frames)
	target.write_bytes(payload)

	result_meta: MutableMapping[str, Any] = {
		"engine": "wan_fixture",
		"engine_version": "fixtures/v1",
		"prompt": prompt,
		"num_frames": num_frames,
		"fps": fps,
		"width": width,
		"height": height,
		"seed": seed,
		"chunk_index": chunk_index,
		"chunk_count": chunk_count,
		"options": dict(options),
	}
	if plan_id:
		result_meta["plan_id"] = plan_id
	if start_frame:
		result_meta["start_frame_hash"] = hashlib.sha1(start_frame).hexdigest()
	if end_frame:
		result_meta["end_frame_hash"] = hashlib.sha1(end_frame).hexdigest()
	if metadata:
		result_meta.update(metadata)

	duration = round(num_frames / fps, 4) if fps > 0 else 0.0
	return VideoRenderResult(
		path=target,
		metadata=result_meta,
		engine="wan_fixture",
		duration_s=duration,
		frame_count=num_frames,
		fps=fps,
	)


def _render_with_wan(
	*,
	prompt: str,
	num_frames: int,
	fps: int,
	width: int,
	height: int,
	seed: int,
	plan_id: Optional[str],
	chunk_index: Optional[int],
	chunk_count: Optional[int],
	metadata: MutableMapping[str, Any],
	options: Mapping[str, Any],
	start_frame: Optional[bytes],
	end_frame: Optional[bytes],
	output_dir: Path,
) -> VideoRenderResult:
	try:  # pragma: no cover - heavy deps resolved in smoke envs
		import torch
		from diffusers import DPMSolverMultistepScheduler, WanImageToVideoPipeline
		from diffusers.utils import export_to_video
	except Exception as exc:  # pragma: no cover
		raise VideoRenderError("diffusers/torch with Wan support required") from exc

	model_id = os.environ.get("VIDEOS_WAN_MODEL", DEFAULT_MODEL_ID)
	dtype_name = os.environ.get("VIDEOS_WAN_DTYPE", "float16")
	torch_dtype = getattr(torch, dtype_name, torch.float16)
	preset = os.environ.get("VIDEOS_WAN_DEVICE_PRESET")
	device_map = _device_map_from_env(preset)
	cache_ttl = _cache_ttl_seconds()
	model_key = f"videos_wan::{model_id}" if model_id else "videos_wan"

	def _loader() -> WanImageToVideoPipeline:
		pipeline = WanImageToVideoPipeline.from_pretrained(
			model_id,
			torch_dtype=torch_dtype,
			low_cpu_mem_usage=True,
			use_safetensors=True,
		)
		pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
		if device_map:
			pipeline.enable_model_cpu_offload()
		else:
			device = os.environ.get("VIDEOS_WAN_DEVICE", "cuda")
			pipeline.to(device)
		return pipeline

	destination = output_dir / f"{_slug(prompt)}-{uuid.uuid4().hex[:8]}.mp4"

	with gpu_utils.model_context(
		model_key,
		loader=_loader,
		weights=model_id,
		device_map=device_map,
		keep_warm=True,
		warm_ttl_s=cache_ttl,
	) as ctx:
		pipeline = ctx.pipeline
		device_pref = (device_map or {}).get("unet") or os.environ.get("VIDEOS_WAN_DEVICE", "cuda")
		try:
			generator = torch.Generator(device=device_pref)
		except Exception:
			generator = torch.Generator()
		generator.manual_seed(seed)
		pipeline_output = pipeline(
			prompt=prompt,
			num_frames=num_frames,
			generator=generator,
			width=width,
			height=height,
			output_type="pil",
			**_adapter_call_kwargs(options),
		)
		frames = getattr(pipeline_output, "frames", None) or getattr(pipeline_output, "images", None)
		if not frames:
			raise VideoRenderError("Wan pipeline returned no frames")
		frame_set = frames[0] if isinstance(frames, (list, tuple)) and isinstance(frames[0], list) else frames
		export_to_video(frame_set, destination.as_posix(), fps=fps)

	result_meta: MutableMapping[str, Any] = {
		"engine": "wan2.1",
		"model_id": model_id,
		"num_frames": num_frames,
		"fps": fps,
		"width": width,
		"height": height,
		"seed": seed,
		"chunk_index": chunk_index,
		"chunk_count": chunk_count,
		"options": dict(options),
		"device_map": dict(device_map or {}),
	}
	if plan_id:
		result_meta["plan_id"] = plan_id
	if metadata:
		result_meta.update(metadata)
	if start_frame:
		result_meta["start_frame_hash"] = hashlib.sha1(start_frame).hexdigest()
	if end_frame:
		result_meta["end_frame_hash"] = hashlib.sha1(end_frame).hexdigest()

	duration = round(num_frames / fps, 4) if fps > 0 else 0.0
	return VideoRenderResult(
		path=destination,
		metadata=result_meta,
		engine="wan2.1",
		duration_s=duration,
		frame_count=num_frames,
		fps=fps,
	)


def _validate_dimensions(*, num_frames: int, fps: int, width: int, height: int) -> None:
	if num_frames <= 0:
		raise ValueError("num_frames must be positive")
	if fps <= 0:
		raise ValueError("fps must be positive")
	if width <= 0 or height <= 0:
		raise ValueError("width/height must be positive")


def _default_output_dir() -> Path:
	base = Path(os.environ.get("ARTIFACTS_DIR", Path.cwd() / "artifacts"))
	target = base / "videos_wan"
	target.mkdir(parents=True, exist_ok=True)
	return target


def _fixture_payload(*, seed: int, width: int, height: int, num_frames: int) -> bytes:
	rng = random.Random(seed & 0xFFFFFFFF)
	size_hint = max(256, min(width * height // 4, 4096))
	byte_count = min(8192, size_hint * max(1, num_frames // 8))
	header = bytearray(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2\x00\x00\x00\x08free\x00\x00\x00\x1cmdat")
	payload = bytearray(rng.getrandbits(8) for _ in range(byte_count))
	content = header + payload
	total_size = len(content)
	content[0:4] = total_size.to_bytes(4, "big", signed=False)
	return bytes(content)


def _fixture_digest(
	*,
	prompt: str,
	num_frames: int,
	fps: int,
	width: int,
	height: int,
	seed: int,
	chunk_index: Optional[int],
	chunk_count: Optional[int],
) -> str:
	payload = {
		"prompt": prompt,
		"num_frames": num_frames,
		"fps": fps,
		"width": width,
		"height": height,
		"seed": seed,
		"chunk_index": chunk_index,
		"chunk_count": chunk_count,
	}
	data = json.dumps(payload, sort_keys=True).encode("utf-8")
	return hashlib.sha1(data).hexdigest()[:10]


def _derive_seed(seed: Optional[int], *, prompt: str, chunk_index: int) -> int:
	if seed is not None:
		return int(seed)
	payload = f"{prompt}:{chunk_index}".encode("utf-8")
	return int(hashlib.sha1(payload).hexdigest()[:8], 16)


def _slug(value: str) -> str:
	cleaned = value.strip().lower()
	cleaned = re.sub(r"[^a-z0-9_-]+", "-", cleaned)
	return cleaned[:48] or "videos"


def _adapter_call_kwargs(options: Mapping[str, Any]) -> Mapping[str, Any]:
	allowed = {"num_inference_steps", "guidance_scale", "negative_prompt", "motion_bucket_id", "megapixels"}
	return {k: v for k, v in options.items() if k in allowed}


def _device_map_from_env(preset: Optional[str]) -> Optional[Mapping[str, str]]:
	raw = os.environ.get("VIDEOS_WAN_DEVICE_MAP")
	if raw:
		try:
			parsed = json.loads(raw)
			if isinstance(parsed, dict):
				return {str(k): str(v) for k, v in parsed.items()}
		except Exception:
			LOG.warning("Failed to parse VIDEOS_WAN_DEVICE_MAP, ignoring")
	if preset:
		try:
			return gpu_utils.compute_device_map(preset)
		except KeyError:
			LOG.warning("Unknown VIDEOS_WAN_DEVICE_PRESET=%s", preset)
	return None


def _cache_ttl_seconds() -> Optional[float]:
	raw = os.environ.get("VIDEOS_WAN_CACHE_TTL")
	if not raw:
		return 900.0
	try:
		value = float(raw)
		return value if value > 0 else None
	except ValueError:
		return 900.0


def _is_truthy(value: Optional[str]) -> bool:
	if value is None:
		return False
	return value.strip().lower() in TRUTHY


__all__ = ["VideoRenderResult", "VideoRenderError", "render_clip", "should_use_real_engine"]
