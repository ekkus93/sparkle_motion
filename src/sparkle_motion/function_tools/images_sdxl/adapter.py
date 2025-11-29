from __future__ import annotations

import binascii
import contextlib
import hashlib
import io
import logging
import os
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sparkle_motion import gpu_utils
from sparkle_motion.utils.dedupe import compute_phash

LOG = logging.getLogger(__name__)
TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ImageRenderResult:
    path: Path
    data: bytes
    metadata: Dict[str, Any]


def should_use_real_engine(env: Optional[Mapping[str, str]] = None) -> bool:
    data = env or os.environ
    if data.get("ADK_USE_FIXTURE", "0").strip().lower() in TRUTHY:
        return False
    for key in ("SMOKE_IMAGES", "SMOKE_ADAPTERS"):
        if data.get(key, "0").strip().lower() in TRUTHY:
            return True
    return False


def render_images(
    prompt: str,
    opts: Optional[Mapping[str, Any]] = None,
    *,
    output_dir: Optional[Path] = None,
) -> List[ImageRenderResult]:
    """Render images using either the deterministic fixture or the real SDXL path."""

    options = dict(opts or {})
    count = int(options.get("count", 1))
    if count <= 0:
        raise ValueError("count must be positive")

    width = int(options.get("width", 1024))
    height = int(options.get("height", 1024))
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    batch_start = int(options.get("batch_start", 0))
    seed = options.get("seed")
    sampler = str(options.get("sampler", "ddim"))
    steps = int(options.get("steps", 30))
    cfg_scale = float(options.get("cfg_scale", 7.5))
    negative_prompt = options.get("negative_prompt")
    extra_metadata = options.get("metadata") or {}

    target_dir = output_dir or _default_output_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    if should_use_real_engine() and os.environ.get("IMAGES_SDXL_FIXTURE_ONLY") != "1":
        try:
            return _render_real_images(
                prompt=prompt,
                count=count,
                seed=seed,
                width=width,
                height=height,
                sampler=sampler,
                steps=steps,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                extra_metadata=extra_metadata,
                batch_start=batch_start,
                output_dir=target_dir,
                options=options,
            )
        except gpu_utils.GpuBusyError:
            raise
        except Exception as exc:  # pragma: no cover - smoke-only path
            LOG.warning("Real SDXL render failed, falling back to fixture: %s", exc)

    return _render_fixture_images(
        prompt=prompt,
        count=count,
        seed=seed,
        width=width,
        height=height,
        sampler=sampler,
        steps=steps,
        cfg_scale=cfg_scale,
        negative_prompt=negative_prompt,
        extra_metadata=extra_metadata,
        batch_start=batch_start,
        output_dir=target_dir,
    )


def _default_output_dir() -> Path:
    override = os.environ.get("IMAGES_SDXL_OUTPUT_DIR")
    if override:
        return Path(override)
    base = os.environ.get("ARTIFACTS_DIR")
    root = Path(base) if base else Path.cwd() / "artifacts"
    return root / "images_sdxl"


def _render_fixture_images(
    *,
    prompt: str,
    count: int,
    seed: Optional[int],
    width: int,
    height: int,
    sampler: str,
    steps: int,
    cfg_scale: float,
    negative_prompt: Optional[str],
    extra_metadata: Mapping[str, Any],
    batch_start: int,
    output_dir: Path,
) -> List[ImageRenderResult]:
    results: List[ImageRenderResult] = []
    for offset in range(count):
        image_index = batch_start + offset
        image_seed = _derive_seed(seed, prompt, image_index)
        pixels = _generate_pixels(width, height, image_seed)
        png_bytes = _encode_png(width, height, pixels)
        filename = f"{_slug(prompt)}-{image_index:04d}.png"
        path = output_dir / filename
        path.write_bytes(png_bytes)
        phash = compute_phash(pixels, width, height)

        base_metadata = {
            "engine": "fixture",
            "engine_version": "images_sdxl_fixture_v1",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": image_seed,
            "index": image_index,
            "width": width,
            "height": height,
            "sampler": sampler,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "phash": phash,
        }
        metadata = dict(extra_metadata)
        metadata.update(base_metadata)
        results.append(ImageRenderResult(path=path, data=png_bytes, metadata=metadata))
    return results


def _render_real_images(
    *,
    prompt: str,
    count: int,
    seed: Optional[int],
    width: int,
    height: int,
    sampler: str,
    steps: int,
    cfg_scale: float,
    negative_prompt: Optional[str],
    extra_metadata: Mapping[str, Any],
    batch_start: int,
    output_dir: Path,
    options: Mapping[str, Any],
) -> List[ImageRenderResult]:
    try:  # pragma: no cover - smoke-only path
        import torch  # type: ignore
        from diffusers import AutoPipelineForText2Image  # type: ignore
    except Exception as exc:  # pragma: no cover - smoke-only path
        raise RuntimeError("diffusers and torch are required for real SDXL renders") from exc

    model_id = os.environ.get("IMAGES_SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
    device = os.environ.get("IMAGES_SDXL_DEVICE", "cuda")
    variant = os.environ.get("IMAGES_SDXL_VARIANT", "fp16")
    torch_dtype = getattr(torch, os.environ.get("IMAGES_SDXL_DTYPE", "float16"), torch.float16)

    def _loader():  # pragma: no cover - smoke-only path
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant=variant,
            use_safetensors=True,
        )
        if device.startswith("cuda"):
            pipeline.to(device)
        if os.environ.get("IMAGES_SDXL_ENABLE_XFORMERS", "1").strip().lower() in TRUTHY:
            with contextlib.suppress(Exception):
                pipeline.enable_xformers_memory_efficient_attention()
        return pipeline

    cache_ttl = _cache_ttl_seconds()
    results: List[ImageRenderResult] = []
    with gpu_utils.model_context(
        f"sdxl:{model_id}",
        loader=_loader,
        weights=model_id,
        offload=True,
        xformers=True,
        keep_warm=True,
        warm_ttl_s=cache_ttl,
        block_until_gpu_free=False,
    ) as ctx:  # pragma: no cover - smoke-only path
        for offset in range(count):
            image_index = batch_start + offset
            image_seed = _derive_seed(seed, prompt, image_index)
            generator_device = device if device.startswith("cuda") else "cpu"
            generator = torch.Generator(generator_device).manual_seed(image_seed)
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_images_per_prompt": 1,
                "num_inference_steps": steps,
                "guidance_scale": cfg_scale,
                "generator": generator,
            }
            for key in ("prompt_2", "negative_prompt_2", "denoising_start", "denoising_end"):
                if key in options and options[key] is not None:
                    pipeline_kwargs[key] = options[key]
            output = ctx.pipeline(**pipeline_kwargs)
            images = getattr(output, "images", output)
            if not images:
                raise RuntimeError("SDXL pipeline returned no images")
            pil_image = images[0]
            rgb_image = pil_image.convert("RGB")
            pixels = list(rgb_image.getdata())
            phash = compute_phash(pixels, rgb_image.width, rgb_image.height)
            buffer = io.BytesIO()
            rgb_image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()
            filename = f"{_slug(prompt)}-{image_index:04d}.png"
            path = output_dir / filename
            path.write_bytes(png_bytes)
            base_metadata = {
                "engine": "sdxl",
                "engine_version": model_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": image_seed,
                "index": image_index,
                "width": rgb_image.width,
                "height": rgb_image.height,
                "sampler": sampler,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "device": device,
                "phash": phash,
            }
            metadata = dict(extra_metadata)
            metadata.update(base_metadata)
            results.append(ImageRenderResult(path=path, data=png_bytes, metadata=metadata))
    return results


def _generate_pixels(width: int, height: int, seed: int) -> List[Sequence[int]]:
    rng = random.Random(seed)
    pixels: List[Sequence[int]] = []
    for _ in range(width * height):
        pixels.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    return pixels


def _encode_png(width: int, height: int, pixels: List[Sequence[int]]) -> bytes:
    raw = bytearray()
    idx = 0
    for _ in range(height):
        raw.append(0)
        for _ in range(width):
            r, g, b = pixels[idx]
            raw.extend((r, g, b))
            idx += 1
    body = bytes(raw)
    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(body, 9)),
        _png_chunk(b"IEND", b""),
    ]
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)

def _derive_seed(seed: Optional[int], prompt: str, index: int) -> int:
    if seed is None:
        digest = hashlib.sha1(f"{prompt}|{index}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)
    return (int(seed) + index) & 0xFFFFFFFF


def _slug(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})
    cleaned = cleaned.lower()
    return cleaned[:48] or "image"


def _cache_ttl_seconds() -> Optional[float]:
    raw = os.environ.get("IMAGES_SDXL_CACHE_TTL_S")
    if raw is None:
        return 900.0
    try:
        ttl = float(raw)
    except ValueError:
        return 900.0
    return None if ttl <= 0 else ttl


__all__ = ["ImageRenderResult", "render_images", "should_use_real_engine"]
