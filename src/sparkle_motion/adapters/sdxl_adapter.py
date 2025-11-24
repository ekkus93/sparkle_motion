"""SDXL adapter stub.

Real implementation should provide `generate_frames(movie_plan, asset_refs, run_dir)`
which writes start/end frame images and returns updated asset_refs. This stub
raises MissingDependencyError with guidance.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from .common import MissingDependencyError


def generate_frames(
  movie_plan: Dict[str, Any],
  asset_refs: Dict[str, Any],
  run_dir: Path,
  repo_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
  num_inference_steps: int = 20,
  guidance_scale: float = 7.5,
  device: Optional[str] = None,
) -> Dict[str, Any]:
  """Generate start/end frames for shots using SDXL (diffusers).

  This is a real adapter implementation that attempts to import the required
  libraries (`torch`, `diffusers`) and will raise `MissingDependencyError`
  with actionable instructions if they are not installed.

  Notes:
  - This function respects `movie_plan.get('metadata', {}).get('seed')` if present
    to allow deterministic outputs via a torch.Generator.
  - It writes PNG files named `<shot_id>_start.png` and `<shot_id>_end.png` into
    `run_dir` and updates the `asset_refs` dict with the paths.
  """

  try:
    import torch
    from diffusers import StableDiffusionXLPipeline
  except Exception as e:  # ImportError or others
    raise MissingDependencyError(
      "SDXL adapter requires additional packages.\n"
      "Install them with: pip install diffusers[torch] transformers safetensors accelerate torch Pillow\n"
      f"Import error: {e!r}"
    )

  # choose device
  if device is None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    device_str = device

  # prepare seed generator if provided
  seed = None
  md = movie_plan.get("metadata", {}) if isinstance(movie_plan, dict) else {}
  seed_val = md.get("seed")
  generator = None
  if seed_val is not None:
    try:
      seed = int(seed_val)
    except Exception:
      seed = None
  if seed is not None:
    gen_device = "cuda" if device_str == "cuda" else "cpu"
    try:
      generator = torch.Generator(device=gen_device).manual_seed(seed)
    except Exception:
      # fallback to default generator
      generator = None

  # load pipeline
  # NOTE: model weights must be available in the environment (or HF token provided)
  pipeline = StableDiffusionXLPipeline.from_pretrained(repo_id, safety_checker=None)
  # move to device and set dtype
  if device_str == "cuda":
    pipeline = pipeline.to(torch.float16).to(device_str)
  else:
    pipeline = pipeline.to(torch.float32).to(device_str)

  shots = movie_plan.get("shots", []) if isinstance(movie_plan, dict) else []
  for s in shots:
    sid = s.get("id")
    if not sid:
      continue

    start_prompt = s.get("start_frame_prompt") or s.get("visual_description") or ""
    end_prompt = s.get("end_frame_prompt") or s.get("visual_description") or ""

    # generate start frame
    out_start = pipeline(
      start_prompt,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      generator=generator,
    )
    img_start = out_start.images[0]
    start_path = run_dir / f"{sid}_start.png"
    img_start.save(start_path)
    asset_refs.setdefault("shots", {}).setdefault(sid, {})["start_frame"] = str(start_path)

    # generate end frame
    out_end = pipeline(
      end_prompt,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      generator=generator,
    )
    img_end = out_end.images[0]
    end_path = run_dir / f"{sid}_end.png"
    img_end.save(end_path)
    asset_refs.setdefault("shots", {}).setdefault(sid, {})["end_frame"] = str(end_path)

  return asset_refs

