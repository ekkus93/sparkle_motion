**Model Install Notes**

This document collects concise, copy-pasteable install and download instructions for running heavy models on a Colab A100 (or similar GPU) environment. It is intended for development runs (Colab/Dev) — do not run these on production machines without checking compatibility with your drivers and policies.

**Quick Check (on target machine)**
- **Check CUDA / GPU:** `nvidia-smi` — confirm the driver and GPU model (A100 typically pairs with CUDA 11.8).
- **Python:** prefer Python 3.10+ in Colab; confirm with `python --version`.

**1) Prepare pip & basic deps (Colab / A100)**
- Upgrade pip and install build tools:
```
python -m pip install --upgrade pip setuptools wheel
```
- Install `ffmpeg` on Colab (needed for many video/audio steps):
```
apt-get update && apt-get install -y ffmpeg
```

**2) Torch + CUDA (Colab A100 / typical setup)**
- Verify CUDA version via `nvidia-smi`. For A100, CUDA 11.8 is common; choose matching wheel.
- Example (recommended starting point, verify the official PyTorch website for newest stable commands):
```
# Example: install PyTorch for CUDA 11.8
python -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
- Notes:
  - Replace `cu118` with the correct CUDA tag if `nvidia-smi` shows a different version.
  - Use the official PyTorch install selector (https://pytorch.org/get-started/locally/) to get exact wheel URLs for your environment.

**3) Diffusers / Model runtime packages**
- Core packages (minimal):
```
python -m pip install --upgrade diffusers transformers accelerate safetensors huggingface_hub
```
- Optional / performance:
```
python -m pip install xformers  # optional, speeds attention on some pipelines (may require a compatible wheel)
python -m pip install bitsandbytes  # optional, for 8-bit optimizations (check compatibility)
```
- If you plan to use `accelerate` for offloading or multi-GPU, run:
```
accelerate config  # interactively create a config, prefer mixed precision + device_map offload options
```

**4) Hugging Face authentication & downloading weights**
- Login to HF to allow private model downloads:
```
huggingface-cli login
```
- Programmatic download example (safe, small-file-first pattern):
```python
from huggingface_hub import hf_hub_download
repo_id = "Wan-AI/Wan2.1-I2V-14B-720P"
# Download a single file (e.g., model weights or pipeline JSON)
path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
print(path)
```
- Or load directly with `diffusers` (example):
```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-I2V-14B-720P",
    torch_dtype=torch.float16,  # prefer float16 on GPU
    safety_checker=None,        # opt-out if you handle checks separately
)
pipe = pipe.to("cuda")
```

**5) Wan2.1 / I2V / T2V specific notes**
- Common HF repo IDs (examples — verify latest names on the model card):
  - `Wan-AI/Wan2.1-I2V-14B-720P`
  - `Wan-AI/Wan2.1-T2V-...`
- Loading pattern: use `DiffusionPipeline.from_pretrained(...)` or the project-provided pipeline class (some Wan releases expose a custom pipeline class such as `WanImageToVideoPipeline`). Prefer `torch_dtype=torch.float16` + `.to("cuda")` for GPU runs.
- Memory & offload:
  - Wan2.1-class models are very large. Use `accelerate` with `device_map="auto"` or `offload_folder` strategies if multiple GPUs or CPU offload are available.
  - Example with accelerate offload (high level):
    - Configure with `accelerate config` and enable `offload_to_cpu` or `offload_state_dict` options.

**6) Wav2Lip (lipsync) quick setup**
- Clone the repo and install minimal requirements (example):
```
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
python -m pip install -r requirements.txt
# download checkpoint (example):
wget -O checkpoints/wav2lip_gan.pth "https://.../wav2lip_gan.pth"
```
- Run inference (example):
```
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face input.mp4 --audio input.wav --outfile out.mp4
```

**7) Coqui TTS (text→speech) quick setup**
- Install:
```
python -m pip install TTS
```
- Example usage:
```python
from TTS.api import TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file(text="Hello world", file_path="out.wav")
```

**8) ffmpeg (assembly & processing)**
- On Colab or Debian-based systems:
```
apt-get update && apt-get install -y ffmpeg
```
- On other systems, install via package manager or provide a prebuilt binary on PATH. `assemble_clips` helper in the repo calls `ffmpeg` on PATH and will raise on non-zero exit codes.

**9) Security and reproducibility notes**
- safetensors: prefer `safetensors` when available for faster load and safety; use `from_pretrained(..., use_safetensors=True)` where supported.
- `trust_remote_code`: many model repo READMEs use `trust_remote_code=True` to load custom pipeline code. Only set `trust_remote_code=True` if you trust the repository — otherwise inspect or vendor the code.
- Pin heavy versions (torch/diffusers/transformers) in your Colab session for reproducibility.

**SDXL-specific notes (Diffusers canonical guidance)**
- Recommended pipeline loading:
```py
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")
```
- Use `from_single_file()` when you have a single `.ckpt` / `.safetensors` file.
- Use the base + refiner workflow when higher fidelity is required; pass `denoising_end`/`denoising_start` to shard denoising between base and refiner or call the refiner on the base output.
- Determinism: pass a `torch.Generator` seeded with your chosen `seed` to the pipeline call (e.g., `generator = torch.Generator(device="cuda").manual_seed(seed)`). Expose `seed` in adapter options and return it with artifact metadata.
- Memory optimizations to consider in `gpu_utils.model_context` and docs:
  - `pipe.enable_model_cpu_offload()` to avoid OOM on single-GPU hosts.
  - `pipe.enable_xformers_memory_efficient_attention()` when `xformers` is installed.
  - `torch.compile` for speed-ups on `torch>=2.0` (optional).
- Micro-conditioning: support `original_size`, `target_size`, and crop conditioning parameters when appropriate; default to `(1024, 1024)` for best quality.
- Invisible watermark: diffusers recommends `invisible-watermark>=0.2.0`; the docs show `add_watermarker=False` to opt-out.

**Quick SDXL install checklist (Colab-friendly)**
```bash
# core
python -m pip install --upgrade pip setuptools wheel
python -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install diffusers[torch] transformers accelerate safetensors huggingface_hub invisible-watermark
# optional performance
python -m pip install xformers bitsandbytes
```

Note: verify the exact `diffusers` and `torch` versions for compatibility; SDXL guidance in the HF docs assumes a recent `diffusers` release (>=0.35.x at time of reference).

**10) Troubleshooting checklist**
- If CUDA errors appear, confirm `nvidia-smi` and that the installed torch wheel matches the driver/CUDA (mismatch leads to runtime failures).
- For memory OOMs: try `torch_dtype=torch.float16`, smaller batch sizes, lower resolution, or `accelerate` offload.
- If `hf_hub_download` fails with auth error: run `huggingface-cli login` and make sure `HF_TOKEN` is set in env.

**Acceptance**
- Behavior: Provides copy-pasteable Colab and local commands to prepare a GPU runtime for Wan2.1, Diffusers, Wav2Lip, Coqui TTS and `ffmpeg`.
- Interfaces: commands and Python snippets above; no code in `src/` will be modified by this doc.
- Persistence/IO: the doc describes how to download and store model weights locally or in Colab Drive; it does not perform downloads itself.
- Limits: exact wheel URLs may change; always verify current wheel links at the official PyTorch and package pages before installing.

---
If you want, I can also:
- Pin specific torch wheel URLs for a chosen torch version (tell me which one you want), or
- Generate a small Colab notebook cell set that runs the exact install sequence and verifies imports.

**Commands (copy-paste)**
```
# Quick: upgrade pip and install core packages
python -m pip install --upgrade pip setuptools wheel
apt-get update && apt-get install -y ffmpeg
# Example PyTorch for CUDA 11.8 (verify at pytorch.org)
python -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
# Diffusers & HF
python -m pip install diffusers transformers accelerate safetensors huggingface_hub
# Optional performance
python -m pip install xformers bitsandbytes
```
