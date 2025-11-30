# sparkle_motion

Running ADK integration tests
-----------------------------
This project includes an env-gated ADK integration test that is skipped
by default to avoid accidental calls to external services. To run the
integration test locally against the bundled fixture shim (safe default):

```bash
PYTHONPATH=.:src ADK_PUBLISH_INTEGRATION=1 ADK_PROJECT=testproject \
	pytest -q tests/test_function_tools/test_script_agent_entrypoint_adk_integration.py::test_publish_artifact_returns_artifact_uri
```

If you want the test to run against a real `google.adk` SDK (installed
and authenticated in your environment), set `ADK_USE_FIXTURE=0` and make
sure `ADK_PROJECT` is set to a valid project id. Example:

```bash
PYTHONPATH=.:src ADK_PUBLISH_INTEGRATION=1 ADK_PROJECT=<your-project> ADK_USE_FIXTURE=0 \
	pytest -q tests/test_function_tools/test_script_agent_entrypoint_adk_integration.py::test_publish_artifact_returns_artifact_uri
```

CI note: Do not add `tests/fixtures` to `PYTHONPATH` globally in CI
jobs where the real ADK SDK is expected; the test inserts the fixtures
by default and provides `ADK_USE_FIXTURE` to opt out when needed.

Install & usage (ADK optional extra)
----------------------------------
To install the optional ADK runtime extras for this repository:

```bash
pip install .[adk]
```

Environment variables required for ADK integration tests / features:

- `ADK_PUBLISH_INTEGRATION`
- `ADK_PROJECT`
- `GOOGLE_APPLICATION_CREDENTIALS`

Set those in your environment when running ADK integration tests or using
`google.adk`-powered features.

Running tests (canonical commands)
----------------------------------
Use one of the commands below depending on the scope you want to run.

- **Run unit tests (folder-based):** runs tests placed under `tests/unit`.

```bash
source /home/phil/mambaforge/bin/activate sparkle_motion
PYTHONPATH=.:src pytest -q tests/unit
```

- **Run all tests (full repo run):** collects and runs every test under `tests/`.

```bash
source /home/phil/mambaforge/bin/activate sparkle_motion
PYTHONPATH=.:src pytest -q
```

Notes:
- We recommend keeping `PYTHONPATH=.:src` so both top-level scripts (e.g., `scripts/*`) and package code under `src/` import correctly during test collection.
- If you prefer marker- or CI-driven selection, use pytest markers (e.g., `-m unit`) or targeted folders instead of the full run.

GPU-backed SDXL smoke tests
---------------------------
The `images_sdxl` FunctionTool defaults to a deterministic PNG fixture so you
can develop without specialized hardware. To exercise the real SDXL pipeline on
hardware such as Google Colab’s T4/A100 instances:

1. **Install the ML/runtime extras** once the repo is available in your Colab
	 session (adjust CUDA wheels as needed):

	 ```bash
	 pip install -r requirements-ml.txt diffusers==0.30.2 torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121
	 ```

2. **Export the GPU env vars** so the adapter leaves fixture mode but still
	 publishes artifacts via the local shim:

	 ```bash
	 export PYTHONPATH=.:src
	 export ADK_USE_FIXTURE=1
	 export SMOKE_IMAGES=1
	 export IMAGES_SDXL_FIXTURE_ONLY=0
	 export IMAGES_SDXL_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
	 export IMAGES_SDXL_DEVICE="cuda"
	 ```

	 Optional knobs:
	 - `IMAGES_SDXL_CACHE_TTL_S` — seconds to keep the SDXL pipeline warm inside
		 `gpu_utils.model_context`.
	 - `IMAGES_SDXL_ENABLE_XFORMERS=0` — disable memory-efficient attention if
		 the bundled `xformers` wheel is incompatible with your runtime.

3. **Run the focused smoke tests** to verify real image generation:

	 ```bash
	 pytest -q tests/test_function_tools/test_images_sdxl_entrypoint.py tests/smoke/test_images_sdxl_smoke.py
	 ```

Troubleshooting:
- A `503` with `{"detail":"gpu busy"}` indicates another process is holding
	the GPU lock; rerun once the previous job exits.
- Missing `torch`/`diffusers` wheels trigger a logged warning and automatic
	fallback to the deterministic fixture so tests still pass, but no real GPU
	validation occurs.

GPU-backed Wan 2.1 smoke tests
------------------------------
The `videos_wan` FunctionTool mirrors the SDXL setup: fixture MP4s by default
so development is GPU-free, and the real Wan2.1 I2V pipeline is enabled only
when explicitly requested. To validate the real pipeline on GPU hardware
(Colab, A100, etc.):

1. **Install the Wan runtime extras** (diffusers with Wan support + CUDA
	wheels). Adjust versions to match your CUDA runtime.

	```bash
	pip install -r requirements-ml.txt diffusers==0.30.2 torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121
	```

2. **Export the env vars** so the adapter leaves fixture mode but still
	publishes artifacts via the local shim:

	```bash
	export PYTHONPATH=.:src
	export ADK_USE_FIXTURE=1                 # keep ADK shim local
	export SMOKE_VIDEOS=1                   # enable Wan adapter
	export VIDEOS_WAN_FIXTURE_ONLY=0        # ensure real pipeline is used
	export VIDEOS_WAN_MODEL="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
	export VIDEOS_WAN_DEVICE_PRESET="a100-80gb"   # or set VIDEOS_WAN_DEVICE_MAP JSON
	```

	Optional knobs:
	- `VIDEOS_WAN_CACHE_TTL` — seconds to keep the Wan weights warm inside
	  `gpu_utils.model_context`.
	- `VIDEOS_WAN_DEVICE` — explicit torch device (defaults to `cuda`).

3. **Run the focused suites** to exercise both the adapter and entrypoint:

	```bash
	pytest -q \
	  tests/unit/test_videos_wan_adapter.py \
	  tests/unit/test_videos_wan_models.py \
	  tests/unit/test_entrypoints_parametrized.py \
	  tests/smoke/test_function_tools_smoke.py -k videos_wan
	```

Troubleshooting mirrors SDXL:
- A `503` with `{"detail":"gpu busy"}` means another process has the GPU lock;
  wait for it to finish.
- Missing WAN weights or incompatible wheels fall back to the deterministic
  fixture, so the tests still pass but no GPU work occurs (check logs for the
  fallback notice).

A2A integration roadmap
-----------------------
Sparkle Motion currently orchestrates every FunctionTool from a single
`production_agent`, so peer-to-peer agent messaging is unnecessary and we have
not wired up the A2A protocol yet. Once we invite external or independently
hosted agents, A2A becomes attractive because it formalizes discovery,
capability negotiation, and transport between heterogeneous agent runtimes.

Why adopt A2A later?
- Third-party or remote teams can plug their agents into the pipeline without us
	baking their SDKs directly into `production_agent`.
- We can scale individual agents independently and still reuse the same message
	contract for telemetry + error propagation.
- A2A helps enforce a clean “tools over the wire” boundary, which is useful once
	runs span multiple services or data centers.

High-level adoption plan
1. Introduce an A2A server (or client) shim inside `production_agent` so plan
	 steps map to A2A invocations instead of direct FunctionTool calls.
2. Wrap the existing adapters (images, videos, TTS, QA, assemble, lipsync) with
	 lightweight A2A endpoints that expose the same payload schema they already
	 validate today.
3. Extend the artifact/telemetry hooks to include A2A trace IDs so cross-agent
	 debugging remains intact, and document the contract for external partners.
# sparkle_motion