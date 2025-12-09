# Makefile — convenience targets for development
#
# Usage examples:
#   make test                # runs the test suite (defaults to tests/)
#   make test TEST=tests/test_stub_adapter.py

TEST ?= tests
# Default Hugging Face repos to download (space-separated). Matches notebook Cell 3.
HF_MODELS ?= stabilityai/stable-diffusion-xl-base-1.0 \
	stabilityai/stable-diffusion-xl-refiner-1.0 \
	Wan-AI/Wan2.1-I2V-14B-720P \
	Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers \
	ResembleAI/chatterbox
# Optional Hugging Face repos to download (space-separated, e.g., "repo1 repo2").
MODELS ?= $(HF_MODELS)
# Location for prepared workspace/model cache when using download-models target.
MODEL_WORKSPACE ?= $(abspath artifacts/model_workspace)
WAV2LIP_WORKSPACE ?= $(MODEL_WORKSPACE)
WAV2LIP_BRANCH ?=
WAV2LIP_SKIP_PIP ?= 0
WAV2LIP_SKIP_CHECKPOINT ?= 0
WAV2LIP_BRANCH_FLAG := $(if $(strip $(WAV2LIP_BRANCH)),--branch $(WAV2LIP_BRANCH),)
WAV2LIP_SKIP_PIP_FLAG := $(if $(filter 1,$(WAV2LIP_SKIP_PIP)),--skip-pip,)
WAV2LIP_SKIP_CHECKPOINT_FLAG := $(if $(filter 1,$(WAV2LIP_SKIP_CHECKPOINT)),--skip-checkpoint,)
# Ensure both repo root and `src` are on PYTHONPATH so tests that import
# top-level scripts and the package both resolve during pytest collection.
PYPATH := $(abspath .):$(abspath src)
CONFIRM ?= 0

.PHONY: test
test:
	@echo "Running pytest against '$(TEST)' with PYTHONPATH=$(PYPATH)"
	PYTHONPATH=$(PYPATH) pytest -q $(TEST)

.PHONY: unit
unit:
	@echo "Running non-GPU unit tests via pytest with PYTHONPATH=$(PYPATH)"
	PYTHONPATH=$(PYPATH) pytest -q tests --ignore=tests/gpu

.PHONY: unit-gpu
unit-gpu:
	@echo "Running GPU-only unit tests via pytest with PYTHONPATH=$(PYPATH)"
	PYTHONPATH=$(PYPATH) pytest -q tests/gpu

.PHONY: unit-all
unit-all:
	@echo "Running all unit tests (non-GPU + GPU) via pytest with PYTHONPATH=$(PYPATH)"
	PYTHONPATH=$(PYPATH) pytest -q tests --ignore=tests/gpu
	PYTHONPATH=$(PYPATH) pytest -q tests/gpu

.PHONY: lint
lint:
	@echo "Checking for legacy runner artifacts"
	python scripts/check_no_legacy.py
	@echo "Running ruff against '.'"
	ruff check .

.PHONY: smoke
smoke:
	@echo "Running smoke tests (tests/smoke) with PYTHONPATH=$(PYPATH)"
	PYTHONPATH=$(PYPATH) pytest -q tests/smoke

.PHONY: download-models
download-models:
	@if [ -z "$(strip $(MODELS))" ]; then \
	  echo "Set MODELS=\"repo1 repo2\" or export HF_MODELS before running make download-models."; \
	  exit 1; \
	fi
	@echo "Preparing workspace $(MODEL_WORKSPACE) and downloading: $(MODELS)"
	PYTHONPATH=$(PYPATH) python scripts/colab_drive_setup.py $(MODEL_WORKSPACE) --no-smoke $(foreach repo,$(MODELS),--model $(repo))

.PHONY: wav2lip-install
wav2lip-install:
	@echo "Installing Wav2Lip assets under $(WAV2LIP_WORKSPACE)"
	PYTHONPATH=$(PYPATH) python scripts/install_wav2lip.py --workspace $(WAV2LIP_WORKSPACE) \
		$(WAV2LIP_BRANCH_FLAG) $(WAV2LIP_SKIP_PIP_FLAG) $(WAV2LIP_SKIP_CHECKPOINT_FLAG)

.PHONY: register-root-agent
register-root-agent:
	@echo "Registering root agent config (CONFIRM=$(CONFIRM))"
	@if [ "$(CONFIRM)" = "1" ]; then \
	  FLAGS="--confirm"; \
	else \
	  FLAGS="--dry-run"; \
	fi; \
	PYTHONPATH=$(PYPATH) python scripts/register_root_agent.py --config configs/root_agent.yaml $$FLAGS
