# Makefile — convenience targets for development
#
# Usage examples:
#   make test                # runs the test suite (defaults to tests/)
#   make test TEST=tests/test_stub_adapter.py

TEST ?= tests
# Ensure both repo root and `src` are on PYTHONPATH so tests that import
# top-level scripts and the package both resolve during pytest collection.
PYPATH := $(abspath .):$(abspath src)
CONFIRM ?= 0

.PHONY: test
test:
	@echo "Running pytest against '$(TEST)' with PYTHONPATH=$(PYPATH)"
	PYTHONPATH=$(PYPATH) pytest -q $(TEST)

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

.PHONY: register-root-agent
register-root-agent:
	@echo "Registering root agent config (CONFIRM=$(CONFIRM))"
	@if [ "$(CONFIRM)" = "1" ]; then \
	  FLAGS="--confirm"; \
	else \
	  FLAGS="--dry-run"; \
	fi; \
	PYTHONPATH=$(PYPATH) python scripts/register_root_agent.py --config configs/root_agent.yaml $$FLAGS
