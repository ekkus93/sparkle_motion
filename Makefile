# Makefile — convenience targets for development
#
# Usage examples:
#   make test                # runs the test suite (defaults to tests/)
#   make test TEST=tests/test_stub_adapter.py

TEST ?= tests
PYPATH := $(abspath src)

.PHONY: test
test:
	@echo "Running pytest against '$(TEST)' with PYTHONPATH=$(PYPATH)"
	PYTHONPATH=$(PYPATH) pytest -q $(TEST)

.PHONY: lint
lint:
	@echo "Running ruff against '.'"
	ruff check .
