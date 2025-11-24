#!/usr/bin/env bash
set -euo pipefail

# tests/run_tests.sh — Run pytest with PYTHONPATH set to the project's src/
# Usage:
#   From repo root:  bash tests/run_tests.sh -q tests/test_stub_adapter.py
#   From tests/:      bash run_tests.sh -q test_stub_adapter.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$REPO_ROOT/src"

cd "$REPO_ROOT"

if [ "$#" -eq 0 ]; then
  # No args: run full tests folder
  pytest "$REPO_ROOT/tests"
else
  pytest "$@"
fi
