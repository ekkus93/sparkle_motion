How to run the tests in this repository
=====================================

This project keeps its importable package sources under `src/`. When running tests directly you must ensure Python can import the package by adding the `src` directory to `PYTHONPATH`.

Examples
--------

- From the repository root (recommended):

```bash
# Runs the specific test file from the repo root
PYTHONPATH=./src pytest -q tests/test_stub_adapter.py
```

- From inside the `tests/` directory (note the different relative path):

```bash
# When your current directory is `tests/` use ../src so Python finds the package
PYTHONPATH=../src pytest -q test_stub_adapter.py
```

Why this is necessary
---------------------
Python's import system searches the directories in `sys.path` for packages. Setting `PYTHONPATH` to the project `src/` directory ensures `import sparkle_motion` resolves to `./src/sparkle_motion` rather than failing to find the package when invoked from other working directories.

Notes
-----
- Some integration tests require `ffmpeg` on PATH; those tests are skipped when `ffmpeg` is not available so you can run smoke tests without extra system dependencies.
- Prefer running tests from the repo root so example commands in CI and docs are consistent.
