```markdown
# Packaging Proposal: ADK integration (draft)

Status: DRAFT — no changes made to `pyproject.toml` yet. This document contains a proposed approach and a sample `pyproject` snippet. Adding dependencies is an "ask-first" action; please confirm before I modify any project manifests.

Summary
-------
- Goal: Make the repository consumable as a package while documenting and optionally adding the ADK runtime dependency used by some integration tests and tools.
- Scope: Draft a minimal `pyproject` change, list required environment variables and gating for integration tests, and provide a safe migration plan.

Why this change
----------------
- Several tools and fixture-mode smoke tests reference an external ADK runtime (ADK). Packaging the repo or adding an explicit optional dependency clarifies install-time requirements and helps CI.

Proposal (high level)
---------------------
1. Do not inject ADK as a hard `install_requires` default for library users. Instead expose it as an optional extra named `adk` or `adk-integration`.
2. Add documentation under `resources/` describing how to run gated ADK integration tests and what env vars/credentials are required.
3. Optionally provide a small `extras` entry so contributors can easily install test deps: e.g. `pip install -e '.\[adk\]'`.

Poetry example (suggested snippet for `pyproject.toml`)
-----------------------------------------------------
This is an example for a Poetry-managed project. Do NOT apply it automatically — confirm the preferred packaging tool and target `pyproject.toml` file(s).

```toml
[tool.poetry]
name = "sparkle-motion"
version = "0.0.0"
description = "..."

[tool.poetry.dependencies]
python = "^3.10"

[tool.poetry.extras]
# Add ADK as an opt-in extra for integration tests and runtime users who need ADK features
adk = ["google-adk>=0.1.0"]
```

Setuptools / PEP 621 (pyproject) example
---------------------------------------
If the project uses `setuptools` with PEP 621 metadata, add an `extras_require` mapping in setup configuration (or update the project metadata accordingly). Example snippet for `setup.cfg` or `setup.py`:

adk =
```ini
[options.extras_require]
adk =
    google-adk>=0.1.0
```

Open questions / choices
------------------------
- Exact package name for ADK runtime (e.g., `adk-sdk`, `adk-python`, or internal package). I did not assume a package name—please confirm the upstream package and desired version.
- Which `pyproject.toml` to edit (root vs subpackages). I can prepare a branch and apply the change to whichever manifest you prefer.

Integration / Gated test instructions
-----------------------------------
These tests require credentials and should be guarded in CI.

Required environment variables (example):
- `ADK_API_KEY` — API key for ADK service
- `ADK_PROJECT` — project id or name
- `ADK_REGION` — optional
- `ADK_USE_FIXTURE=1` — run tests in fixture-mode

Example local run (once credentials are available):

```bash
# install optional extras (if we add the extra)
pip install -e '.\[adk\]'

# run gated integration tests (only those tagged or in a specific folder)
ADK_API_KEY=... ADK_PROJECT=... ADK_USE_FIXTURE=1 pytest -q tests/integration -k adk -q
```

Acceptance criteria for applying manifest changes
------------------------------------------------
- Confirm package name + version for the ADK runtime.
- Confirm which `pyproject.toml` to update (root or subpackages).
- Confirm whether `adk` should be an extra (recommended) or a direct dependency.

Next steps I can take after your confirmation
-------------------------------------------
1. Add the `adk` extra to the confirmed `pyproject.toml` (or `setup.cfg`) and run `pytest` to ensure no regressions.
2. Add CI gating in `.github/workflows` to skip ADK tests by default and enable them via repository secrets and a manual workflow or schedule.

If you want me to apply the manifest edits now, please confirm the package name and which `pyproject.toml` file to modify.

```
