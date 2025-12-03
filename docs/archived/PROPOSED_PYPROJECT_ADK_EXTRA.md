Proposed `pyproject`/packaging snippet to add an optional `adk` extra.

This file is a proposal only and is not applied to any manifest. It shows
two common patterns (Poetry and setuptools/PEP-621) that can be applied
to the project after you confirm package name and target manifest.

Poetry example
--------------
Add an optional extra named `adk` that keeps ADK out of the base install:

```toml
[tool.poetry.extras]
adk = ["google-adk>=0.1.0"]
```

Setuptools / PEP-621 example
---------------------------
Add an `extras_require` mapping (example for `setup.cfg` or equivalent):

```ini
[options.extras_require]
adk =
    google-adk>=0.1.0
```

Notes / next actions
--------------------
- Confirm the exact package name for the ADK runtime to use (example: `google-adk`).
- Confirm which manifest to edit (root `pyproject.toml` or a subpackage manifest).
- I will prepare a commit-ready diff and run tests if you reply with: `apply packaging manifest change: package=<pkgname> target=<path>`
