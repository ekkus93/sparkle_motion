ADK Test Fixtures
=================

Purpose
-------
This directory contains lightweight, file-backed shim implementations of
the ADK `ArtifactService` used by tests. They live under
`tests/fixtures/google/adk/artifacts/` so tests can explicitly load them
without risking shadowing an installed `google` package in developer
environments.

How tests use the fixtures
--------------------------
- Tests that need the shim explicitly insert `tests/fixtures` at the
  front of `sys.path` before importing the code under test. This ensures
  that `import google.adk.artifacts...` resolves to the fixture modules.

Example (in tests):

```py
# repo-root/tests/test_...py
repo_root = Path(__file__).resolve().parents[2]
fixtures_dir = str(repo_root / "tests" / "fixtures")
if fixtures_dir not in sys.path:
    sys.path.insert(0, fixtures_dir)
    importlib.invalidate_caches()
```

CI guidance
-----------
- Do not add `tests/fixtures` to `PYTHONPATH` globally in CI jobs where
  the real ADK SDK is expected to be available; doing so will force the
  fixtures to shadow the real SDK. Only tests that intentionally use the
  shim should modify `sys.path` locally.

Local runs
----------
- To run the ADK integration test against the fixture shim locally:

```bash
PYTHONPATH=src ADK_PUBLISH_INTEGRATION=1 ADK_PROJECT=testproject pytest -q tests/test_function_tools/test_script_agent_entrypoint_adk_integration.py::test_publish_artifact_returns_artifact_uri
```

Notes
-----
- The fixtures are intentionally minimal and synchronous-friendly for
  test simplicity. They are not intended to be production replacements
  for the real ADK SDK.
