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
# sparkle_motion