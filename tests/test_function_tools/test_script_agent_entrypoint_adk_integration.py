import os
import sys
import importlib
import pytest
from pathlib import Path

# This integration test runs only when ADK integration is explicitly enabled
# via environment variables. It will be skipped otherwise. To enable locally:
#
# export ADK_PUBLISH_INTEGRATION=1
# export ADK_PROJECT=<your-adk-project>
# (ensure the ADK SDK is installed and authenticated in the environment)

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ADK_PUBLISH_INTEGRATION") == "1" and os.environ.get("ADK_PROJECT")),
    reason="ADK integration not enabled (set ADK_PUBLISH_INTEGRATION=1 and ADK_PROJECT)",
)


def test_publish_artifact_returns_artifact_uri(tmp_path: Path):
    # If ADK integration is enabled for this run, we by default prefer the
    # test fixture shim so we don't accidentally import an installed
    # `google` SDK. To explicitly run against a real ADK SDK (if installed
    # and authenticated), set `ADK_USE_FIXTURE=0` in the environment.
    if os.environ.get("ADK_PUBLISH_INTEGRATION") == "1":
        use_fixture = os.environ.get("ADK_USE_FIXTURE", "1") != "0"
        if use_fixture:
            repo_root = Path(__file__).resolve().parents[2]
            fixtures_dir = str(repo_root / "tests" / "fixtures")
            if fixtures_dir not in sys.path:
                sys.path.insert(0, fixtures_dir)
                importlib.invalidate_caches()

    # Import the publish helper directly from the entrypoint module
    from sparkle_motion.function_tools.script_agent.entrypoint import publish_artifact

    # Create a real file to publish
    p = tmp_path / "test_art.json"
    p.write_text('{"ok": true}', encoding="utf-8")

    uri = publish_artifact(str(p))

    assert isinstance(uri, str) and uri.startswith("artifact://"), f"Expected artifact:// URI, got {uri}"
