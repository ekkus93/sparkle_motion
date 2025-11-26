import os
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
    # Import the publish helper directly from the entrypoint module
    from sparkle_motion.function_tools.script_agent.entrypoint import publish_artifact

    # Create a real file to publish
    p = tmp_path / "test_art.json"
    p.write_text('{"ok": true}', encoding="utf-8")

    uri = publish_artifact(str(p))

    assert isinstance(uri, str) and uri.startswith("artifact://"), f"Expected artifact:// URI, got {uri}"
