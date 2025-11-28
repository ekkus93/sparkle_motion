import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_up(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def test_integration_invoke_creates_artifact(tmp_path):
    """Start the real ASGI app with uvicorn and POST /invoke to assert artifact created."""
    port = _find_free_port()
    host = "127.0.0.1"
    base = f"http://{host}:{port}"

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ADK_USE_FIXTURE"] = "1"
    env["ARTIFACTS_DIR"] = str(artifacts_dir)
    env["DETERMINISTIC"] = "1"
    # ensure app module is importable from repo
    env["PYTHONPATH"] = env.get("PYTHONPATH", "")
    if "src" not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = (env["PYTHONPATH"] + os.pathsep + "src").lstrip(os.pathsep)

    cmd = [sys.executable, "-m", "uvicorn", "sparkle_motion.function_tools.script_agent.entrypoint:app", "--host", host, "--port", str(port)]

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert _wait_up(f"{base}/health", timeout=8.0), "server did not start in time"

        # POST an invoke
        resp = httpx.post(f"{base}/invoke", json={"prompt": "integration-test"}, timeout=10.0)
        assert resp.status_code == 200, resp.text
        j = resp.json()
        assert j.get("status") == "success"
        artifact_uri = j.get("artifact_uri")
        assert artifact_uri and artifact_uri.startswith("file://")

        file_path = Path(artifact_uri.replace("file://", ""))
        assert file_path.exists()
        data = json.loads(file_path.read_text(encoding="utf-8"))
        request_payload = data.get("request") or {}
        assert request_payload.get("prompt") == "integration-test"
        plan = data.get("validated_plan") or {}
        assert isinstance(plan.get("shots"), list) and plan["shots"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
