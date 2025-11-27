from __future__ import annotations
import os
import json
import logging
import threading
from typing import Any, Dict
from urllib.request import Request, urlopen

LOG = logging.getLogger("function_tools.entrypoint_common")


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 1.0) -> None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        # best-effort: short timeout
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            # read to ensure request completes
            _ = resp.read(16)
    except Exception as e:
        LOG.debug("telemetry post failed: %s", e)


def send_telemetry(event: str, payload: Dict[str, Any]) -> None:
    """Send a small telemetry ping if `TELEMETRY_ENDPOINT` is set.

    This is intentionally best-effort and will not raise on failure.
    It attempts a short HTTP POST to avoid blocking tool execution.
    """
    try:
        endpoint = os.environ.get("TELEMETRY_ENDPOINT")
        if not endpoint:
            LOG.debug("telemetry disabled (no endpoint)")
            return
        body = {"event": event, "payload": payload}
        # run in background thread so we don't block runtime-critical paths
        t = threading.Thread(target=_post_json, args=(endpoint, body), kwargs={"timeout": 1.0}, daemon=True)
        t.start()
    except Exception as e:
        LOG.debug("send_telemetry error: %s", e)


__all__ = ["send_telemetry"]
