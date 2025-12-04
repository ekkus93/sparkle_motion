"""Notebook control-panel helpers for interacting with local FunctionTools.

This module instantiates the prompt/title inputs and the Generate Plan /
Run Production / Pause / Resume / Stop buttons described in
``docs/NOTEBOOK_AGENT_INTEGRATION.md``. Import ``create_control_panel`` from a
Colab or Jupyter notebook cell to display the widgets:

```
from notebooks.control_panel import create_control_panel
panel = create_control_panel()
```

The resulting object exposes the underlying ipywidgets container via the
``.container`` attribute so callers can re-arrange it if needed.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx


def _load_widgets_module() -> Any:
    try:
        return importlib.import_module("ipywidgets")
    except ImportError as exc:  # pragma: no cover - env specific
        raise RuntimeError(
            "ipywidgets is required for the notebook control panel. Install it via 'pip install ipywidgets'."
        ) from exc


def _load_display() -> Any:
    try:
        ipy_module = importlib.import_module("IPython.display")
    except ImportError as exc:  # pragma: no cover - env specific
        raise RuntimeError(
            "IPython is required for the notebook control panel. Install it via 'pip install ipython'."
        ) from exc
    display_fn = getattr(ipy_module, "display", None)
    if display_fn is None:
        raise RuntimeError("IPython.display.display is not available in this environment.")
    return display_fn


widgets = _load_widgets_module()
display = _load_display()

from sparkle_motion import tool_registry
from notebooks import preview_helpers
from sparkle_motion.filesystem_artifacts.config import DEFAULT_ARTIFACTS_FS_BASE_URL
from sparkle_motion.utils.env import ARTIFACTS_BACKEND_FILESYSTEM, resolve_artifacts_backend

DEFAULT_HTTP_TIMEOUT_S = 30.0
_LOG_DIR_ENV = os.environ.get("CONTROL_PANEL_LOG_DIR")
DEFAULT_LOG_DIR = Path(_LOG_DIR_ENV).expanduser() if _LOG_DIR_ENV else Path(__file__).resolve().parents[1] / "artifacts" / "logs"
DEFAULT_FS_BASE_URL = os.environ.get("ARTIFACTS_FS_BASE_URL", DEFAULT_ARTIFACTS_FS_BASE_URL)



def _resolve_artifact_backend() -> str:
    try:
        return resolve_artifacts_backend(os.environ)
    except ValueError:
        return "unknown"


def _env_logging_enabled() -> bool:
    value = os.environ.get("CONTROL_PANEL_LOG", "").strip().lower()
    if value in {"0", "false", "off"}:
        return False
    if value in {"1", "true", "on"}:
        return True
    return True


class PanelLogger:
    """Append-only JSONL logger for notebook control panel events."""

    def __init__(self, *, enabled: bool, directory: Path) -> None:
        self.enabled = enabled
        self.path: Optional[Path] = None
        if not enabled:
            return
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"control_panel_{timestamp}.log"
        header = {
            "timestamp": _utc_isoformat(),
            "event": "logger.start",
            "data": {"path": str(self.path)},
        }
        self._append_line(header)

    def log(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self.path is None:
            return
        entry = {"timestamp": _utc_isoformat(), "event": event}
        if data:
            entry["data"] = data
        self._append_line(entry)

    def _append_line(self, entry: Dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=_json_fallback))
            handle.write("\n")


class EndpointResolutionError(RuntimeError):
    """Raised when a required tool endpoint is missing from the registry."""


@dataclass(frozen=True)
class PanelEndpoints:
    """Grouping of script and production agent endpoints used by the UI."""

    script_invoke: str
    production_invoke: str
    production_ready: str
    production_status: str
    production_artifacts: str
    production_control_base: str

    @classmethod
    def from_registry(cls, profile: str = "local-colab") -> "PanelEndpoints":
        script = _require_endpoint("script_agent", profile)
        production = _require_endpoint("production_agent", profile)
        production_base = _strip_invoke_suffix(production)
        return cls(
            script_invoke=script,
            production_invoke=production,
            production_ready=f"{production_base}/ready",
            production_status=f"{production_base}/status",
            production_artifacts=f"{production_base}/artifacts",
            production_control_base=f"{production_base}/control",
        )


@dataclass
class PanelState:
    """Mutable state shared across widget callbacks."""

    plan_uri: Optional[str] = None
    plan_payload: Optional[Dict[str, Any]] = None
    last_run_request_id: Optional[str] = None
    last_run_id: Optional[str] = None


class ControlPanel:
    """ipywidgets-based control surface for script_agent and production_agent."""

    def __init__(
        self,
        *,
        endpoints: Optional[PanelEndpoints] = None,
        profile: str = "local-colab",
        http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
        auto_display: bool = True,
        log_events: Optional[bool] = None,
        log_directory: Optional[Path] = None,
    ) -> None:
        self.endpoints = endpoints or PanelEndpoints.from_registry(profile)
        self.http_timeout_s = http_timeout_s
        self.state = PanelState()
        log_enabled = log_events if log_events is not None else _env_logging_enabled()
        self.logger = PanelLogger(enabled=log_enabled, directory=log_directory or DEFAULT_LOG_DIR)
        if self.logger.enabled and self.logger.path:
            print(f"[control_panel] Logging events to {self.logger.path}")
        self.artifact_backend = _resolve_artifact_backend()
        self.fs_base_url = os.environ.get("ARTIFACTS_FS_BASE_URL", DEFAULT_FS_BASE_URL)
        # Inputs
        self.title_input = widgets.Text(description="Title", placeholder="Short film title")
        self.prompt_input = widgets.Textarea(description="Prompt", placeholder="Describe the story, tone, beats…", layout=widgets.Layout(width="100%", height="120px"))
        self.plan_uri_input = widgets.Text(description="Plan URI", placeholder="file:///.../plan.json", layout=widgets.Layout(width="100%"))
        self.load_plan_button = widgets.Button(description="Load Plan JSON", tooltip="Parse plan_uri artifact")
        self.mode_input = widgets.Dropdown(options=[("Dry-Run", "dry"), ("Run", "run")], description="Mode", value="dry")
        self.run_id_input = widgets.Text(description="Run ID", placeholder="Autofilled after production run")
        self.stage_input = widgets.Text(
            description="Finalize Stage",
            placeholder="finalize",
            tooltip="Override only when inspecting a non-finalize stage",
        )
        self.inline_plan_checkbox = widgets.Checkbox(value=False, description="Send plan inline", tooltip="Embed plan JSON in production request")

        # Buttons
        self.generate_button = widgets.Button(description="Generate Plan", button_style="primary", tooltip="Call script_agent /invoke")
        self.run_button = widgets.Button(description="Run Production", button_style="success", tooltip="Call production_agent /invoke")
        self.pause_button = widgets.Button(description="Pause", tooltip="POST /control/pause")
        self.resume_button = widgets.Button(description="Resume", tooltip="POST /control/resume")
        self.stop_button = widgets.Button(description="Stop", button_style="danger", tooltip="POST /control/stop")
        self.poll_toggle = widgets.ToggleButton(description="Poll Status", value=False, tooltip="Toggle periodic GET /status", icon="refresh")
        self.poll_interval_input = widgets.BoundedFloatText(value=3.0, min=1.0, max=30.0, step=0.5, description="Interval (s)")
        self.auto_artifacts_checkbox = widgets.Checkbox(value=True, description="Auto-fetch artifacts")
        self.status_probe_label = widgets.HTML(value=_format_status_probe_label(False))
        self.status_probe_button = widgets.Button(description="Check Status", icon="search", tooltip="Probe /status availability")
        self.refresh_artifacts_button = widgets.Button(description="Refresh Artifacts", tooltip="GET /artifacts")
        self.backend_label = widgets.HTML(value=self._format_backend_badge())
        self.fs_backend_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="60px"))
        self.fs_health_button = widgets.Button(description="Check filesystem health", icon="heartbeat")
        self.fs_health_button.on_click(self._handle_fs_health_check)
        self.fs_health_button.disabled = self.artifact_backend != ARTIFACTS_BACKEND_FILESYSTEM
        with self.fs_backend_output:
            print(self._initial_backend_message())

        # Outputs
        self.plan_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="80px"))
        self.production_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="120px"))
        self.plan_details_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="80px", max_height="200px", overflow="auto"))
        self.status_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="80px"))
        self.status_stream_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="140px", max_height="200px", overflow="auto"))
        self.artifacts_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="120px", max_height="240px", overflow="auto"))

        self.generate_button.on_click(self._handle_generate_plan)
        self.run_button.on_click(self._handle_run_production)
        self.pause_button.on_click(partial(self._handle_control_action, action="pause"))
        self.resume_button.on_click(partial(self._handle_control_action, action="resume"))
        self.stop_button.on_click(partial(self._handle_control_action, action="stop"))
        self.poll_toggle.observe(self._handle_poll_toggle, names="value")
        self.refresh_artifacts_button.on_click(self._handle_refresh_artifacts)
        self.load_plan_button.on_click(self._handle_load_plan_payload)
        self.status_probe_button.on_click(self._handle_status_probe)

        controls_row = widgets.HBox([
            self.generate_button,
            self.run_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ])
        plan_controls = widgets.HBox([
            self.plan_uri_input,
            self.load_plan_button,
        ])
        run_mode_controls = widgets.HBox([
            self.mode_input,
            self.inline_plan_checkbox,
        ])
        status_controls = widgets.HBox([
            self.poll_toggle,
            self.poll_interval_input,
            self.auto_artifacts_checkbox,
            self.status_probe_label,
            self.status_probe_button,
        ])
        artifact_controls = widgets.HBox([
            self.stage_input,
            self.refresh_artifacts_button,
        ])
        backend_controls = widgets.VBox([
            widgets.HTML("<b>Artifact backend</b>"),
            widgets.HBox([self.backend_label, self.fs_health_button]),
            self.fs_backend_output,
        ])
        inputs = widgets.VBox([
            self.title_input,
            self.prompt_input,
            plan_controls,
            run_mode_controls,
            self.run_id_input,
        ])
        outputs = widgets.VBox([
            backend_controls,
            widgets.HTML("<b>Script Agent</b>"),
            self.plan_output,
            widgets.HTML("<b>Plan Details</b>"),
            self.plan_details_output,
            widgets.HTML("<b>Production Agent</b>"),
            self.production_output,
            widgets.HTML("<b>Control Responses</b>"),
            self.status_output,
            widgets.HTML("<b>Status polling</b>"),
            status_controls,
            self.status_stream_output,
            widgets.HTML("<b>Artifacts</b>"),
            artifact_controls,
            self.artifacts_output,
        ])
        self.container = widgets.VBox([inputs, controls_row, outputs])

        self._poll_task: Optional[asyncio.Task[Any]] = None
        self._status_history: Deque[str] = deque(maxlen=20)
        self._status_endpoint_ready = False
        self.poll_toggle.disabled = True

        self._probe_status_endpoint(initial=True)

        if auto_display:
            display(self.container)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _handle_generate_plan(self, _: Any) -> None:
        payload = _build_script_payload(self.prompt_input.value, self.title_input.value)
        if payload is None:
            with self.plan_output:
                self.plan_output.clear_output()
                print("Provide at least a prompt or title before calling script_agent.")
            return

        with self.plan_output:
            self.plan_output.clear_output()
            print(f"POST {self.endpoints.script_invoke}")

        self.logger.log("script.invoke.request", {"endpoint": self.endpoints.script_invoke, "payload": payload})
        try:
            response = _http_post_json(self.endpoints.script_invoke, payload, timeout_s=self.http_timeout_s)
        except httpx.HTTPError as exc:
            with self.plan_output:
                print(f"script_agent request failed: {exc}")
            self.logger.log("script.invoke.error", {"error": str(exc)})
            return

        artifact_uri = response.get("artifact_uri")
        self.state.plan_uri = artifact_uri
        self.state.plan_payload = None  # until we parse the JSON artifact explicitly
        if artifact_uri:
            self.plan_uri_input.value = artifact_uri
            self._maybe_cache_plan_from_artifact(artifact_uri)
        with self.plan_output:
            print("status:", response.get("status"))
            print("request_id:", response.get("request_id"))
            print("artifact_uri:", artifact_uri or "<missing>")
        self.logger.log(
            "script.invoke.response",
            {
                "status": response.get("status"),
                "request_id": response.get("request_id"),
                "artifact_uri": artifact_uri,
            },
        )

    def _handle_run_production(self, _: Any) -> None:
        plan_uri = _coalesce_plan_uri(self.plan_uri_input.value, self.state.plan_uri)
        plan_payload = self.state.plan_payload
        payload: Dict[str, Any] = {"mode": self.mode_input.value}

        send_inline_plan = self.inline_plan_checkbox.value
        if send_inline_plan and not plan_payload:
            with self.production_output:
                self.production_output.clear_output()
                print("Inline plan requested, but no plan payload is cached. Load the plan JSON first.")
            return

        if send_inline_plan and plan_payload:
            payload["plan"] = plan_payload
        elif plan_uri:
            payload["plan_uri"] = plan_uri
        elif plan_payload:
            payload["plan"] = plan_payload
        else:
            with self.production_output:
                self.production_output.clear_output()
                print("No plan available. Generate a plan first or load one via plan_uri.")
            return

        with self.production_output:
            self.production_output.clear_output()
            print(f"POST {self.endpoints.production_invoke} (mode={payload['mode']})")

        self.logger.log(
            "production.invoke.request",
            {
                "endpoint": self.endpoints.production_invoke,
                "mode": payload.get("mode"),
                "plan_inline": "plan" in payload,
                "plan_uri": payload.get("plan_uri"),
            },
        )
        try:
            response = _http_post_json(self.endpoints.production_invoke, payload, timeout_s=self.http_timeout_s)
        except httpx.HTTPError as exc:
            with self.production_output:
                print(f"production_agent request failed: {exc}")
            self.logger.log("production.invoke.error", {"error": str(exc)})
            return

        self.state.last_run_request_id = response.get("request_id")
        self.state.last_run_id = response.get("run_id")
        if self.state.last_run_id:
            self.run_id_input.value = self.state.last_run_id
        elif self.state.last_run_request_id and not self.run_id_input.value:
            self.run_id_input.value = self.state.last_run_request_id
        steps = response.get("steps", [])
        with self.production_output:
            print("status:", response.get("status"))
            print("run_id:", self.state.last_run_id or "<missing>")
            print("request_id:", self.state.last_run_request_id or "<missing>")
            print("artifact_uris:", response.get("artifact_uris", []))
            print(f"steps returned: {len(steps)}")
        self.logger.log(
            "production.invoke.response",
            {
                "status": response.get("status"),
                "request_id": self.state.last_run_request_id,
                "run_id": self.state.last_run_id,
                "artifact_count": len(response.get("artifact_uris", [])),
                "steps": len(steps),
            },
        )

        # Successful runs imply the status endpoint should now be reachable.
        self._probe_status_endpoint()

    def _handle_load_plan_payload(self, _: Any) -> None:
        plan_uri = _coalesce_plan_uri(self.plan_uri_input.value, self.state.plan_uri)
        if not plan_uri:
            with self.plan_details_output:
                self.plan_details_output.clear_output()
                print("Provide a Plan URI before loading the plan payload.")
            return
        try:
            payload = _load_plan_payload_from_artifact(plan_uri)
        except Exception as exc:
            with self.plan_details_output:
                self.plan_details_output.clear_output()
                print(f"Failed to load plan payload: {exc}")
            self.logger.log("plan.load.error", {"plan_uri": plan_uri, "error": str(exc)})
            return
        self.state.plan_payload = payload
        self._render_plan_summary(payload, source="Manual load")
        self.logger.log("plan.load.success", {"plan_uri": plan_uri, "shots": len(payload.get("shots", []))})

    def _handle_fs_health_check(self, _: Any) -> None:
        with self.fs_backend_output:
            self.fs_backend_output.clear_output()
            if self.artifact_backend != ARTIFACTS_BACKEND_FILESYSTEM:
                print("Filesystem backend is not active. Set ARTIFACTS_BACKEND=filesystem to enable the shim.")
                return
            url = f"{self.fs_base_url.rstrip('/')}/healthz"
            headers: Dict[str, str] = {}
            token = os.environ.get("ARTIFACTS_FS_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            try:
                response = httpx.get(url, headers=headers, timeout=self.http_timeout_s)
            except httpx.HTTPError as exc:
                print(f"Filesystem backend health probe failed: {exc}")
                return
            print(f"{url} → {response.status_code}")
            if response.text:
                print(response.text)

    def _format_backend_badge(self) -> str:
        backend = self.artifact_backend
        if backend == ARTIFACTS_BACKEND_FILESYSTEM:
            return "<span style=\"color:#1b7c1b;font-weight:bold\">Filesystem backend</span>"
        if backend == "adk":
            return "<span style=\"color:#1b4c99;font-weight:bold\">ADK backend</span>"
        return "<span style=\"color:#a94442;font-weight:bold\">Backend unset</span>"

    def _initial_backend_message(self) -> str:
        if self.artifact_backend == ARTIFACTS_BACKEND_FILESYSTEM:
            return f"Filesystem shim base URL: {self.fs_base_url}"
        if self.artifact_backend == "adk":
            return "ADK ArtifactService is active. Use the filesystem shim helpers when you need an offline backend."
        return "Artifacts backend is not configured. Export ARTIFACTS_BACKEND before launching the control panel."

    def _handle_control_action(self, _: Any, *, action: str) -> None:
        run_id = self._resolve_run_id()
        if not run_id:
            with self.status_output:
                self.status_output.clear_output()
                print("Provide a Run ID before sending control actions.")
            return

        endpoint = f"{self.endpoints.production_control_base}/{action}"
        payload = {"run_id": run_id}
        with self.status_output:
            self.status_output.clear_output()
            print(f"POST {endpoint}")

        try:
            response = _http_post_json(endpoint, payload, timeout_s=self.http_timeout_s)
        except httpx.HTTPError as exc:
            with self.status_output:
                print(f"{action.title()} failed: {exc}")
            self.logger.log("production.control.error", {"action": action, "error": str(exc)})
            return

        with self.status_output:
            print(f"{action.title()} acknowledged:", response)
        self.logger.log("production.control.response", {"action": action, "response": response})

    # ------------------------------------------------------------------
    # Status polling & artifact helpers
    # ------------------------------------------------------------------
    def _handle_poll_toggle(self, change: Dict[str, Any]) -> None:
        if change.get("new"):
            self._start_status_polling()
        else:
            self._stop_status_polling()

    def _start_status_polling(self) -> None:
        if not self._status_endpoint_ready:
            self._probe_status_endpoint()
        if not self._status_endpoint_ready:
            self._append_status_message("Status endpoint is offline; polling will start once it becomes available.")
            self.poll_toggle.value = False
            return
        run_id = self._resolve_run_id()
        if not run_id:
            self._append_status_message("Provide a Run ID before starting polling.")
            self.poll_toggle.value = False
            return
        loop = asyncio.get_event_loop()
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        self._poll_task = loop.create_task(self._poll_status_loop(run_id))

    def _stop_status_polling(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        self._poll_task = None
        self.poll_toggle.value = False
        self._append_status_message("Status polling stopped.")

    async def _poll_status_loop(self, run_id: str) -> None:
        interval = max(1.0, float(self.poll_interval_input.value or 3.0))
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout_s) as client:
                while self.poll_toggle.value:
                    try:
                        ready = await _http_get_json_async(client, self.endpoints.production_ready)
                        status = await _http_get_json_async(
                            client,
                            self.endpoints.production_status,
                            params={"run_id": run_id},
                        )
                    except httpx.HTTPError as exc:
                        self._append_status_message(f"Status polling failed: {exc}")
                        self.poll_toggle.value = False
                        break

                    self._record_status_snapshot(ready, status)

                    if self.auto_artifacts_checkbox.value:
                        await self._fetch_artifacts_async(client, run_id)

                    await asyncio.sleep(interval)
        except asyncio.CancelledError:  # pragma: no cover - notebook runtime only
            with self.status_stream_output:
                print("Status polling cancelled.")
        finally:
            self._poll_task = None
            self.poll_toggle.value = False

    async def _fetch_artifacts_async(self, client: httpx.AsyncClient, run_id: str) -> None:
        params = {"run_id": run_id}
        stage = self.stage_input.value.strip()
        if stage:
            params["stage"] = stage
        try:
            artifacts = await _http_get_json_async(client, self.endpoints.production_artifacts, params=params)
        except httpx.HTTPError as exc:
            with self.artifacts_output:
                self.artifacts_output.clear_output()
                print(f"Artifacts fetch failed: {exc}")
            return
        self._render_artifacts(artifacts)

    def _handle_refresh_artifacts(self, _: Any) -> None:
        run_id = self._resolve_run_id()
        if not run_id:
            with self.artifacts_output:
                self.artifacts_output.clear_output()
                print("Provide a Run ID before fetching artifacts.")
            return
        params = {"run_id": run_id}
        stage = self.stage_input.value.strip()
        if stage:
            params["stage"] = stage
        try:
            data = _http_get_json(self.endpoints.production_artifacts, params=params, timeout_s=self.http_timeout_s)
        except httpx.HTTPError as exc:
            with self.artifacts_output:
                self.artifacts_output.clear_output()
                print(f"Artifacts fetch failed: {exc}")
            self.logger.log("artifacts.fetch.error", {"params": params, "error": str(exc)})
            return
        self._render_artifacts(data)
        self.logger.log("artifacts.fetch.success", {"params": params, "artifact_keys": list(data.keys())})

    def _render_artifacts(self, payload: Dict[str, Any]) -> None:
        with self.artifacts_output:
            self.artifacts_output.clear_output()
            stages = payload.get("stages") or []
            if not stages:
                print(_format_json(payload))
                return

            for idx, stage_manifest in enumerate(stages, start=1):
                stage_name = stage_manifest.get("stage") or stage_manifest.get("name") or f"stage_{idx}"
                print(f"Stage: {stage_name}")
                print(preview_helpers.render_stage_summary(stage_manifest))
                preview_helpers.display_artifact_previews(
                    stage_manifest,
                    max_items=None,
                    image_width=320,
                    video_width=640,
                    autoplay_audio=False,
                )
                print("\n" + "-" * 40)

    def _maybe_cache_plan_from_artifact(self, artifact_uri: Optional[str]) -> None:
        if not artifact_uri:
            return
        try:
            payload = _load_plan_payload_from_artifact(artifact_uri)
        except Exception as exc:
            with self.plan_details_output:
                self.plan_details_output.clear_output()
                print(f"Plan artifact parse failed: {exc}")
            return
        self.state.plan_payload = payload
        self._render_plan_summary(payload, source="Loaded from artifact")

    def _render_plan_summary(self, plan_payload: Dict[str, Any], *, source: str) -> None:
        summary = _summarize_plan(plan_payload, source=source)
        with self.plan_details_output:
            self.plan_details_output.clear_output()
            print(summary)

    def _handle_status_probe(self, _: Any) -> None:
        self._probe_status_endpoint()

    def _probe_status_endpoint(self, *, initial: bool = False) -> None:
        available = _status_endpoint_available(self.endpoints.production_status, timeout_s=min(self.http_timeout_s, 5.0))
        if available == self._status_endpoint_ready and not initial:
            return
        self._status_endpoint_ready = available
        self.poll_toggle.disabled = not available
        self._update_status_probe_label(available)
        message = "Status endpoint detected. Toggle to begin polling." if available else "Status endpoint unavailable. Start the production agent, then press 'Check Status'."
        self._append_status_message(message)

    def _update_status_probe_label(self, available: bool) -> None:
        self.status_probe_label.value = _format_status_probe_label(available)

    def _record_status_snapshot(self, ready: Dict[str, Any], status: Dict[str, Any]) -> None:
        block = f"[{_utc_timestamp()}]\n{_format_status_snapshot(ready, status)}"
        self._status_history.append(block)
        self._render_status_history()

    def _append_status_message(self, message: str) -> None:
        entry = f"[{_utc_timestamp()}] {message}"
        self._status_history.append(entry)
        self._render_status_history()

    def _render_status_history(self) -> None:
        with self.status_stream_output:
            self.status_stream_output.clear_output()
            for idx, entry in enumerate(self._status_history):
                print(entry)
                if idx < len(self._status_history) - 1:
                    print("-" * 40)

    def _resolve_run_id(self) -> Optional[str]:
        value = self.run_id_input.value.strip()
        if value:
            return value
        if self.state.last_run_id:
            return self.state.last_run_id
        return self.state.last_run_request_id


def create_control_panel(
    *,
    endpoints: Optional[PanelEndpoints] = None,
    profile: str = "local-colab",
    http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
    auto_display: bool = True,
) -> ControlPanel:
    """Factory helper used by notebooks to build the control panel."""

    return ControlPanel(
        endpoints=endpoints,
        profile=profile,
        http_timeout_s=http_timeout_s,
        auto_display=auto_display,
    )


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _require_endpoint(tool_id: str, profile: str) -> str:
    endpoint = tool_registry.get_local_endpoint(tool_id, profile)
    if not endpoint:
        raise EndpointResolutionError(f"No endpoint found for '{tool_id}' (profile='{profile}') in configs/tool_registry.yaml")
    return endpoint


def _strip_invoke_suffix(url: str) -> str:
    if url.endswith("/invoke"):
        return url[: -len("/invoke")]
    parsed = urlsplit(url)
    if parsed.path in {"", "/"}:
        return url.rstrip("/")
    trimmed = parsed._replace(path="", query="", fragment="")
    return urlunsplit(trimmed).rstrip("/")


def _build_script_payload(prompt: str, title: str) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    prompt_value = prompt.strip()
    title_value = title.strip()
    if prompt_value:
        payload["prompt"] = prompt_value
    if title_value:
        payload["title"] = title_value
    return payload or None


def _coalesce_plan_uri(by_widget: str, cached: Optional[str]) -> Optional[str]:
    value = (by_widget or "").strip()
    if value:
        return value
    return cached


def _load_plan_payload_from_artifact(artifact_uri: str) -> Dict[str, Any]:
    if not artifact_uri:
        raise ValueError("artifact_uri is empty")
    parsed = urlparse(artifact_uri)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"Unsupported artifact scheme '{parsed.scheme}'. Provide a local file path or file:// URI.")
    path_value = parsed.path if parsed.scheme else artifact_uri
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Plan artifact path not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plan artifact JSON decode failed: {exc}") from exc
    plan_payload = payload.get("validated_plan") or payload.get("plan") or payload
    if not isinstance(plan_payload, dict):
        raise ValueError("Plan artifact missing 'validated_plan' dictionary payload")
    return plan_payload


def _summarize_plan(plan_payload: Dict[str, Any], *, source: str) -> str:
    title = plan_payload.get("title") or "<untitled>"
    metadata = plan_payload.get("metadata") if isinstance(plan_payload.get("metadata"), dict) else {}
    plan_id = metadata.get("plan_id") or metadata.get("id") or "<unknown>"
    shots = plan_payload.get("shots") if isinstance(plan_payload.get("shots"), list) else []
    base_images = plan_payload.get("base_images") if isinstance(plan_payload.get("base_images"), list) else []
    dialogue = plan_payload.get("dialogue_timeline") if isinstance(plan_payload.get("dialogue_timeline"), list) else []
    return (
        f"{source}: title='{title}' | plan_id={plan_id} | shots={len(shots)} | "
        f"base_images={len(base_images)} | dialogue_entries={len(dialogue)}"
    )


def _http_post_json(url: str, payload: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]:
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise httpx.HTTPError(f"Non-JSON response from {url}")
        return data


def _http_get_json(url: str, *, params: Optional[Dict[str, Any]] = None, timeout_s: float) -> Dict[str, Any]:
    with httpx.Client(timeout=timeout_s) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise httpx.HTTPError(f"Non-JSON response from {url}")
        return data


async def _http_get_json_async(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise httpx.HTTPError(f"Non-JSON response from {url}")
    return data


def _format_status_snapshot(ready: Dict[str, Any], status: Dict[str, Any]) -> str:
    blocks = ["/ready:", _format_json(ready), "\n/status:", _format_json(status)]
    finalize_summary = _summarize_finalize_timeline(status)
    if finalize_summary:
        blocks.extend(["\nFinalize:", finalize_summary])
    control_summary = _summarize_control_state(status)
    if control_summary:
        blocks.extend(["\nControl:", control_summary])
    return "\n".join(blocks)

def _summarize_finalize_timeline(status_payload: Dict[str, Any]) -> Optional[str]:
    timeline = status_payload.get("timeline")
    if not isinstance(timeline, list):
        return None
    finalize_entries: List[Dict[str, Any]] = []
    for entry in timeline:
        if not isinstance(entry, dict):
            continue
        meta = entry.get("meta") or {}
        stage = (entry.get("stage") or meta.get("stage") or "").lower()
        step_id = (entry.get("step_id") or "").lower()
        if stage == "finalize" or step_id.startswith("finalize"):
            finalize_entries.append(entry)
    if not finalize_entries:
        return None
    latest = finalize_entries[-1]
    summary_parts: List[str] = []
    status_value = latest.get("status") or latest.get("state")
    if status_value:
        summary_parts.append(f"status={status_value}")
    meta = latest.get("meta") or {}
    artifact_uri = meta.get("artifact_uri")
    if artifact_uri:
        summary_parts.append(f"artifact_uri={artifact_uri}")
    return " | ".join(summary_parts) if summary_parts else None


def _summarize_control_state(status_payload: Dict[str, Any]) -> Optional[str]:
    control = status_payload.get("control")
    if not isinstance(control, dict):
        return None
    parts: List[str] = []
    if control.get("pause_requested"):
        parts.append("pause_requested=True")
    if control.get("stop_requested"):
        parts.append("stop_requested=True")
    last_command = control.get("last_command")
    if isinstance(last_command, dict):
        command = last_command.get("command")
        timestamp = last_command.get("at")
        if command:
            when = f" at {timestamp}" if timestamp else ""
            parts.append(f"last_command={command}{when}")
    return "; ".join(parts) if parts else None


def _format_json(payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        return str(payload)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def _utc_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status_endpoint_available(url: str, *, timeout_s: float) -> bool:
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(url, params={"run_id": "__probe__"})
    except httpx.RequestError:
        return False
    return response.status_code < 500


def _format_status_probe_label(available: bool) -> str:
    color = "#3c763d" if available else "#d9534f"
    text = "Status endpoint ready" if available else "Status endpoint unavailable"
    return f"<span style='color:{color}; font-size:0.85em;'>{text}</span>"


def _json_fallback(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    return repr(value)


__all__ = [
    "ControlPanel",
    "EndpointResolutionError",
    "PanelEndpoints",
    "create_control_panel",
]
