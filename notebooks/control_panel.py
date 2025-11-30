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
import json
from dataclasses import dataclass
import importlib
from functools import partial
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

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

DEFAULT_HTTP_TIMEOUT_S = 30.0


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


class ControlPanel:
    """ipywidgets-based control surface for script_agent and production_agent."""

    def __init__(
        self,
        *,
        endpoints: Optional[PanelEndpoints] = None,
        profile: str = "local-colab",
        http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
        auto_display: bool = True,
    ) -> None:
        self.endpoints = endpoints or PanelEndpoints.from_registry(profile)
        self.http_timeout_s = http_timeout_s
        self.state = PanelState()

        # Inputs
        self.title_input = widgets.Text(description="Title", placeholder="Short film title")
        self.prompt_input = widgets.Textarea(description="Prompt", placeholder="Describe the story, tone, beats…", layout=widgets.Layout(width="100%", height="120px"))
        self.plan_uri_input = widgets.Text(description="Plan URI", placeholder="file:///.../plan.json", layout=widgets.Layout(width="100%"))
        self.mode_input = widgets.Dropdown(options=[("Dry-Run", "dry"), ("Run", "run")], description="Mode", value="dry")
        self.run_id_input = widgets.Text(description="Run ID", placeholder="Autofilled after production run")
        self.stage_input = widgets.Text(description="Stage", placeholder="qa_publish (optional)")

        # Buttons
        self.generate_button = widgets.Button(description="Generate Plan", button_style="primary", tooltip="Call script_agent /invoke")
        self.run_button = widgets.Button(description="Run Production", button_style="success", tooltip="Call production_agent /invoke")
        self.pause_button = widgets.Button(description="Pause", tooltip="POST /control/pause")
        self.resume_button = widgets.Button(description="Resume", tooltip="POST /control/resume")
        self.stop_button = widgets.Button(description="Stop", button_style="danger", tooltip="POST /control/stop")
        self.poll_toggle = widgets.ToggleButton(description="Poll Status", value=False, tooltip="Toggle periodic GET /status", icon="refresh")
        self.poll_interval_input = widgets.BoundedFloatText(value=3.0, min=1.0, max=30.0, step=0.5, description="Interval (s)")
        self.auto_artifacts_checkbox = widgets.Checkbox(value=True, description="Auto-fetch artifacts")
        self.refresh_artifacts_button = widgets.Button(description="Refresh Artifacts", tooltip="GET /artifacts")

        # Outputs
        self.plan_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="80px"))
        self.production_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", min_height="120px"))
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

        controls_row = widgets.HBox([
            self.generate_button,
            self.run_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ])
        status_controls = widgets.HBox([
            self.poll_toggle,
            self.poll_interval_input,
            self.auto_artifacts_checkbox,
        ])
        artifact_controls = widgets.HBox([
            self.stage_input,
            self.refresh_artifacts_button,
        ])
        inputs = widgets.VBox([
            self.title_input,
            self.prompt_input,
            self.plan_uri_input,
            self.mode_input,
            self.run_id_input,
        ])
        outputs = widgets.VBox([
            widgets.HTML("<b>Script Agent</b>"),
            self.plan_output,
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

        try:
            response = _http_post_json(self.endpoints.script_invoke, payload, timeout_s=self.http_timeout_s)
        except httpx.HTTPError as exc:
            with self.plan_output:
                print(f"script_agent request failed: {exc}")
            return

        artifact_uri = response.get("artifact_uri")
        self.state.plan_uri = artifact_uri
        self.state.plan_payload = None  # until we parse the JSON artifact explicitly
        if artifact_uri:
            self.plan_uri_input.value = artifact_uri
        with self.plan_output:
            print("status:", response.get("status"))
            print("request_id:", response.get("request_id"))
            print("artifact_uri:", artifact_uri or "<missing>")

    def _handle_run_production(self, _: Any) -> None:
        plan_uri = _coalesce_plan_uri(self.plan_uri_input.value, self.state.plan_uri)
        payload: Dict[str, Any] = {"mode": self.mode_input.value}
        if plan_uri:
            payload["plan_uri"] = plan_uri
        elif self.state.plan_payload:
            payload["plan"] = self.state.plan_payload
        else:
            with self.production_output:
                self.production_output.clear_output()
                print("No plan URI available. Generate a plan first or paste a plan_uri.")
            return

        with self.production_output:
            self.production_output.clear_output()
            print(f"POST {self.endpoints.production_invoke} (mode={payload['mode']})")

        try:
            response = _http_post_json(self.endpoints.production_invoke, payload, timeout_s=self.http_timeout_s)
        except httpx.HTTPError as exc:
            with self.production_output:
                print(f"production_agent request failed: {exc}")
            return

        self.state.last_run_request_id = response.get("request_id")
        if self.state.last_run_request_id and not self.run_id_input.value:
            self.run_id_input.value = self.state.last_run_request_id
        steps = response.get("steps", [])
        with self.production_output:
            print("status:", response.get("status"))
            print("request_id:", self.state.last_run_request_id)
            print("artifact_uris:", response.get("artifact_uris", []))
            print(f"steps returned: {len(steps)}")

    def _handle_control_action(self, _: Any, *, action: str) -> None:
        run_id = self.run_id_input.value.strip() or self.state.last_run_request_id
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
            return

        with self.status_output:
            print(f"{action.title()} acknowledged:", response)

    # ------------------------------------------------------------------
    # Status polling & artifact helpers
    # ------------------------------------------------------------------
    def _handle_poll_toggle(self, change: Dict[str, Any]) -> None:
        if change.get("new"):
            self._start_status_polling()
        else:
            self._stop_status_polling()

    def _start_status_polling(self) -> None:
        run_id = self.run_id_input.value.strip() or self.state.last_run_request_id
        if not run_id:
            with self.status_stream_output:
                self.status_stream_output.clear_output()
                print("Provide a Run ID before starting polling.")
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
        with self.status_stream_output:
            self.status_stream_output.clear_output()
            print("Status polling stopped.")

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
                        with self.status_stream_output:
                            print(f"Status polling failed: {exc}")
                        self.poll_toggle.value = False
                        break

                    with self.status_stream_output:
                        self.status_stream_output.clear_output()
                        print(_format_status_snapshot(ready, status))

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
        run_id = self.run_id_input.value.strip() or self.state.last_run_request_id
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
            return
        self._render_artifacts(data)

    def _render_artifacts(self, payload: Dict[str, Any]) -> None:
        with self.artifacts_output:
            self.artifacts_output.clear_output()
            print(_format_json(payload))


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
    return "\n".join(blocks)


def _format_json(payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        return str(payload)


__all__ = [
    "ControlPanel",
    "EndpointResolutionError",
    "PanelEndpoints",
    "create_control_panel",
]
