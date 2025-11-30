from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from . import adk_helpers
from . import observability
from sparkle_motion.utils.env import fixture_mode_enabled

AgentMode = Literal["per-tool", "shared"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AdkFactoryError(RuntimeError):
    """Base error for all adk_factory failures."""

    def __init__(self, message: str, *, tool_name: str, model_spec: Optional[str], cause: Optional[Exception] = None) -> None:
        detail = message
        if cause is not None:
            detail = f"{message}: {cause}"
        super().__init__(detail)
        self.tool_name = tool_name
        self.model_spec = model_spec
        self.__cause__ = cause


class MissingAdkSdkError(AdkFactoryError):
    """Raised when the google.adk SDK is required but unavailable."""


class AdkAgentCreationError(AdkFactoryError):
    """Raised when an agent cannot be created via the SDK."""


class AdkAgentLifecycleError(AdkFactoryError):
    """Raised when we fail to close or manage an existing agent."""


@dataclass(frozen=True)
class AgentConfig:
    tool_name: str
    model_spec: Optional[str] = None
    mode: AgentMode = "per-tool"
    seed: Optional[int] = None
    metadata: Optional[Mapping[str, Any]] = None


@dataclass
class _AgentHandle:
    agent: Any
    config: AgentConfig
    fixture: bool
    created_at: datetime = field(default_factory=_utcnow)
    last_used_at: datetime = field(default_factory=_utcnow)


class _DummyAgent:
    def __init__(self, name: str, model_spec: Optional[str] = None) -> None:
        self.name = name
        self.model_spec = model_spec
        self._closed = False

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "model_spec": self.model_spec}

    def close(self) -> None:
        self._closed = True


_agents: Dict[str, _AgentHandle] = {}


def _fixture_enabled() -> bool:
    return fixture_mode_enabled()


def _agent_key(cfg: AgentConfig) -> str:
    parts = [cfg.mode, cfg.tool_name]
    if cfg.model_spec:
        parts.append(cfg.model_spec)
    return "|".join(parts)


def safe_probe_sdk() -> Optional[Tuple[object, Optional[object]]]:
    """Wrapper around adk_helpers.probe_sdk that never raises SystemExit."""

    def _log(event_type: str, payload: Mapping[str, Any]) -> None:
        _emit_memory_event(event_type, payload)

    try:
        result = adk_helpers.probe_sdk()
    except SystemExit as exc:
        _log(
            "adk_factory.sdk_probe_failure",
            {"reason": "system_exit", "message": str(exc)},
        )
        return None
    except Exception as exc:
        _log(
            "adk_factory.sdk_probe_failure",
            {"reason": "exception", "message": str(exc)},
        )
        return None

    if result is None:
        _log(
            "adk_factory.sdk_probe_failure",
            {"reason": "sdk_missing"},
        )
    return result


def require_adk(*, allow_fixture: bool = False, tool_name: str = "adk_factory", model_spec: Optional[str] = None) -> Tuple[object, Optional[object]]:
    """Require the google.adk SDK, raising MissingAdkSdkError on failure."""

    if allow_fixture and _fixture_enabled():
        _emit_memory_event(
            "adk_factory.require_adk.fixture_fallback",
            {"tool": tool_name, "model_spec": model_spec, "allow_fixture": True},
        )
        return (None, None)

    res = safe_probe_sdk()
    if res:
        return res

    _emit_memory_event(
        "adk_factory.require_adk.failure",
        {"tool": tool_name, "model_spec": model_spec},
    )
    raise MissingAdkSdkError("google.adk SDK not available", tool_name=tool_name, model_spec=model_spec)


def get_agent(
    tool_name: str,
    model_spec: Optional[str] = None,
    *,
    mode: AgentMode = "per-tool",
    seed: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Return (and optionally cache) an agent handle for the given tool."""

    cfg = AgentConfig(tool_name=tool_name, model_spec=model_spec, mode=_validate_mode(mode), seed=seed, metadata=metadata)
    key = _agent_key(cfg)
    handle = _agents.get(key)
    if handle is not None:
        handle.last_used_at = _utcnow()
        return handle.agent

    agent = create_agent(cfg)
    _agents[key] = _AgentHandle(agent=agent, config=cfg, fixture=_fixture_enabled())
    return agent


def create_agent(config: AgentConfig | None = None, **kwargs: Any) -> Any:
    """Create a new agent instance according to the provided configuration."""

    cfg = config or AgentConfig(**kwargs)

    if _fixture_enabled():
        agent = _fixture_agent(cfg)
        _record_creation_events(agent, cfg, fixture=True)
        return agent

    sdk = None
    try:
        sdk = require_adk(tool_name=cfg.tool_name, model_spec=cfg.model_spec)
    except MissingAdkSdkError as exc:
        raise exc
    except Exception as exc:  # pragma: no cover - defensive belt
        raise MissingAdkSdkError("Failed to import google.adk", tool_name=cfg.tool_name, model_spec=cfg.model_spec, cause=exc)

    try:
        agent = _construct_agent_from_sdk(sdk, cfg)
    except Exception as exc:
        raise AdkAgentCreationError("Unable to construct ADK agent", tool_name=cfg.tool_name, model_spec=cfg.model_spec, cause=exc)

    _record_creation_events(agent, cfg, fixture=False)
    return agent


def close_agent(
    identifier: Any,
    *,
    model_spec: Optional[str] = None,
    mode: AgentMode = "per-tool",
    suppress_errors: bool = True,
) -> None:
    """Close an agent by tool name (preferred) or by direct instance."""

    if isinstance(identifier, str):
        cfg = AgentConfig(tool_name=identifier, model_spec=model_spec, mode=_validate_mode(mode))
        handle = _agents.pop(_agent_key(cfg), None)
        agent = handle.agent if handle else None
        cfg = handle.config if handle else cfg
    else:
        agent = identifier
        cfg = AgentConfig(tool_name=getattr(identifier, "name", "unknown"), model_spec=model_spec, mode=_validate_mode(mode))

    if agent is None:
        return

    try:
        _shutdown_agent(agent)
        observability.emit_agent_event("agent.closed", {"tool": cfg.tool_name, "model_spec": cfg.model_spec})
    except Exception as exc:
        if suppress_errors:
            return
        raise AdkAgentLifecycleError("Failed to close agent", tool_name=cfg.tool_name, model_spec=cfg.model_spec, cause=exc)


def shutdown() -> None:
    """Close all tracked agents (best-effort)."""

    errors: list[AdkAgentLifecycleError] = []
    for key, handle in list(_agents.items()):
        try:
            _shutdown_agent(handle.agent)
            observability.emit_agent_event("agent.closed", {"tool": handle.config.tool_name, "model_spec": handle.config.model_spec})
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(
                AdkAgentLifecycleError(
                    "Failed to close agent during shutdown",
                    tool_name=handle.config.tool_name,
                    model_spec=handle.config.model_spec,
                    cause=exc,
                )
            )
        finally:
            _agents.pop(key, None)

    if errors:
        # raise only the first to keep behavior deterministic
        raise errors[0]


def _validate_mode(mode: AgentMode) -> AgentMode:
    if mode not in ("per-tool", "shared"):
        raise ValueError(f"Unsupported agent mode: {mode}")
    return mode


def _fixture_agent(cfg: AgentConfig) -> _DummyAgent:
    agent = _DummyAgent(cfg.tool_name, model_spec=cfg.model_spec)
    try:
        setattr(agent, "seed", cfg.seed)
    except Exception:
        pass
    return agent


def _construct_agent_from_sdk(sdk: Tuple[object, Optional[object]], cfg: AgentConfig) -> Any:
    adk_mod, client = sdk
    constructors = [
        getattr(adk_mod, "LlmAgent", None),
        getattr(adk_mod, "Llm", None),
        getattr(adk_mod, "Agent", None),
        getattr(adk_mod, "create_agent", None),
        getattr(adk_mod, "create_llm_agent", None),
    ]

    for ctor in [c for c in constructors if callable(c)]:
        agent = _try_agent_constructor(ctor, cfg)
        if agent is not None:
            return agent

    client_ctor = None
    if client is not None:
        client_ctor = getattr(client, "create", None) or getattr(client, "open", None)
    if callable(client_ctor):
        try:
            return client_ctor(cfg.tool_name, cfg.model_spec)
        except TypeError:
            return client_ctor(cfg.tool_name)

    raise RuntimeError("No supported agent constructor exposed by google.adk")


def _try_agent_constructor(ctor, cfg: AgentConfig) -> Any | None:
    try:
        return ctor(model=cfg.model_spec)
    except TypeError:
        pass
    try:
        return ctor(cfg.model_spec)
    except TypeError:
        pass
    try:
        return ctor(name=cfg.tool_name, model=cfg.model_spec)
    except TypeError:
        pass
    try:
        return ctor()
    except Exception:
        return None


def _record_creation_events(agent: Any, cfg: AgentConfig, *, fixture: bool) -> None:
    observability.record_seed(cfg.seed, tool_name=cfg.tool_name)
    payload = {
        "tool": cfg.tool_name,
        "model_spec": cfg.model_spec,
        "mode": cfg.mode,
        "fixture": fixture,
        "seed": cfg.seed,
    }
    if cfg.metadata:
        payload["metadata"] = dict(cfg.metadata)
    observability.emit_agent_event("agent.created", payload)

    if fixture:
        _log_fixture_bypass(cfg)


def _log_fixture_bypass(cfg: AgentConfig) -> None:
    _emit_memory_event(
        "adk_factory.fixture_agent",
        {"tool": cfg.tool_name, "model_spec": cfg.model_spec, "mode": cfg.mode},
    )


def _shutdown_agent(agent: Any) -> None:
    for name in ("close", "shutdown", "stop", "disconnect", "dispose"):
        fn = getattr(agent, name, None)
        if callable(fn):
            fn()


def _emit_memory_event(event_type: str, payload: Mapping[str, Any]) -> None:
    writer = getattr(adk_helpers, "write_memory_event", None)
    if not callable(writer):
        return

    session_getter = getattr(observability, "get_session_id", None)
    run_id = session_getter() if callable(session_getter) else None

    try:
        writer(run_id=run_id, event_type=event_type, payload=dict(payload))
    except TypeError:
        # Older helper shims may still be wired via <<kwargs>> variations; ignore failures.
        pass
    except Exception:
        pass
