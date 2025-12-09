"""Central registry for Sparkle Motion schema artifacts.

The canonical artifact/catalog table lives in ``docs/SCHEMA_ARTIFACTS.md``. When
adding a new schema, update that document first, then ensure the corresponding
entry exists in ``configs/schema_artifacts.yaml`` so this module can surface the
URI + local fallback consistently for agents and FunctionTools.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml

from sparkle_motion.utils.env import fixture_mode_enabled


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "schema_artifacts.yaml"


@dataclass(frozen=True)
class SchemaArtifact:
    name: str
    uri: str
    local_path: Path | None
    prefer_local: bool | None = None
@dataclass(frozen=True)
class SchemaCatalog:
    """Loaded view of all schemas declared in schema_artifacts.yaml.

    Consumers should not construct this class directly; call ``load_catalog``
    instead so updates remain synchronized with ``docs/SCHEMA_ARTIFACTS.md``.
    """

    version: str
    schemas: Dict[str, SchemaArtifact]

    def list_schema_names(self) -> Iterable[str]:
        return self.schemas.keys()

    def get_schema(self, name: str) -> SchemaArtifact:
        try:
            return self.schemas[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown schema '{name}'. Known schemas: {sorted(self.schemas)}") from exc


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _should_prefer_local(prefer_local: Optional[bool]) -> Tuple[bool, bool]:
    if prefer_local is not None:
        return prefer_local, False
    fixture = fixture_mode_enabled(default=False)
    return fixture, fixture


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _resolve_reference(*, name: str, uri: str, local_path: Optional[Path], prefer_local: Optional[bool]) -> Tuple[str, bool]:
    use_local, env_forced = _should_prefer_local(prefer_local)
    if use_local and local_path is not None:
        if local_path.exists():
            if env_forced:
                _warn(f"Fixture mode enabled; using local schema fallback for '{name}' ({local_path})")
            return local_path.resolve().as_uri(), True
        _warn(
            f"Local schema path '{local_path}' for '{name}' does not exist; falling back to artifact URI {uri}"
        )
    elif use_local and local_path is None:
        _warn(f"Fixture mode enabled but schema '{name}' does not define a local_path; using artifact URI {uri}")
    return uri, False


@lru_cache(maxsize=1)
def load_catalog(config_path: Path | None = None) -> SchemaCatalog:
    """Load schema definitions from configs/schema_artifacts.yaml.

    The file layout and required keys are documented in ``docs/SCHEMA_ARTIFACTS.md``;
    consult that doc when adding new schema entries or rotating artifact URIs so
    onboarding guides and code stay aligned.
    """

    path = config_path or DEFAULT_CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    schemas: Dict[str, SchemaArtifact] = {}
    for name, entry in data["schemas"].items():
        raw_local = entry.get("local_path")
        schemas[name] = SchemaArtifact(
            name=name,
            uri=entry["uri"],
            local_path=_resolve_path(raw_local) if raw_local else None,
            prefer_local=entry.get("prefer_local"),
        )

    return SchemaCatalog(version=str(data["version"]), schemas=schemas)


def get_schema_uri(name: str) -> str:
    return load_catalog().get_schema(name).uri


def get_schema_path(name: str) -> Path:
    schema = load_catalog().get_schema(name)
    if schema.local_path is None:
        raise ValueError(
            f"Schema '{name}' does not define a local path; artifact URI {schema.uri} must be fetched via resolve_schema_uri"
        )
    resolved, _ = _resolve_reference(
        name=f"schema:{name}",
        uri=schema.uri,
        local_path=schema.local_path,
        prefer_local=True,
    )
    if not resolved.startswith("file://"):
        raise ValueError(f"Schema '{name}' resolved to non-file URI {resolved}; cannot return filesystem path")
    return Path(resolved[7:])


def resolve_schema_uri(name: str, *, prefer_local: Optional[bool] = None) -> str:
    schema = load_catalog().get_schema(name)
    resolved, _ = _resolve_reference(
        name=f"schema:{name}",
        uri=schema.uri,
        local_path=schema.local_path,
        prefer_local=prefer_local,
    )
    return resolved


def list_schema_names() -> Iterable[str]:
    return load_catalog().list_schema_names()


def movie_plan_schema() -> SchemaArtifact:
    return load_catalog().get_schema("movie_plan")


def asset_refs_schema() -> SchemaArtifact:
    return load_catalog().get_schema("asset_refs")


def stage_event_schema() -> SchemaArtifact:
    return load_catalog().get_schema("stage_event")


def checkpoint_schema() -> SchemaArtifact:
    return load_catalog().get_schema("checkpoint")


def run_context_schema() -> SchemaArtifact:
    return load_catalog().get_schema("run_context")


def stage_manifest_schema() -> SchemaArtifact:
    return load_catalog().get_schema("stage_manifest")