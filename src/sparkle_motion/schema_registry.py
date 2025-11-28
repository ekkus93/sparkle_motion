from __future__ import annotations

"""Central registry for Sparkle Motion schema and policy artifacts."""

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml

_FIXTURE_ENV = "ADK_USE_FIXTURE"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "schema_artifacts.yaml"


@dataclass(frozen=True)
class SchemaArtifact:
    name: str
    uri: str
    local_path: Path


@dataclass(frozen=True)
class QAPolicyBundle:
    uri: str
    bundle_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SchemaCatalog:
    version: str
    schemas: Dict[str, SchemaArtifact]
    qa_policy: QAPolicyBundle

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


def _fixture_mode_enabled() -> bool:
    return os.environ.get(_FIXTURE_ENV) == "1"


def _should_prefer_local(prefer_local: Optional[bool]) -> Tuple[bool, bool]:
    if prefer_local is not None:
        return prefer_local, False
    fixture = _fixture_mode_enabled()
    return fixture, fixture


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _resolve_reference(*, name: str, uri: str, local_path: Path, prefer_local: Optional[bool]) -> Tuple[str, bool]:
    use_local, env_forced = _should_prefer_local(prefer_local)
    if use_local:
        if local_path.exists():
            if env_forced:
                _warn(f"Fixture mode enabled; using local schema fallback for '{name}' ({local_path})")
            return local_path.resolve().as_uri(), True
        _warn(
            f"Local schema path '{local_path}' for '{name}' does not exist; falling back to artifact URI {uri}"
        )
    return uri, False


@lru_cache(maxsize=1)
def load_catalog(config_path: Path | None = None) -> SchemaCatalog:
    path = config_path or DEFAULT_CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    schemas: Dict[str, SchemaArtifact] = {}
    for name, entry in data["schemas"].items():
        schemas[name] = SchemaArtifact(
            name=name,
            uri=entry["uri"],
            local_path=_resolve_path(entry["local_path"]),
        )

    qa_policy_entry = data["qa_policy"]
    qa_policy = QAPolicyBundle(
        uri=qa_policy_entry["uri"],
        bundle_path=_resolve_path(qa_policy_entry["bundle_path"]),
        manifest_path=_resolve_path(qa_policy_entry["manifest_path"]),
    )

    return SchemaCatalog(version=str(data["version"]), schemas=schemas, qa_policy=qa_policy)


def get_schema_uri(name: str) -> str:
    return load_catalog().get_schema(name).uri


def get_schema_path(name: str) -> Path:
    return load_catalog().get_schema(name).local_path


def resolve_schema_uri(name: str, *, prefer_local: Optional[bool] = None) -> str:
    schema = load_catalog().get_schema(name)
    resolved, _ = _resolve_reference(name=f"schema:{name}", uri=schema.uri, local_path=schema.local_path, prefer_local=prefer_local)
    return resolved


def list_schema_names() -> Iterable[str]:
    return load_catalog().list_schema_names()


def get_qa_policy_bundle() -> QAPolicyBundle:
    return load_catalog().qa_policy


def resolve_qa_policy_bundle(*, prefer_local: Optional[bool] = None) -> Tuple[str, str]:
    bundle = get_qa_policy_bundle()
    resolved_bundle, _ = _resolve_reference(
        name="qa_policy.bundle", uri=bundle.uri, local_path=bundle.bundle_path, prefer_local=prefer_local
    )
    resolved_manifest, _ = _resolve_reference(
        name="qa_policy.manifest",
        uri=bundle.uri,
        local_path=bundle.manifest_path,
        prefer_local=prefer_local,
    )
    return resolved_bundle, resolved_manifest


def movie_plan_schema() -> SchemaArtifact:
    return load_catalog().get_schema("movie_plan")


def asset_refs_schema() -> SchemaArtifact:
    return load_catalog().get_schema("asset_refs")


def qa_report_schema() -> SchemaArtifact:
    return load_catalog().get_schema("qa_report")


def stage_event_schema() -> SchemaArtifact:
    return load_catalog().get_schema("stage_event")


def checkpoint_schema() -> SchemaArtifact:
    return load_catalog().get_schema("checkpoint")