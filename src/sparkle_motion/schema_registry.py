from __future__ import annotations

"""Central registry for Sparkle Motion schema and policy artifacts."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import yaml

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


def list_schema_names() -> Iterable[str]:
    return load_catalog().list_schema_names()


def get_qa_policy_bundle() -> QAPolicyBundle:
    return load_catalog().qa_policy