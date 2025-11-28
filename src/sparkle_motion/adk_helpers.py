from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple

from typing_extensions import Literal, TypedDict

from . import telemetry, schema_registry


class ArtifactPublishError(RuntimeError):
    """Raised when artifact publishing fails even after fallbacks."""

    def __init__(self, message: str, *, path: Path, artifact_type: str, cause: Optional[BaseException] = None) -> None:
        detail = f"{message}: {cause}" if cause else message
        super().__init__(detail)
        self.path = str(path)
        self.artifact_type = artifact_type
        if cause is not None:
            self.__cause__ = cause


class HumanInputRequestError(RuntimeError):
    """Raised when the platform rejects or cannot create a review task."""

    def __init__(self, message: str, *, run_id: str, reason: str, cause: Optional[BaseException] = None) -> None:
        detail = f"{message}: {cause}" if cause else message
        super().__init__(detail)
        self.run_id = run_id
        self.reason = reason
        if cause is not None:
            self.__cause__ = cause


class SchemaRegistryError(RuntimeError):
    """Raised when schema artifact metadata cannot be loaded."""

    def __init__(self, message: str, *, config_path: Path, cause: Optional[BaseException] = None) -> None:
        detail = f"{message}: {cause}" if cause else message
        super().__init__(detail)
        self.config_path = str(config_path)
        if cause is not None:
            self.__cause__ = cause


class ArtifactRef(TypedDict, total=False):
    uri: str
    storage: Literal["adk", "local"]
    artifact_type: str
    media_type: Optional[str]
    metadata: dict[str, Any]
    run_id: Optional[str]


@dataclass(frozen=True)
class HelperBackend:
    """Test hook for overriding helper behavior."""

    publish: Optional[Callable[..., ArtifactRef]] = None
    publish_local: Optional[Callable[..., ArtifactRef]] = None
    request_human_input: Optional[Callable[..., str]] = None
    schema_loader: Optional[Callable[[Optional[Path]], schema_registry.SchemaCatalog]] = None
    memory_service_factory: Optional[Callable[[], object]] = None


_BACKEND_STACK: list[HelperBackend] = []
_DEFAULT_BACKEND = HelperBackend()
_FIXTURE_HUMAN_TASKS: list[dict[str, Any]] = []


def _current_backend() -> HelperBackend:
    return _BACKEND_STACK[-1] if _BACKEND_STACK else _DEFAULT_BACKEND


@contextmanager
def set_backend(backend: HelperBackend):
    """Temporarily override helper implementations (primarily for tests)."""

    _BACKEND_STACK.append(backend)
    try:
        yield
    finally:
        _BACKEND_STACK.pop()


def _generate_run_id() -> str:
    return os.environ.get("SPARKLE_RUN_ID") or f"local-{uuid.uuid4().hex}"


def _guess_media_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_runs_root() -> Path:
    override = os.environ.get("SPARKLE_LOCAL_RUNS_ROOT")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "artifacts" / "runs"


def _copy_local_artifact(*, source: Path, run_id: str, artifact_type: str) -> Path:
    import shutil

    dest_root = _resolve_runs_root() / run_id
    dest_root.mkdir(parents=True, exist_ok=True)
    checksum = _hash_file(source)[:12]
    suffix = source.suffix or ".bin"
    dest_name = f"{artifact_type}-{checksum}{suffix}"
    dest_path = dest_root / dest_name
    shutil.copy2(source, dest_path)
    return dest_path


def _record_memory_event(event_type: str, run_id: Optional[str], payload: MutableMapping[str, Any]) -> None:
    try:
        write_memory_event(run_id=run_id, event_type=event_type, payload=dict(payload))
    except MemoryWriteError:
        pass


def _build_artifact_metadata(
    *,
    path: Path,
    artifact_type: str,
    media_type: Optional[str],
    metadata: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    base = dict(metadata or {})
    base.setdefault("artifact_type", artifact_type)
    base.setdefault("media_type", media_type or _guess_media_type(path))
    stats = path.stat()
    base.setdefault("size_bytes", stats.st_size)
    base.setdefault("modified_ts", int(stats.st_mtime))
    base.setdefault("source_path", str(path))
    base.setdefault("checksum_sha256", _hash_file(path))
    return base


def _payload_to_bytes(payload: bytes | str | Mapping[str, Any]) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _finalize_artifact_ref(
    ref: Mapping[str, Any] | None,
    *,
    artifact_type: str,
    metadata: dict[str, Any],
    run_id: str,
) -> ArtifactRef:
    if ref is None:
        raise ArtifactPublishError("Missing artifact reference from publisher", path=Path(metadata.get("source_path", "")), artifact_type=artifact_type)

    uri = ref.get("uri") if isinstance(ref, Mapping) else None
    if not uri:
        raise ArtifactPublishError("Publisher did not return an artifact URI", path=Path(metadata.get("source_path", "")), artifact_type=artifact_type)

    storage = ref.get("storage") if isinstance(ref, Mapping) else None
    if not storage:
        storage = "adk" if str(uri).startswith("artifact://") else "local"

    ref_metadata = dict(ref.get("metadata", metadata) if isinstance(ref, Mapping) else metadata)
    ref_metadata.setdefault("artifact_type", artifact_type)
    ref_metadata.setdefault("media_type", metadata.get("media_type"))

    result: ArtifactRef = {
        "uri": str(uri),
        "storage": storage,  # type: ignore[assignment]
        "artifact_type": artifact_type,
        "media_type": ref.get("media_type") if isinstance(ref, Mapping) else metadata.get("media_type"),
        "metadata": ref_metadata,
        "run_id": ref.get("run_id") if isinstance(ref, Mapping) else run_id,
    }
    if result["run_id"] is None:
        result["run_id"] = run_id
    return result


def _extract_task_id(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, str) and response.strip():
        return response.strip()
    if isinstance(response, Mapping):
        for key in ("task_id", "id", "uri", "review_id"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if hasattr(response, "task_id"):
        value = getattr(response, "task_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if hasattr(response, "id"):
        value = getattr(response, "id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _emit_publish_completed(result: ArtifactRef, *, source: str) -> None:
    try:
        telemetry.emit_event(
            "adk_helpers.publish_artifact.completed",
            {
                "uri": result["uri"],
                "storage": result["storage"],
                "artifact_type": result["artifact_type"],
                "source": source,
            },
        )
    except Exception:
        pass


def _invoke_human_input_sdk(adk_module: object, payload: dict[str, Any], *, dry_run: bool) -> Optional[str]:
    hubs = (
        "reviews",
        "review_service",
        "human_input",
        "human_input_service",
        "event_actions",
    )
    methods = ("request", "request_input", "request_review", "create", "open")

    for hub_name in hubs:
        hub = getattr(adk_module, hub_name, None)
        if hub is None:
            continue
        for method_name in methods:
            fn = getattr(hub, method_name, None)
            if fn is None:
                continue
            if dry_run:
                print(f"[dry-run] SDK would call {hub_name}.{method_name} for reason={payload.get('reason')}")
                return f"dry-run://human-task/{payload.get('run_id') or 'unknown'}"
            try:
                try:
                    res = fn(payload)
                except TypeError:
                    try:
                        res = fn(**payload)
                    except TypeError:
                        res = fn(payload.get("run_id"), payload.get("reason"), payload.get("artifact_uri"), payload.get("metadata"))
            except Exception:
                continue
            task_id = _extract_task_id(res)
            if task_id:
                return task_id
    return None


class MemoryWriteError(RuntimeError):
    """Raised when write_memory_event cannot persist the event."""

    def __init__(self, message: str, *, run_id: Optional[str], event_type: str, cause: Optional[BaseException] = None) -> None:
        detail = f"{message}: {cause}" if cause else message
        super().__init__(detail)
        self.run_id = run_id
        self.event_type = event_type
        if cause is not None:
            self.__cause__ = cause


def probe_sdk() -> Optional[Tuple[object, Optional[object]]]:
    """Try to import google.adk and discover an artifacts client.

    Returns `(adk_module, client_candidate)` when the SDK is available or
    `None` when the SDK import fails. This helper is intentionally
    non-fatal so callers can choose whether to fail loudly or fall back
    to CLI/local behaviour.
    """
    try:
        import google.adk as adk  # type: ignore
    except Exception:
        return None

    for cand in ("artifacts", "ArtifactService", "artifacts_client", "artifact_client"):
        client = getattr(adk, cand, None)
        if client is not None:
            return adk, client

    return adk, None


def require_adk() -> Tuple[object, Optional[object]]:
    """Require the ADK SDK and return (adk_module, client).

    Raises SystemExit(1) with a clear message when the SDK is not present.
    Callers that must fail-fast (e.g., runtime entrypoints) should use this
    wrapper instead of directly calling `probe_sdk()`.
    """
    res = probe_sdk()
    if not res:
        print("ERROR: google.adk SDK not importable; see README for installation.", file=sys.stderr)
        raise SystemExit(1)
    return res


def publish_artifact(
    *,
    local_path: str | Path,
    artifact_type: str,
    metadata: Optional[Mapping[str, Any]] = None,
    media_type: Optional[str] = None,
    run_id: Optional[str] = None,
    dry_run: bool = False,
    project: Optional[str] = None,
) -> ArtifactRef:
    """Publish an artifact through SDK/CLI with local fallback."""

    path = Path(local_path)
    if not path.exists():
        raise ArtifactPublishError("Artifact path does not exist", path=path, artifact_type=artifact_type)
    if not path.is_file():
        raise ArtifactPublishError("Artifact path is not a file", path=path, artifact_type=artifact_type)

    resolved_run = run_id or _generate_run_id()
    metadata_dict = _build_artifact_metadata(path=path, artifact_type=artifact_type, media_type=media_type, metadata=metadata)
    metadata_dict.setdefault("fixture_mode", os.environ.get("ADK_USE_FIXTURE") == "1")

    backend = _current_backend().publish
    if backend:
        ref = backend(
            local_path=path,
            artifact_type=artifact_type,
            metadata=metadata_dict,
            media_type=metadata_dict.get("media_type"),
            run_id=resolved_run,
            dry_run=dry_run,
            project=project,
        )
        result = _finalize_artifact_ref(ref, artifact_type=artifact_type, metadata=metadata_dict, run_id=resolved_run)
        _record_memory_event(
            "adk_helpers.publish_artifact",
            resolved_run,
            {"uri": result["uri"], "storage": result["storage"], "metadata": result["metadata"]},
        )
        _emit_publish_completed(result, source="backend")
        return result

    fixture_mode = os.environ.get("ADK_USE_FIXTURE") == "1"

    if dry_run and not fixture_mode:
        uri = f"dry-run://artifact/{artifact_type}/{uuid.uuid4().hex[:8]}"
        result: ArtifactRef = {
            "uri": uri,
            "storage": "adk",
            "artifact_type": artifact_type,
            "media_type": metadata_dict.get("media_type"),
            "metadata": metadata_dict,
            "run_id": resolved_run,
        }
        _record_memory_event(
            "adk_helpers.publish_artifact",
            resolved_run,
            {"uri": result["uri"], "storage": result["storage"], "metadata": metadata_dict},
        )
        _emit_publish_completed(result, source="dry-run")
        return result

    publish_errors: list[Exception] = []

    if not fixture_mode:
        sdk_probed = probe_sdk()
        if sdk_probed:
            adk_module, client = sdk_probed
            try:
                uri = publish_with_sdk(adk_module, client, str(path), artifact_type, dry_run, project)
                if uri:
                    result: ArtifactRef = {
                        "uri": uri,
                        "storage": "adk",
                        "artifact_type": artifact_type,
                        "media_type": metadata_dict.get("media_type"),
                        "metadata": metadata_dict,
                        "run_id": resolved_run,
                    }
                    _record_memory_event(
                        "adk_helpers.publish_artifact",
                        resolved_run,
                        {"uri": uri, "storage": "adk", "metadata": metadata_dict},
                    )
                    _emit_publish_completed(result, source="sdk")
                    return result
            except Exception as exc:  # pragma: no cover - defensive
                publish_errors.append(exc)

        try:
            uri = publish_with_cli(str(path), artifact_type, project, dry_run, artifact_map=None)
            if uri:
                result = {
                    "uri": uri,
                    "storage": "adk",
                    "artifact_type": artifact_type,
                    "media_type": metadata_dict.get("media_type"),
                    "metadata": metadata_dict,
                    "run_id": resolved_run,
                }
                _record_memory_event(
                    "adk_helpers.publish_artifact",
                    resolved_run,
                    {"uri": uri, "storage": "adk", "metadata": metadata_dict},
                )
                _emit_publish_completed(result, source="cli")
                return result
        except Exception as exc:  # pragma: no cover - defensive
            publish_errors.append(exc)

    # Local fallback for fixture mode or when publishing services unavailable
    try:
        if dry_run:
            uri = f"dry-run://local/{artifact_type}/{uuid.uuid4().hex[:8]}"
            result = {
                "uri": uri,
                "storage": "local",
                "artifact_type": artifact_type,
                "media_type": metadata_dict.get("media_type"),
                "metadata": metadata_dict,
                "run_id": resolved_run,
            }
        else:
            local_dest = _copy_local_artifact(source=path, run_id=resolved_run, artifact_type=artifact_type)
            metadata_dict["local_path"] = str(local_dest)
            result = {
                "uri": local_dest.as_uri(),
                "storage": "local",
                "artifact_type": artifact_type,
                "media_type": metadata_dict.get("media_type"),
                "metadata": metadata_dict,
                "run_id": resolved_run,
            }
    except Exception as exc:
        raise ArtifactPublishError("Failed to persist artifact locally", path=path, artifact_type=artifact_type, cause=exc)

    _record_memory_event(
        "adk_helpers.publish_artifact",
        resolved_run,
        {"uri": result["uri"], "storage": result["storage"], "metadata": metadata_dict, "fallback": True},
    )
    if publish_errors and not fixture_mode:
        telemetry.emit_event(
            "adk_helpers.publish_artifact.fallback",
            {"errors": [str(e) for e in publish_errors], "storage": "local", "artifact_type": artifact_type},
        )
    _emit_publish_completed(result, source="local")
    return result


def publish_local(
    *,
    payload: bytes | str | Mapping[str, Any],
    artifact_type: str,
    suffix: str = ".json",
    metadata: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
) -> ArtifactRef:
    """Persist payload locally for fixture/testing scenarios."""

    resolved_run = run_id or _generate_run_id()
    dest_root = _resolve_runs_root() / resolved_run
    dest_root.mkdir(parents=True, exist_ok=True)
    suffix = suffix or ".bin"
    dest_path = dest_root / f"{artifact_type}-{uuid.uuid4().hex}{suffix}"
    data = _payload_to_bytes(payload)
    dest_path.write_bytes(data)

    meta = dict(metadata or {})
    meta.setdefault("artifact_type", artifact_type)
    meta.setdefault("media_type", mimetypes.guess_type(str(dest_path))[0] or "application/octet-stream")
    meta.setdefault("size_bytes", len(data))
    meta.setdefault("source_path", str(dest_path))
    meta.setdefault("checksum_sha256", _hash_file(dest_path))
    meta.setdefault("fixture_mode", True)

    result: ArtifactRef = {
        "uri": dest_path.as_uri(),
        "storage": "local",
        "artifact_type": artifact_type,
        "media_type": meta.get("media_type"),
        "metadata": meta,
        "run_id": resolved_run,
    }
    _record_memory_event(
        "adk_helpers.publish_local",
        resolved_run,
        {"uri": result["uri"], "storage": "local", "metadata": meta},
    )
    _emit_publish_completed(result, source="local-helper")
    return result


def request_human_input(
    *,
    run_id: Optional[str],
    reason: str,
    artifact_uri: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    dry_run: bool = False,
) -> str:
    """Request human input via SDK/fixture and record the decision."""

    if not reason:
        raise ValueError("reason is required")

    resolved_run = run_id or _generate_run_id()
    payload = {
        "run_id": resolved_run,
        "reason": reason,
        "artifact_uri": artifact_uri,
        "metadata": dict(metadata or {}),
    }

    backend = _current_backend().request_human_input
    if backend:
        task_id = backend(payload=payload, dry_run=dry_run)
        if not task_id:
            raise HumanInputRequestError("Backend override did not return a task id", run_id=resolved_run, reason=reason)
        _record_memory_event(
            "adk_helpers.request_human_input",
            resolved_run,
            {"task_id": task_id, "reason": reason, "artifact_uri": artifact_uri, "metadata": payload["metadata"]},
        )
        return task_id

    fixture_mode = os.environ.get("ADK_USE_FIXTURE") == "1"
    if fixture_mode or dry_run:
        task_id = f"fixture-review-{uuid.uuid4().hex[:8]}"
        record = {"task_id": task_id, **payload}
        _FIXTURE_HUMAN_TASKS.append(record)
        _record_memory_event(
            "adk_helpers.request_human_input",
            resolved_run,
            record,
        )
        return task_id

    sdk_res = probe_sdk()
    if not sdk_res:
        raise HumanInputRequestError("ADK SDK is required to request human input", run_id=resolved_run, reason=reason)
    adk_module, _ = sdk_res

    task_id = _invoke_human_input_sdk(adk_module, payload, dry_run=dry_run)
    if not task_id:
        raise HumanInputRequestError("No SDK method accepted the human-input payload", run_id=resolved_run, reason=reason)

    _record_memory_event(
        "adk_helpers.request_human_input",
        resolved_run,
        {"task_id": task_id, "reason": reason, "artifact_uri": artifact_uri, "metadata": payload["metadata"]},
    )
    return task_id


def ensure_schema_artifacts(config_path: Optional[Path] = None) -> schema_registry.SchemaCatalog:
    """Load schema artifacts config, honoring backend overrides."""

    backend = _current_backend().schema_loader
    path = Path(config_path) if config_path is not None else None
    if backend:
        catalog = backend(path)
        if catalog is None:
            raise SchemaRegistryError("Backend override did not return a SchemaCatalog", config_path=path or schema_registry.DEFAULT_CONFIG_PATH)
        return catalog

    try:
        return schema_registry.load_catalog(path)
    except FileNotFoundError as exc:
        raise SchemaRegistryError("Schema artifact config not found", config_path=path or schema_registry.DEFAULT_CONFIG_PATH, cause=exc) from exc
    except Exception as exc:
        raise SchemaRegistryError("Failed to load schema artifact config", config_path=path or schema_registry.DEFAULT_CONFIG_PATH, cause=exc) from exc


def _publish_with_sdk_service(adk_module, file_path: str, artifact_name: str, dry_run: bool, project: Optional[str] = None) -> Optional[str]:
    try:
        bucket = os.environ.get("ADK_ARTIFACTS_GCS_BUCKET")
        if bucket:
            from google.adk.artifacts.gcs_artifact_service import GcsArtifactService as _Gcs

            svc = _Gcs(bucket)
        else:
            from google.adk.artifacts.file_artifact_service import FileArtifactService as _FileSvc

            root = os.environ.get("ADK_ARTIFACTS_ROOT", "artifacts/adk")
            svc = _FileSvc(root)

        try:
            from google.genai.types import Part
        except Exception:
            types_mod = getattr(adk_module, "artifacts", None)
            Part = None
            if types_mod is not None:
                try:
                    from google.genai.types import Part  # try once more
                except Exception:
                    Part = None

        if dry_run:
            print(f"[dry-run] SDK would save artifact via {svc.__class__.__name__}: file={file_path}, name={artifact_name}")
            return f"artifact://{project or artifact_name}/schemas/{artifact_name}/v1"

        if Part is None:
            print("Unable to locate Part type for SDK publish", file=sys.stderr)
            return None

        file_uri = str(Path(file_path).resolve())
        try:
            mime_type, _ = mimetypes.guess_type(file_uri)
            mime_type = mime_type or "application/octet-stream"
            with open(file_path, "rb") as _fh:
                data = _fh.read()
            part = Part.from_bytes(data=data, mime_type=mime_type)
        except Exception:
            try:
                part = Part.from_uri(file_uri=file_uri)
            except Exception:
                raise

        app_name = project or os.environ.get("ADK_PROJECT") or Path.cwd().name
        user_id = getpass.getuser() or "user"

        try:
            rev = asyncio.run(svc.save_artifact(app_name=app_name, user_id=user_id, filename=artifact_name, artifact=part))
        except TypeError:
            rev = asyncio.run(svc.save_artifact(app_name=app_name, user_id=user_id, filename=artifact_name, artifact=part))

        return f"artifact://{app_name}/schemas/{artifact_name}/v{rev}"
    except Exception as e:
        print(f"SDK artifact-service publish failed: {e}", file=sys.stderr)
        return None


def publish_with_sdk(adk_module, client, file_path: str, artifact_name: str, dry_run: bool, project: Optional[str] = None) -> Optional[str]:
    candidates = []
    if client is not None:
        candidates.extend([getattr(client, n, None) for n in ("push", "publish", "upload", "create")])
    candidates.extend([getattr(adk_module, n, None) for n in ("push_schema", "publish_schema", "push_artifact", "publish_artifact")])

    for fn in [c for c in candidates if callable(c)]:
        try:
            if dry_run:
                print(f"[dry-run] SDK would call: {fn.__qualname__}({file_path!r}, name={artifact_name!r})")
                return f"artifact://{artifact_name}/v1"

            try:
                res = fn(file_path)
            except TypeError:
                try:
                    res = fn(path=file_path)
                except TypeError:
                    with open(file_path, "rb") as fh:
                        res = fn(fh)

            print(f"Published {file_path} via SDK using {fn.__qualname__}")
            try:
                if isinstance(res, dict) and "uri" in res:
                    return res["uri"]
                if hasattr(res, "uri"):
                    return getattr(res, "uri")
            except Exception:
                pass
            return f"artifact://{artifact_name}/v1"
        except Exception as e:
            print(f"SDK publish attempt using {getattr(fn, '__qualname__', fn)} failed: {e}", file=sys.stderr)
            continue

    try:
        svc_uri = _publish_with_sdk_service(adk_module, file_path, artifact_name, dry_run, project)
        if svc_uri:
            return svc_uri
    except Exception as e:
        print(f"artifact-service publish attempt failed: {e}", file=sys.stderr)

    print("No usable SDK publish method discovered; falling back to CLI.", file=sys.stderr)
    return None


def publish_with_cli(file_path: str, artifact_name: str, project: Optional[str], dry_run: bool, artifact_map: Optional[dict] = None) -> Optional[str]:
    cmd = ["adk", "artifacts", "push", "--file", file_path]
    if artifact_name:
        cmd += ["--name", artifact_name]
    if project:
        cmd += ["--project", project]

    if dry_run:
        print("[dry-run] CLI would run:", " ".join(cmd))
        return f"artifact://{project or artifact_name}/{artifact_name}/v1"

    print("Running CLI:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if out.strip():
        print(out)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None

    m = re.search(r"artifact://[^\s'\"]+", out)
    if m:
        return m.group(0)

    try:
        j = json.loads(proc.stdout) if proc.stdout and (proc.stdout.strip().startswith("{") or proc.stdout.strip().startswith("[")) else None
        if j:
            def find_uri(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() == "uri" and isinstance(v, str) and v.startswith("artifact://"):
                            return v
                        res = find_uri(v)
                        if res:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = find_uri(item)
                        if res:
                            return res
                return None

            uri = find_uri(j)
            if uri:
                return uri
    except Exception:
        pass

    proj = project
    if not proj and artifact_map:
        for v in artifact_map.get("schemas", {}).values():
            uri = v.get("uri") if isinstance(v, dict) else None
            if isinstance(uri, str) and uri.startswith("artifact://"):
                try:
                    proj = uri.split("/")[2]
                    break
                except Exception:
                    continue

    if not proj:
        proj = artifact_name

    return f"artifact://{proj}/schemas/{artifact_name}/v1"


def run_adk_cli(cmd: list[str], dry_run: bool = False) -> tuple[int, str, str]:
    """Run an `adk` CLI command (list form). Returns (returncode, stdout, stderr).

    This centralizes CLI invocation and makes it easier to mock/test.
    """
    if dry_run:
        print("[dry-run] CLI would run:", " ".join(cmd))
        return 0, "", ""

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout or ""
    err = proc.stderr or ""
    return proc.returncode, out, err


def register_entity_with_sdk(adk_module, payload: dict, entity_kind: str = "tool", name: Optional[str] = None, dry_run: bool = False) -> Optional[str]:
    """Best-effort attempt to register an entity (tool/workflow) via SDK.

    Tries a set of plausible hubs and method names and returns an id/uri on success.
    """
    # map entity kinds to plausible hubs and methods
    hubs_methods = {
        "tool": [
            ("ToolRegistry", ["register_tool", "create_tool"]),
            ("tool_registry", ["register_tool", "create_tool"]),
            ("tools", ["register", "create"]),
        ],
        "workflow": [
            ("workflows", ["register", "create"]),
            ("workflow_registry", ["register_workflow", "create_workflow"]),
        ],
    }

    candidates = hubs_methods.get(entity_kind, [])

    try:
        for hub_name, methods in candidates:
            hub = getattr(adk_module, hub_name, None)
            if hub is None:
                continue
            for method in methods:
                fn = getattr(hub, method, None)
                if not fn:
                    continue
                if dry_run:
                    print(f"[dry-run] SDK would call {hub_name}.{method} -> name={name}")
                    return f"dry-run://sdk/{name or 'entity'}"
                try:
                    # try different calling conventions
                    try:
                        res = fn(name, payload)
                    except TypeError:
                        res = fn(payload)
                    # extract id/uri
                    if isinstance(res, dict) and (res.get("id") or res.get("uri")):
                        return res.get("id") or res.get("uri")
                    if hasattr(res, "id"):
                        return getattr(res, "id")
                    if hasattr(res, "uri"):
                        return getattr(res, "uri")
                    return str(res)
                except Exception:
                    continue
    except Exception:
        return None

    return None


def register_entity_with_cli(cmd: list[str], dry_run: bool = False) -> Optional[str]:
    """Run an adk CLI registration command and attempt to extract an id/uri."""
    try:
        rc, out, err = run_adk_cli(cmd, dry_run=dry_run)
    except TypeError:
        # Some test fakes patch subprocess.run with a signature that doesn't
        # accept stdout/stderr kwargs; fall back to a simpler invocation.
        proc = subprocess.run(cmd, check=False)
        rc = getattr(proc, "returncode", 1)
        out = getattr(proc, "stdout", "") or ""
        err = getattr(proc, "stderr", "") or ""
    combined = (out or "") + "\n" + (err or "")
    if rc != 0:
        return None
    # try to parse JSON stdout
    try:
        j = json.loads(out) if out and (out.strip().startswith("{") or out.strip().startswith("[")) else None
        if j:
            def find_uri(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() in ("uri", "id") and isinstance(v, str) and (v.startswith("artifact://") or v.startswith("tool://") or v.startswith("http")):
                            return v
                        res = find_uri(v)
                        if res:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = find_uri(item)
                        if res:
                            return res
                return None

            uri = find_uri(j)
            if uri:
                return uri
    except Exception:
        pass

    # fallback: search tokens
    for token in combined.split():
        if token.startswith("artifact://") or token.startswith("tool://") or token.startswith("http"):
            return token.strip()

    return (out or "").strip() or None


class _InMemoryMemoryService:
    """A tiny in-memory MemoryService used for deterministic fixture tests.

    This implements a minimal subset of the MemoryService API used by tests:
    - store_session_metadata(session_id, metadata: dict)
    - get_session_metadata(session_id) -> dict
    - append_reviewer_decision(session_id, decision: dict)
    - get_reviewer_decisions(session_id) -> list[dict]
    """

    def __init__(self) -> None:
        self._meta: dict[str, dict] = {}
        self._decisions: dict[str, list] = {}
        self._events: list[dict[str, Any]] = []

    def store_session_metadata(self, session_id: str, metadata: dict) -> None:
        cur = self._meta.get(session_id, {})
        cur.update(metadata or {})
        self._meta[session_id] = cur
        try:
            telemetry.emit_event("memory.store_session_metadata", {"session_id": session_id, "metadata": metadata})
        except Exception:
            pass

    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        res = self._meta.get(session_id)
        try:
            telemetry.emit_event("memory.get_session_metadata", {"session_id": session_id, "found": bool(res)})
        except Exception:
            pass
        return res

    def append_reviewer_decision(self, session_id: str, decision: dict) -> None:
        self._decisions.setdefault(session_id, []).append(decision)
        try:
            telemetry.emit_event("memory.append_reviewer_decision", {"session_id": session_id, "decision": decision})
        except Exception:
            pass

    def get_reviewer_decisions(self, session_id: str) -> list:
        res = list(self._decisions.get(session_id, []))
        try:
            telemetry.emit_event("memory.get_reviewer_decisions", {"session_id": session_id, "count": len(res)})
        except Exception:
            pass
        return res

    def write_memory_event(self, *, run_id: Optional[str], event_type: str, payload: dict[str, Any], timestamp: int) -> None:
        record = {
            "run_id": run_id,
            "event_type": event_type,
            "payload": payload,
            "timestamp": timestamp,
        }
        self._events.append(record)
        try:
            telemetry.emit_event("memory.write_event", {"impl": "inmemory", **record})
        except Exception:
            pass

    def list_memory_events(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        if run_id is None:
            return list(self._events)
        return [evt for evt in self._events if evt.get("run_id") == run_id]

    def clear_memory_events(self) -> None:
        self._events.clear()


class _SQLiteMemoryService:
    """Simple SQLite-backed MemoryService implementation for tests.

    Implements the small API used by tests and runner: store/get session
    metadata and append/list reviewer decisions.
    """

    def __init__(self, db_path: str | Path) -> None:
        import sqlite3

        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                metadata TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                decision TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS ix_memory_events_runid ON memory_events(run_id)")
        self._conn.commit()

    def store_session_metadata(self, session_id: str, metadata: dict) -> None:
        import json

        cur = self._conn.cursor()
        cur.execute("SELECT metadata FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if row and row[0]:
            cur_meta = json.loads(row[0])
            cur_meta.update(metadata or {})
            data = json.dumps(cur_meta)
            cur.execute("UPDATE sessions SET metadata = ? WHERE session_id = ?", (data, session_id))
        else:
            data = json.dumps(metadata or {})
            cur.execute("INSERT OR REPLACE INTO sessions(session_id, metadata) VALUES(?,?)", (session_id, data))
        self._conn.commit()
        try:
            telemetry.emit_event("memory.store_session_metadata", {"session_id": session_id, "metadata": metadata, "db": self._db_path})
        except Exception:
            pass

    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        import json

        cur = self._conn.cursor()
        cur.execute("SELECT metadata FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            try:
                telemetry.emit_event("memory.get_session_metadata", {"session_id": session_id, "found": False, "db": self._db_path})
            except Exception:
                pass
            return None
        res = json.loads(row[0])
        try:
            telemetry.emit_event("memory.get_session_metadata", {"session_id": session_id, "found": True, "db": self._db_path})
        except Exception:
            pass
        return res

    def append_reviewer_decision(self, session_id: str, decision: dict) -> None:
        import json

        cur = self._conn.cursor()
        cur.execute("INSERT INTO decisions(session_id, decision) VALUES(?,?)", (session_id, json.dumps(decision)))
        self._conn.commit()
        try:
            telemetry.emit_event("memory.append_reviewer_decision", {"session_id": session_id, "decision": decision, "db": self._db_path})
        except Exception:
            pass

    def get_reviewer_decisions(self, session_id: str) -> list:
        import json

        cur = self._conn.cursor()
        cur.execute("SELECT decision FROM decisions WHERE session_id = ? ORDER BY id", (session_id,))
        rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

        # (Note: in practice the telemetry above could be emitted here as well,
        # but tests typically call this method immediately after append and/or
        # store; callers can inspect events via the telemetry helper.)

    def write_memory_event(self, *, run_id: Optional[str], event_type: str, payload: dict[str, Any], timestamp: int) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO memory_events(run_id, timestamp, event_type, payload) VALUES(?,?,?,?)",
            (run_id, timestamp, event_type, data),
        )
        self._conn.commit()
        try:
            telemetry.emit_event("memory.write_event", {"impl": "sqlite", "run_id": run_id, "event_type": event_type, "timestamp": timestamp, "db": self._db_path})
        except Exception:
            pass

    def list_memory_events(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        if run_id is None:
            cur.execute("SELECT run_id, timestamp, event_type, payload FROM memory_events ORDER BY id")
            rows = cur.fetchall()
        else:
            cur.execute(
                "SELECT run_id, timestamp, event_type, payload FROM memory_events WHERE run_id = ? ORDER BY id",
                (run_id,),
            )
            rows = cur.fetchall()
        events = []
        for run_id_val, ts, event_type, payload in rows:
            events.append({
                "run_id": run_id_val,
                "timestamp": ts,
                "event_type": event_type,
                "payload": json.loads(payload),
            })
        return events

    def clear_memory_events(self) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM memory_events")
        self._conn.commit()


def get_memory_service() -> object:
    """Return a usable MemoryService implementation.

    In fixture mode (when `ADK_USE_FIXTURE=1`) this returns an in-memory
    service suitable for unit tests. In non-fixture environments we attempt
    to locate an SDK-backed MemoryService and otherwise raise RuntimeError.
    """
    backend_factory = _current_backend().memory_service_factory
    if backend_factory:
        svc = backend_factory()
        if svc is None:
            raise RuntimeError("Backend memory service factory returned None")
        return svc

    if os.environ.get("ADK_USE_FIXTURE") == "1":
        # reuse a single in-memory instance so multiple tools in the same
        # process share session state during fixture-mode tests
        global _FIXTURE_MEMORY_SERVICE
        try:
            _FIXTURE_MEMORY_SERVICE
        except NameError:
            _FIXTURE_MEMORY_SERVICE = _InMemoryMemoryService()
        try:
            telemetry.emit_event("memory.get_memory_service", {"impl": "inmemory"})
        except Exception:
            pass
        return _FIXTURE_MEMORY_SERVICE

    # If an explicit SQLite path is provided, use a lightweight file-backed
    # MemoryService for persistence tests or local deployments.
    sqlite_path = os.environ.get("ADK_MEMORY_SQLITE")
    if sqlite_path:
        try:
            svc = _SQLiteMemoryService(sqlite_path)
            try:
                telemetry.emit_event("memory.get_memory_service", {"impl": "sqlite", "path": sqlite_path})
            except Exception:
                pass
            return svc
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite MemoryService: {e}")

    # Attempt to locate a MemoryService via the ADK SDK if present.
    res = probe_sdk()
    if not res:
        raise RuntimeError("ADK SDK not available to provide MemoryService")
    adk_mod, _ = res

    # Common SDK naming conventions for memory/session services
    for cand in ("memory_service", "MemoryService", "session_service", "SessionService"):
        svc = getattr(adk_mod, cand, None)
        if svc is not None:
            try:
                telemetry.emit_event("memory.get_memory_service", {"impl": "sdk", "candidate": cand})
            except Exception:
                pass
            return svc

    raise RuntimeError("No MemoryService available via SDK")


def _normalize_timestamp(ts: Optional[datetime]) -> int:
    dt = ts or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _fixture_mode_enabled() -> bool:
    return os.environ.get("ADK_USE_FIXTURE") == "1"


def write_memory_event(
    *,
    run_id: Optional[str],
    event_type: str,
    payload: Mapping[str, Any],
    ts: Optional[datetime] = None,
) -> None:
    """Persist a structured memory event via the active MemoryService."""

    if not event_type:
        raise ValueError("event_type is required")

    fixture_mode = _fixture_mode_enabled()
    normalized_run_id = run_id or os.environ.get("SPARKLE_RUN_ID") or "unknown"
    payload_dict = dict(payload or {})
    timestamp = _normalize_timestamp(ts)

    try:
        svc = get_memory_service()
    except SystemExit as exc:
        if fixture_mode:
            try:
                telemetry.emit_event(
                    "memory.write_event.skipped",
                    {"run_id": normalized_run_id, "event_type": event_type, "reason": "fixture"},
                )
            except Exception:
                pass
            return
        raise MemoryWriteError("MemoryService unavailable", run_id=normalized_run_id, event_type=event_type, cause=exc) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise MemoryWriteError("MemoryService unavailable", run_id=normalized_run_id, event_type=event_type, cause=exc) from exc

    writer = (
        getattr(svc, "write_memory_event", None)
        or getattr(svc, "append_memory_event", None)
        or getattr(svc, "record_memory_event", None)
        or getattr(svc, "log_event", None)
    )

    if writer is None:
        raise MemoryWriteError("MemoryService does not expose a memory-event writer", run_id=normalized_run_id, event_type=event_type)

    try:
        try:
            writer(run_id=normalized_run_id, event_type=event_type, payload=payload_dict, timestamp=timestamp)
        except TypeError:
            try:
                writer(normalized_run_id, event_type, payload_dict, timestamp)
            except TypeError:
                writer({"run_id": normalized_run_id, "event_type": event_type, "payload": payload_dict, "timestamp": timestamp})
    except Exception as exc:
        raise MemoryWriteError("Failed to write memory event", run_id=normalized_run_id, event_type=event_type, cause=exc) from exc

    try:
        telemetry.emit_event(
            "memory.write_event.completed",
            {
                "run_id": normalized_run_id,
                "event_type": event_type,
                "timestamp": timestamp,
            },
        )
    except Exception:
        pass
