from __future__ import annotations

import asyncio
import getpass
import mimetypes
import os
import re
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, Tuple


def probe_sdk() -> Optional[Tuple[object, Optional[object]]]:
    """Try to import google.adk and discover an artifacts client.

    Returns (adk_module, client_candidate) or None when SDK import fails.
    The client_candidate may be None when the module exists but no obvious
    artifacts helper is present.
    """
    try:
        import google.adk as adk  # type: ignore
    except Exception as e:
        # ADK Python SDK is required for all tooling in this repository.
        # Fail fast and provide a clear error so callers don't silently
        # fall back to alternate code paths.
        print("ERROR: google.adk SDK not importable: {}".format(e), file=sys.stderr)
        raise SystemExit(1)

    for cand in ("artifacts", "ArtifactService", "artifacts_client", "artifact_client"):
        client = getattr(adk, cand, None)
        if client is not None:
            return adk, client

    return adk, None


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

    def store_session_metadata(self, session_id: str, metadata: dict) -> None:
        cur = self._meta.get(session_id, {})
        cur.update(metadata or {})
        self._meta[session_id] = cur

    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        return self._meta.get(session_id)

    def append_reviewer_decision(self, session_id: str, decision: dict) -> None:
        self._decisions.setdefault(session_id, []).append(decision)

    def get_reviewer_decisions(self, session_id: str) -> list:
        return list(self._decisions.get(session_id, []))


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

    def get_session_metadata(self, session_id: str) -> Optional[dict]:
        import json

        cur = self._conn.cursor()
        cur.execute("SELECT metadata FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])

    def append_reviewer_decision(self, session_id: str, decision: dict) -> None:
        import json

        cur = self._conn.cursor()
        cur.execute("INSERT INTO decisions(session_id, decision) VALUES(?,?)", (session_id, json.dumps(decision)))
        self._conn.commit()

    def get_reviewer_decisions(self, session_id: str) -> list:
        import json

        cur = self._conn.cursor()
        cur.execute("SELECT decision FROM decisions WHERE session_id = ? ORDER BY id", (session_id,))
        rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]


def get_memory_service() -> object:
    """Return a usable MemoryService implementation.

    In fixture mode (when `ADK_USE_FIXTURE=1`) this returns an in-memory
    service suitable for unit tests. In non-fixture environments we attempt
    to locate an SDK-backed MemoryService and otherwise raise RuntimeError.
    """
    if os.environ.get("ADK_USE_FIXTURE") == "1":
            # reuse a single in-memory instance so multiple tools in the same
            # process share session state during fixture-mode tests
            global _FIXTURE_MEMORY_SERVICE
            try:
                _FIXTURE_MEMORY_SERVICE
            except NameError:
                _FIXTURE_MEMORY_SERVICE = _InMemoryMemoryService()
            return _FIXTURE_MEMORY_SERVICE

    # If an explicit SQLite path is provided, use a lightweight file-backed
    # MemoryService for persistence tests or local deployments.
    sqlite_path = os.environ.get("ADK_MEMORY_SQLITE")
    if sqlite_path:
        try:
            return _SQLiteMemoryService(sqlite_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite MemoryService: {e}")

    # Attempt to locate a MemoryService via the ADK SDK if present.
    try:
        adk_mod, _ = probe_sdk()
    except SystemExit:
        raise RuntimeError("ADK SDK not available to provide MemoryService")

    # Common SDK naming conventions for memory/session services
    for cand in ("memory_service", "MemoryService", "session_service", "SessionService"):
        svc = getattr(adk_mod, cand, None)
        if svc is not None:
            return svc

    raise RuntimeError("No MemoryService available via SDK")
