import asyncio
import os
from pathlib import Path
from typing import Any


class GcsArtifactService:
    """GCS-backed ArtifactService shim for local testing.

    This shim writes files under `artifacts/gcs/<bucket>/<app_name>/...`.
    """

    def __init__(self, bucket: str):
        # map bucket to a local directory for testing
        self.root = Path("artifacts/gcs") / bucket
        self.root.mkdir(parents=True, exist_ok=True)

    async def save_artifact(self, app_name: str, user_id: str, filename: str, artifact: Any) -> int:
        app_dir = self.root / app_name
        app_dir.mkdir(parents=True, exist_ok=True)
        out_path = app_dir / filename

        try:
            if hasattr(artifact, "data"):
                data = getattr(artifact, "data")
                if isinstance(data, bytes):
                    out_path.write_bytes(data)
                else:
                    out_path.write_text(str(data), encoding="utf-8")
            elif isinstance(artifact, str):
                # copy or write
                if artifact.startswith("file://"):
                    src = artifact[len("file://") :]
                    from shutil import copy2

                    copy2(src, out_path)
                else:
                    out_path.write_text(artifact, encoding="utf-8")
            else:
                out_path.write_text(repr(artifact), encoding="utf-8")
        except Exception:
            out_path.write_text('{"stub": true}', encoding="utf-8")

        # Persist an incremental revision counter per-app to simulate real ADK
        rev_file = app_dir / ".rev"
        try:
            with open(rev_file, "a+", encoding="utf-8") as f:
                from .portable_lock import exclusive_lock

                with exclusive_lock(f):
                    f.seek(0)
                    raw = f.read().strip()
                    try:
                        rev = int(raw or "0")
                    except Exception:
                        rev = 0
                    rev += 1
                    f.seek(0)
                    f.truncate()
                    f.write(str(rev))
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass
        except Exception:
            try:
                if rev_file.exists():
                    rev = int(rev_file.read_text(encoding="utf-8").strip() or "0")
                else:
                    rev = 0
            except Exception:
                rev = 0
            rev = rev + 1
            try:
                rev_file.write_text(str(rev), encoding="utf-8")
            except Exception:
                pass

        return rev
