import asyncio
import fcntl
import os
from pathlib import Path
from typing import Any


class FileArtifactService:
    """Simple file-backed ArtifactService shim.

    save_artifact(...) will write the artifact bytes or copy the referenced
    file under a root directory organized by app_name and filename, and
    return a numeric revision id.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def save_artifact(self, app_name: str, user_id: str, filename: str, artifact: Any) -> int:
        # Ensure app subdir exists
        app_dir = self.root / app_name
        app_dir.mkdir(parents=True, exist_ok=True)

        out_path = app_dir / filename

        # artifact can be a Part-like object, a bytes payload, or a file URI string
        try:
            if hasattr(artifact, "data"):
                # Part.from_bytes path: write raw bytes
                data = getattr(artifact, "data")
                if isinstance(data, bytes):
                    out_path.write_bytes(data)
                else:
                    out_path.write_text(str(data), encoding="utf-8")
            elif isinstance(artifact, (bytes, bytearray)):
                out_path.write_bytes(bytes(artifact))
            elif isinstance(artifact, str) and os.path.exists(artifact):
                # artifact is a local file path
                from shutil import copy2

                copy2(artifact, out_path)
            elif isinstance(artifact, str) and artifact.startswith("file://"):
                src = artifact[len("file://") :]
                from shutil import copy2

                copy2(src, out_path)
            elif isinstance(artifact, str):
                # write the string into the file
                out_path.write_text(artifact, encoding="utf-8")
            else:
                # fallback: serialize repr
                out_path.write_text(repr(artifact), encoding="utf-8")
        except Exception:
            # best-effort fallback: write a JSON stub
            out_path.write_text('{"stub": true}', encoding="utf-8")

        # Use a small revision file to provide incremental revision numbers
        rev_file = app_dir / ".rev"
        try:
            # open the rev file for read/write (create if missing)
            with open(rev_file, "a+", encoding="utf-8") as f:
                # Acquire an exclusive lock while reading/updating
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
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
                finally:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
        except Exception:
            # best-effort fallback
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
