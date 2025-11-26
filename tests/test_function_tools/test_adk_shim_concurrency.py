import asyncio
import importlib.util
from pathlib import Path


def _load_local_service(module_name: str, filename: str):
    p = Path(__file__).resolve()
    target_rel = Path("src") / "google" / "adk" / "artifacts" / filename
    mod_path = None
    for parent in p.parents:
        candidate = parent / target_rel
        if candidate.exists():
            mod_path = candidate
            break
    if mod_path is None:
        raise FileNotFoundError(f"Could not locate local {filename} under src/google/adk/artifacts")

    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


async def _concurrent_saves(svc, app: str, filename: str, data, n: int):
    tasks = [asyncio.create_task(svc.save_artifact(app, "user", filename, data)) for _ in range(n)]
    return await asyncio.gather(*tasks)


def test_file_shim_concurrent_revisions(tmp_path: Path):
    mod = _load_local_service("local_file_artifact_service", "file_artifact_service.py")
    FileArtifactService = mod.FileArtifactService

    root = tmp_path / "file_artifacts"
    svc = FileArtifactService(root)

    app = "concurrent-app"
    filename = "data.bin"
    data = b"concurrent"

    revs = asyncio.run(_concurrent_saves(svc, app, filename, data, 10))
    assert sorted(revs) == list(range(1, 11))

    out_path = root / app / filename
    assert out_path.exists()
    assert out_path.read_bytes() == data


def test_gcs_shim_concurrent_revisions(tmp_path: Path):
    mod = _load_local_service("local_gcs_artifact_service", "gcs_artifact_service.py")
    GcsArtifactService = mod.GcsArtifactService

    bucket = "concurrent-bucket"
    svc = GcsArtifactService(bucket)
    # redirect to tmp
    svc.root = tmp_path / "gcs" / bucket
    svc.root.mkdir(parents=True, exist_ok=True)

    app = "concurrent-app"
    filename = "data.txt"
    data = "concurrent-gcs"

    revs = asyncio.run(_concurrent_saves(svc, app, filename, data, 8))
    assert sorted(revs) == list(range(1, 9))

    out_path = svc.root / app / filename
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == data
