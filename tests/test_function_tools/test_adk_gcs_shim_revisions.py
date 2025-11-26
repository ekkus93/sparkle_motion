import asyncio
import importlib.util
from pathlib import Path


def _load_local_gcs_artifact_service():
    # Locate the local shim under src/google/adk/artifacts
    p = Path(__file__).resolve()
    target_rel = Path("src") / "google" / "adk" / "artifacts" / "gcs_artifact_service.py"
    mod_path = None
    for parent in p.parents:
        candidate = parent / target_rel
        if candidate.exists():
            mod_path = candidate
            break
    if mod_path is None:
        raise FileNotFoundError("Could not locate local gcs_artifact_service.py under src/google/adk/artifacts")

    spec = importlib.util.spec_from_file_location("local_gcs_artifact_service", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.GcsArtifactService


def test_gcs_artifact_revision_increments(tmp_path: Path):
    GcsArtifactService = _load_local_gcs_artifact_service()

    bucket = "testbucket"
    svc = GcsArtifactService(bucket)

    # Redirect root to tmp_path so test doesn't write into repo artifacts/
    svc.root = tmp_path / "gcs" / bucket
    svc.root.mkdir(parents=True, exist_ok=True)

    app = "testapp"
    filename = "payload.txt"
    data = "hello-gcs"

    for expected_rev in range(1, 4):
        rev = asyncio.run(svc.save_artifact(app, "user", filename, data))
        assert rev == expected_rev

        out_path = svc.root / app / filename
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == data
