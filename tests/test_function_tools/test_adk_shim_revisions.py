import asyncio
import importlib.util
from pathlib import Path


def _load_local_file_artifact_service():
    # Load the local shim directly from src/ to avoid shadowing by any
    # installed `google` packages in the environment. Walk upward from the
    # test file to find the repository root that contains `src/google/...`.
    p = Path(__file__).resolve()
    target_rel = Path("src") / "google" / "adk" / "artifacts" / "file_artifact_service.py"
    mod_path = None
    for parent in p.parents:
        candidate = parent / target_rel
        if candidate.exists():
            mod_path = candidate
            break
    if mod_path is None:
        raise FileNotFoundError("Could not locate local file_artifact_service.py under src/google/adk/artifacts")

    spec = importlib.util.spec_from_file_location("local_file_artifact_service", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.FileArtifactService


def test_file_artifact_revision_increments(tmp_path: Path):
    FileArtifactService = _load_local_file_artifact_service()

    root = tmp_path / "file_artifacts"
    svc = FileArtifactService(root)

    app = "testapp"
    filename = "payload.bin"
    data = b"abc123"

    # Call save_artifact multiple times and assert revisions increment
    for expected_rev in range(1, 4):
        rev = asyncio.run(svc.save_artifact(app, "user", filename, data))
        assert rev == expected_rev

        out_path = root / app / filename
        assert out_path.exists()
        assert out_path.read_bytes() == data
