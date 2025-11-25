#!/usr/bin/env python
"""Package the QA policy bundle for ADK ArtifactService upload."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import List, Tuple

FileSpec = List[Tuple[Path, str]]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_files(files: FileSpec, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src, dest_name in files:
        target = destination / dest_name
        shutil.copy2(src, target)


def build_manifest(version_dir: Path, artifact_id: str, version: str) -> Path:
    entries = []
    for child in sorted(version_dir.iterdir()):
        if child.is_file():
            entries.append(
                {
                    "name": child.name,
                    "size": child.stat().st_size,
                    "sha256": sha256sum(child),
                }
            )

    manifest = {
        "artifact_id": artifact_id,
        "version": version,
        "files": entries,
    }
    path = version_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_archive(version_dir: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for child in version_dir.iterdir():
            tar.add(child, arcname=f"{version_dir.name}/{child.name}")
    return archive_path


def package_qa_policy(
    *,
    version: str,
    artifact_id: str,
    output_root: Path,
    policy_path: Path,
    schema_path: Path,
) -> None:
    version_dir = output_root / version
    copy_files(
        [
            (policy_path, "qa_policy.yaml"),
            (schema_path, "qa_policy.schema.json"),
        ],
        version_dir,
    )
    manifest_path = build_manifest(version_dir, artifact_id, version)
    archive_path = build_archive(version_dir, output_root / f"qa_policy_{version}.tar.gz")
    print(f"Wrote bundle directory: {version_dir}")
    print(f"Wrote manifest:        {manifest_path}")
    print(f"Wrote tarball:         {archive_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1", help="Semantic bundle version (e.g., v1)")
    parser.add_argument(
        "--artifact-id",
        default="sparkle-motion/qa_policy",
        help="ArtifactService namespace/id to record in the manifest",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "qa_policy",
        help="Root directory where versioned bundles are written",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "qa_policy.yaml",
        help="Source QA policy YAML file",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "qa_policy.schema.json",
        help="Source QA policy JSON Schema file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_qa_policy(
        version=args.version,
        artifact_id=args.artifact_id,
        output_root=args.output_root,
        policy_path=args.policy,
        schema_path=args.schema,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
