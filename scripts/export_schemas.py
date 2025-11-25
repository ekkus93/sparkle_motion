#!/usr/bin/env python
"""Export canonical JSON Schemas for Sparkle Motion contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Type

from pydantic import BaseModel

from sparkle_motion import schemas as sm_schemas

SchemaMap = Dict[str, Tuple[Type[BaseModel], str]]

SCHEMA_TARGETS: SchemaMap = {
    "MoviePlan": (sm_schemas.MoviePlan, "MoviePlan contract for ScriptAgent"),
    "AssetRefs": (sm_schemas.AssetRefs, "Asset references produced by stages"),
    "QAReport": (sm_schemas.QAReport, "Automated QA output"),
    "StageEvent": (sm_schemas.StageEvent, "Run manifest StageEvent entries"),
    "Checkpoint": (sm_schemas.Checkpoint, "Per-stage checkpoint payload"),
}


def export_schemas(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (model, description) in SCHEMA_TARGETS.items():
        schema = model.model_json_schema()
        schema.setdefault("$id", f"sparkle_motion/{name}.schema.json")
        schema.setdefault("title", name)
        schema.setdefault("description", description)

        path = output_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {path.relative_to(output_dir.parent) if output_dir.parent in path.parents else path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "schemas",
        help="Directory for generated JSON schemas (default: repo_root/schemas)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_schemas(args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
