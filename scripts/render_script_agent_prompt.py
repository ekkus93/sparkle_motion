#!/usr/bin/env python
"""Render the ScriptAgent PromptTemplate JSON bound to schema artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from sparkle_motion import prompt_templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Optional output path for the rendered JSON. Defaults to"
            " artifacts/prompt_templates/<template_id>.json"
        ),
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-pro",
        help="Model identifier to record in the PromptTemplate metadata.",
    )
    parser.add_argument(
        "--template-id",
        default="script_agent_movie_plan_v1",
        help="Template ID stored in the serialized payload.",
    )
    parser.add_argument(
        "--description",
        default="Structured MoviePlan generator backed by schema artifacts",
        help="Human-readable description for the template.",
    )
    parser.add_argument(
        "--input-var",
        dest="input_vars",
        action="append",
        default=None,
        help="Repeatable flag to override the default input variables order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_vars: Iterable[str] | None = args.input_vars
    path = prompt_templates.render_script_agent_prompt_template(
        output_path=args.output_path,
        model=args.model,
        template_id=args.template_id,
        description=args.description,
        input_variables=input_vars if input_vars else None,
    )
    print(f"Wrote ScriptAgent prompt template to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
