"""PromptTemplate helpers for ScriptAgent and future ADK consumers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from sparkle_motion import schema_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_DIR = REPO_ROOT / "artifacts" / "prompt_templates"


@dataclass(frozen=True)
class PromptTemplateSpec:
    """Serializable representation of an ADK PromptTemplate.

    Stays lightweight so unit tests can exercise structure without importing the
    full ADK SDK. The `response_json_schema` portion is purposely stored as the
    schema artifact URI plus a local fallback path so both hosted ADK runtimes
    and the Colab-local profile receive consistent contracts.
    """

    template_id: str
    description: str
    model: str
    system_prompt: str
    user_prompt: str
    response_schema_uri: str
    response_schema_path: Path
    input_variables: Tuple[str, ...]

    def to_payload(self) -> Dict[str, Any]:
        """Return a dict ready for `adk llm-prompts push` JSON serialization."""
        return self.to_payload(portable=True)

    def to_payload(self, portable: bool = True) -> Dict[str, Any]:
        """Return a dict ready for `adk llm-prompts push` JSON serialization.

        If `portable` is True the `local_fallback_path` will be repo-relative
        when possible; otherwise an absolute path will be returned.
        """

        local_path = _portable_path(self.response_schema_path) if portable else str(self.response_schema_path)

        return {
            "id": self.template_id,
            "description": self.description,
            "model": self.model,
            "input_variables": list(self.input_variables),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
            ],
            "response_json_schema": {
                "artifact_uri": self.response_schema_uri,
                "local_fallback_path": local_path,
            },
        }


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:  # pragma: no cover - defensive for non-repo paths
        return str(path)


def _default_script_agent_prompts() -> Tuple[str, str]:
    system_prompt = (
        "You are ScriptAgent, a meticulous film planner who turns loose ideas into"
        " structured MoviePlan JSON. Always reason through tone, pacing, and"
        " narrative beats before emitting the final answer."
        " Validate that every shot is cinematic, time-bounded, and production-ready."
    )
    user_prompt = (
        "You will receive a film idea, duration target, and optional style notes.\n"
        "Return a single JSON object that conforms to the MoviePlan schema.\n\n"
        "Idea: {idea}\n"
        "Target duration (minutes): {duration_minutes}\n"
        "Style notes: {style_notes}\n\n"
        "Guidelines:\n"
        "- Include between 6 and 12 shots unless explicitly requested otherwise.\n"
        "- Populate dialogue for talking closeups; leave empty elsewhere.\n"
        "- Add motion prompts for dynamic sequences (cranes, dolly, drone).\n"
        "- Use metadata.seed to capture any deterministic hints from operators.\n"
        "- Never emit explanations outside the JSON payload."
    )
    return system_prompt, user_prompt


def build_script_agent_prompt_template(
    *,
    model: str = "gemini-1.5-pro",
    template_id: str = "script_agent_movie_plan_v1",
    description: str = "Structured MoviePlan generator backed by schema artifacts",
    input_variables: Iterable[str] | None = None,
) -> PromptTemplateSpec:
    """Construct the ScriptAgent PromptTemplate bound to schema artifacts."""

    schema = schema_registry.load_catalog().get_schema("movie_plan")
    system_prompt, user_prompt = _default_script_agent_prompts()

    variables = tuple(input_variables or ("idea", "duration_minutes", "style_notes"))

    return PromptTemplateSpec(
        template_id=template_id,
        description=description,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema_uri=schema.uri,
        response_schema_path=schema.local_path,
        input_variables=variables,
    )


def render_script_agent_prompt_template(
    output_path: Path | None = None,
    use_repo_relative_local_fallback: bool = True,
    **build_kwargs: Any,
) -> Path:
    """Render the ScriptAgent PromptTemplate JSON to disk. Returns the path."""

    spec = build_script_agent_prompt_template(**build_kwargs)
    target_path = output_path or (DEFAULT_PROMPT_DIR / f"{spec.template_id}.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(spec.to_payload(portable=use_repo_relative_local_fallback), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return target_path


__all__ = [
    "PromptTemplateSpec",
    "build_script_agent_prompt_template",
    "render_script_agent_prompt_template",
]
