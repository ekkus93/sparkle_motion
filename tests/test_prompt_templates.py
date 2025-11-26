from __future__ import annotations

import json
from pathlib import Path

from sparkle_motion import prompt_templates, schema_registry


def test_build_script_agent_prompt_template_uses_schema_registry() -> None:
    spec = prompt_templates.build_script_agent_prompt_template(model="test-model")

    assert spec.response_schema_uri == schema_registry.get_schema_uri("movie_plan")
    assert spec.response_schema_path == schema_registry.get_schema_path("movie_plan")

    payload = spec.to_payload()
    assert payload["response_json_schema"]["artifact_uri"] == spec.response_schema_uri
    try:
        expected_local = str(
            spec.response_schema_path.relative_to(prompt_templates.REPO_ROOT)
        )
    except ValueError:
        expected_local = str(spec.response_schema_path)
    assert payload["response_json_schema"]["local_fallback_path"] == expected_local
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


def test_render_script_agent_prompt_template(tmp_path: Path) -> None:
    output = tmp_path / "script_agent_prompt.json"

    rendered = prompt_templates.render_script_agent_prompt_template(
        output_path=output,
        model="gemini-test",
        template_id="script_agent_movie_plan_test",
    )

    data = json.loads(rendered.read_text(encoding="utf-8"))
    assert data["id"] == "script_agent_movie_plan_test"
    assert data["model"] == "gemini-test"
    assert data["response_json_schema"]["artifact_uri"] == schema_registry.get_schema_uri(
        "movie_plan"
    )


def test_to_payload_repo_relative_local_fallback(tmp_path: Path) -> None:
    """Ensure `to_payload(portable=True)` emits a repo-relative local_fallback_path
    when the schema file is inside the repository root."""

    from sparkle_motion.prompt_templates import PromptTemplateSpec, REPO_ROOT

    # create a schema file under the repo so _portable_path can produce a relative path
    repo_schema_dir = REPO_ROOT / "artifacts" / "schemas"
    repo_schema_dir.mkdir(parents=True, exist_ok=True)
    repo_file = repo_schema_dir / "test_repo_relative.schema.json"
    repo_file.write_text("{}", encoding="utf-8")

    spec = PromptTemplateSpec(
        template_id="t-repo-relative",
        description="d",
        model="m",
        system_prompt="s",
        user_prompt="u",
        response_schema_uri="artifact://example/test",
        response_schema_path=repo_file,
        input_variables=(),
    )

    payload = spec.to_payload(portable=True)
    local = payload["response_json_schema"]["local_fallback_path"]

    # Should be a relative path (not absolute) and equal to path relative to REPO_ROOT
    assert not Path(local).is_absolute()
    assert local == str(repo_file.relative_to(REPO_ROOT))