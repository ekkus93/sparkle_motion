from pathlib import Path
import json

from sparkle_motion import schema_registry


def test_tool_metadata_files_exist_and_valid():
    repo_root = Path(__file__).resolve().parents[1]
    ft_dir = repo_root / "function_tools"
    assert ft_dir.exists(), "function_tools directory missing"

    metas = sorted(ft_dir.glob("*/metadata.json"))
    assert metas, "No metadata.json files found under function_tools/*"

    required_keys = {"name", "version", "invoke_path", "health_path", "response_json_schema"}

    for m in metas:
        data = json.loads(m.read_text(encoding="utf-8"))
        missing = required_keys - set(data.keys())
        assert not missing, f"{m} missing keys: {sorted(missing)}"

        # response_json_schema should be non-empty (string or object)
        rjs = data.get("response_json_schema")
        assert rjs, f"{m} has empty response_json_schema"

        # If schemas.output exists, it should match response_json_schema
        schemas = data.get("schemas") or {}
        if "output" in schemas:
            assert schemas["output"] == data["response_json_schema"], (
                f"{m}: schemas.output does not match response_json_schema"
            )

    script_agent_meta = next((m for m in metas if m.parent.name == "script_agent"), None)
    assert script_agent_meta is not None, "script_agent metadata missing"
    data = json.loads(script_agent_meta.read_text(encoding="utf-8"))
    assert data["response_json_schema"] == schema_registry.movie_plan_schema().uri
