# Scripts / Operator tools

This document explains the helper scripts in `scripts/` and how to use them in the local/Colab development profile for this repository.

Implementation plan:
- Create a single `scripts/README.md` containing per-script sections.
- Provide usage examples for SDK-first and CLI-fallback flows where applicable.
- Add Colab-safety notes (no Docker) and quick 'try it' commands for local dev (conda env `sparkle_motion`).

Conflicts:
- None.

## Where to run

Preferred development environment: the `sparkle_motion` conda environment (recorded in project memory). Example local test invocation uses:

```bash
# run tests in the repo from the workspace root
conda run -n sparkle_motion bash -lc 'PYTHONPATH=.:src pytest -q'
```

On Colab: do NOT use Docker. Use the in-process FastAPI runner (see `run_function_tool.py`) and mount your Drive + load credentials via the supplied notebook cells.

## Environment / credentials

Most scripts interact with ADK and therefore require ADK credentials to be available in the environment (how they are provided depends on your setup):

- `ADK_CREDENTIALS_FILE` (optional): path to a service account JSON (if you use it).
- `GOOGLE_APPLICATION_CREDENTIALS` (optional): standard Google env var pointing to credentials.
- `ADK_PROJECT` (optional): canonical ADK project name used by CLI or SDK.

Notes:
- Scripts use a guarded import pattern: they try the `google.adk` SDK first and fall back to the `adk` CLI when the SDK is unavailable. CLI fallback requires `adk` to be on PATH and authenticated.
- When running from Colab, prefer using the CLI fallback only if you can safely authenticate in that environment and the notebook explicitly documents the steps; otherwise use in-process flows with local credentials loaded via `.env` / Drive.

## Quick audit: SDK vs CLI behavior

- SDK-first: guarded `import google.adk` inside script. If SDK present and exposes the expected API surface, scripts will call SDK functions directly.
- CLI-fallback: if SDK is missing or the SDK surface is probed and not found, scripts write a temporary artifact JSON and call the `adk` CLI subcommand (for example `adk tools register --file <tmp>`).

Scripts expose flags such as `--dry-run` and `--use-cli` to force behavior and allow safe operators to test without making changes.

**ADK Helpers**

- **Module**: `sparkle_motion.adk_helpers` centralizes ADK SDK probing and CLI fallback behavior used by the scripts.
- **Key functions**:
	- `probe_sdk()` — tries to import `google.adk` and returns a `(module, client_candidate)` tuple (or `None` when SDK missing).
	- `run_adk_cli(cmd: list[str], dry_run: bool = False)` — runs an `adk` CLI command (list form) and returns `(returncode, stdout, stderr)`; supports a dry-run mode.
	- `register_entity_with_sdk(adk_module, payload, entity_kind="tool", name=None, dry_run=False)` — best-effort SDK registration helper for tools/workflows.
	- `register_entity_with_cli(cmd: list[str], dry_run: bool = False)` — runs a registration CLI command and attempts to parse an id/uri from JSON or token heuristics.
	- `publish_with_sdk(...)` and `publish_with_cli(...)` — helpers used by `scripts/publish_schemas.py` to centralize schema publishing logic.

- **Why this helps**: centralizing the probing/parsing and dry-run semantics makes the scripts easier to maintain, more testable (we patch `run_adk_cli` in unit tests), and reduces duplication across `register_*`, `push_prompt_template.py`, and `publish_schemas.py`.

- **Small example (script pattern)**:

```py
from sparkle_motion.adk_helpers import probe_sdk, register_entity_with_sdk, register_entity_with_cli

sdk = probe_sdk()
if sdk:
		# sdk is a tuple (module, client) — pass module to helper
		uri = register_entity_with_sdk(sdk[0], payload, entity_kind="tool", name="my_tool", dry_run=False)
else:
		cmd = ["adk", "tools", "register", "--file", "/tmp/tool.json"]
		uri = register_entity_with_cli(cmd, dry_run=False)
```

- **Tests**: see `tests/test_adk_helpers_unit.py` for concrete unit tests that exercise dry-run behavior, JSON stdout parsing, token fallbacks, and SDK return shapes.

---

## scripts/register_tools.py

Purpose:
- Read `configs/tool_registry.yaml` and register FunctionTools (local / canonical) with ADK.

Key options (run `python scripts/register_tools.py --help`):
- `--file` / `--registry`: path to the registry YAML (default: `configs/tool_registry.yaml`).
- `--dry-run`: validate & print what would be registered without mutating ADK.
- `--use-cli`: force CLI fallback even if SDK is available.
- `--force` / `--idempotent`: attempt to overwrite/replace if already registered.

Examples:

SDK-first (preferred when SDK available):
```bash
PYTHONPATH=src python scripts/register_tools.py --registry configs/tool_registry.yaml --dry-run
```

Force CLI flow:
```bash
PYTHONPATH=src python scripts/register_tools.py --use-cli --registry configs/tool_registry.yaml
```

Troubleshooting:
- If the script fails claiming the SDK is missing, either install `google-adk` in the `sparkle_motion` env or run with `--use-cli` and ensure `adk` is authenticated on PATH.

---

## scripts/register_workflow.py

Purpose:
- Publish WorkflowAgent definitions and workflow YAMLs/JSONs to ADK.
- Similar guarded SDK / CLI-fallback pattern to `register_tools.py`.

Key options (run `python scripts/register_workflow.py --help`):
- `--file` / `--workflow`: path to workflow definition(s).
- `--dry-run`, `--use-cli`, `--force`.

Example:
```bash
PYTHONPATH=src python scripts/register_workflow.py --file workflows/sample_workflow.yaml --dry-run
```

Notes:
- Workflow definitions should reference artifact URIs for prompts and schemas. If you haven't yet published those artifacts, see `scripts/push_prompt_template.py` and `scripts/publish_schemas.py` (planned).

---

## scripts/push_prompt_template.py

Purpose:
- Render prompt templates created by `src/sparkle_motion/prompt_templates.py` and push them to ADK llm-prompts.

Behavior:
- Renders a JSON file for the prompt template and then either uses `google.adk` SDK to push or calls `adk llm-prompts push --file <file>` as a fallback.

Example (render and push):
```bash
PYTHONPATH=src python scripts/push_prompt_template.py --template script_agent_movie_plan_v1 --push --dry-run
```

Troubleshooting:
- If your prompt references a schema artifact URI that is not yet published, `push` will warn or fail; publish schemas first.

---

## scripts/run_function_tool.py (Colab-friendly runner)

Purpose:
- Start a local in-process FastAPI server that exposes endpoints for a FunctionTool defined in the repo, useful on Colab (no Docker).

Why this exists:
- On Colab we must avoid Docker; this runner hosts a tool in-process and maps the expected FunctionTool HTTP endpoints (health, invoke).

Example (run a tool from the registry):
```bash
PYTHONPATH=src python scripts/run_function_tool.py --tool my_tool_id --port 8000
# then inspect http://localhost:8000/healthz
```

Notes for Colab:
- Mount Drive and load credentials using the provided notebook cells before starting the runner.
- If you need to expose the port in Colab, use ngrok or the Colab-provided forwarding mechanism (not covered here). Prefer testing with the FastAPI TestClient in local unit tests instead of exposing ports in notebooks.

---

## scripts/package_qa_policy.py

Purpose:
- Package the QA policy artifacts (policy YAML, schema, manifest) into a versioned artifact bundle in `artifacts/qa_policy/v1/`.

Usage:
```bash
python scripts/package_qa_policy.py --policy configs/qa_policy.yaml --out artifacts/qa_policy/v1
```

Follow-up:
- After packaging, use `adk` CLI or the SDK to upload the artifact to your ADK artifact store (see TODO: `publish_schemas.py`).

---

## scripts/publish_schemas.py (planned)

Purpose:
- Push JSON Schemas from `schemas/` or `configs/schema_artifacts.yaml` into the ADK artifact registry.

Status: TODO (not yet implemented). When implemented it will follow the same guarded SDK / CLI fallback pattern and accept a `--dry-run` flag.

---

## Helper scripts (expanded examples)

`export_schemas.py`
- Purpose: Export canonical JSON Schema files from the Pydantic models in `src/sparkle_motion/schemas.py` into the repository `schemas/` directory. This authoring utility keeps model definitions and published JSON Schema artifacts in sync.
- Typical flags / examples:

```bash
# write schemas to the default `schemas/` dir
PYTHONPATH=src python scripts/export_schemas.py

# write to a custom output directory
PYTHONPATH=src python scripts/export_schemas.py --output-dir /tmp/sparkle-schemas
```

Notes: Run this after editing Pydantic models so the exported JSON Schema files reflect the canonical runtime contract used by prompt templates and Workflow definitions.

`render_script_agent_prompt.py`
- Purpose: Render the ScriptAgent PromptTemplate JSON using `src/sparkle_motion/prompt_templates.py` and write the result to `artifacts/prompt_templates/` (or a path you specify).
- Typical flags / examples:

```bash
# render with defaults
PYTHONPATH=src python scripts/render_script_agent_prompt.py

# render with a specific model id and output path
PYTHONPATH=src python scripts/render_script_agent_prompt.py \
	--model "gemini-1.5-pro" \
	--template-id "script_agent_movie_plan_v1" \
	--output-path artifacts/prompt_templates/script_agent_movie_plan_v1.json
```

Notes: The rendered prompt references schema artifact URIs (or local fallback paths). Validate the JSON before pushing with `scripts/push_prompt_template.py`.

`colab_drive_setup.py`
- Purpose: Colab convenience script to mount Google Drive, prepare workspace directories, optionally download model snapshots, and run a smoke test. It delegates to `src/sparkle_motion/colab_helper`.
- Typical flags / examples (run inside Colab):

```bash
# mount MyDrive/SparkleMotion and download a HF model, then run smoke check
PYTHONPATH=src python scripts/colab_drive_setup.py SparkleMotion --repo-id stabilityai/stable-diffusion-xl-base-1.0

# dry-run: show planned actions without downloading
PYTHONPATH=src python scripts/colab_drive_setup.py SparkleMotion --repo-id stabilityai/stable-diffusion-xl-base-1.0 --dry-run
```

Notes: This script imports `google.colab.drive` and therefore only runs in Colab (it will raise outside Colab). It writes artifacts into your Drive workspace and can be used to prepare the GPU environment and model snapshots.

`check_no_legacy.py`
- Purpose: A repo guard that scans `src/sparkle_motion` for banned legacy patterns (e.g., the strings "ADK-style", "legacy runner") and exits non-zero on violations. Useful for pre-commit/lint enforcement.
- Typical usage:

```bash
# run locally to assert no legacy artifacts are present
python scripts/check_no_legacy.py

# use in CI or pre-commit hook; exit code !=0 indicates violations
python scripts/check_no_legacy.py || echo "Legacy patterns found"
```

Notes: Edit `scripts/check_no_legacy.py` if you need to add or remove forbidden patterns; prefer keeping the banned list small and intentional.

## Common troubleshooting & tips

- Ensure `PYTHONPATH` includes the `src` directory when running scripts (most examples show `PYTHONPATH=src`).
- If an operation is idempotent but the ADK server returns an "AlreadyExists" error, try running with `--force` (scripts attempt to be idempotent where possible).
- To debug SDK behavior, run the script with `--dry-run` and `--verbose` (if available) so it prints the candidate SDK entrypoints it probes and the generated temporary JSON payload.
- Use `--use-cli` if you prefer to use the `adk` CLI for a particular run. The scripts will still validate input locally before invoking the CLI.

## How to contribute docs changes

- Keep the README style rules from memory in mind: do not add "Recommended next steps" sections or conversational CTAs in repo-level READMEs. Use `resources/TODO.md` or project issue tracker for tasks.
- Small doc edits should be one file per PR to keep provenance clear.

## Try it — minimal flow

1. Render and validate a prompt template without pushing:
```bash
PYTHONPATH=src python scripts/push_prompt_template.py --template script_agent_movie_plan_v1 --dry-run
```

2. Register tools (dry run first):
```bash
PYTHONPATH=src python scripts/register_tools.py --registry configs/tool_registry.yaml --dry-run
```

3. When comfortable, register with CLI (if SDK missing):
```bash
PYTHONPATH=src python scripts/register_tools.py --registry configs/tool_registry.yaml --use-cli
```

## Tests

- Unit tests live under `tests/` and cover the register scripts and workflow registration logic. Example:

```bash
conda run -n sparkle_motion bash -lc 'PYTHONPATH=.:src pytest -q tests/test_register_tools.py'
```

- For Colab-friendly checks, use the FastAPI TestClient from `starlette.testclient` to exercise `run_function_tool.py` without binding network ports.

---

If you'd like, I can also:
- Create a short `scripts/USAGE.md` with copyable command examples for common operator tasks (publish prompts, publish schemas, register tools, run a local tool), or
- Implement `scripts/publish_schemas.py` next and wire it into this README.

(If you want me to proceed with either, say which one and I'll implement it now.)
