# TODO — Sparkle Motion (ADK-native rebuild)

This list replaces the legacy “ADK-style" backlog. Every task assumes we are
starting fresh with the ADK-first architecture described in `THE_PLAN.md` and
`ARCHITECTURE.md`. No effort will be spent patching the old runner; we will
delete or quarantine it once the new pipeline is live.

**Compute note:** tasks below target today’s setup where all heavy models run
inside a single Google Colab A100 session. Because the full toolchain cannot
live in VRAM simultaneously, every stage explicitly loads/unloads its model
before handing the GPU to the next tool. Each stage is defined as an ADK
FunctionTool boundary so we can later redirect it to hosted APIs (OpenAI
GPT-5.1, ElevenLabs, etc.) without reworking the workflow. All stateful ADK
services (SessionService, ArtifactService, MemoryService, secrets, telemetry)
use the **local-colab** profile: SQLite databases under `/content/`, mounted
Google Drive for artifacts, filesystem `.env` secrets, telemetry disabled.

## 0) Legacy teardown & guardrails — STATUS: completed (2025-11-25)

- Deliverable: clean foundation that prevents the old codepath from reappearing.
- Subtasks:
  - Archive/remove the legacy runner, adapters, and local service shims.
    - Directories `src/sparkle_motion/`, `schemas/`, and `tests/` were wiped
      (manual verification on 2025-11-25) so the legacy runner no longer exists.
  - Add CI/linters that fail if deprecated modules are imported (temporary until
    deletion is confirmed).
    - `scripts/check_no_legacy.py` plus the `make lint` target now fail if any
      file reintroduces "ADK-style" or legacy runner artifacts.
  - Update docs/readmes to warn contributors that only the ADK workflow is
    supported going forward.
    - Architecture/THE_PLAN already specify ADK-only; contributors must follow
      those docs when rebuilding the pipeline.

## 1) ADK project bootstrap — STATUS: completed (2025-11-25)

- Deliverable: the **local-colab** ADK profile wired to SQLite + Google Drive +
  filesystem secrets so the workflow can run entirely inside one Colab runtime,
  along with the manifest + bootstrap script that keep it reproducible.
- Proof (2025-11-25):
  - `resources/adk_projects.json` now declares service-account env vars for every
    WorkflowAgent/FunctionTool role, and `scripts/bootstrap_adk_projects.sh`
    creates/refreshes the SQLite DBs, Drive artifact root, and `.sparkle.env`
    template (including placeholders for all credentials).
  - `resources/ADK_PROJECT_BOOTSTRAP.md` documents Drive mounting, running the
    bootstrap script, verifying outputs, and populating service accounts, plus a
    dedicated IAM section.
  - `notebooks/sparkle_motion.ipynb` contains Drive-mount and python-dotenv
    cells so Colab operators can run the bootstrap flow end-to-end.
Update configs/tool_registry.yaml to include an explicit local-colab endpoint

## 2) Schema + config publishing — STATUS: in-progress

- Deliverable: canonical MoviePlan, AssetRefs, QAReport, and Checkpoint schemas
  plus QA policy bundles published as ADK artifacts.
- Current state: schema export helpers and some schema artifacts exist in
  `schemas/` (see `scripts/export_schemas.py`). `configs/qa_policy.yaml` and
  `scripts/package_qa_policy.py` were added to build QA bundles. Remaining work
  is to publish these artifacts to ADK (`adk artifacts push` or SDK) and pin
  artifact URIs in `configs/schema_artifacts.yaml`.
- Subtasks:
  - [x] Add schema export helper (`scripts/export_schemas.py`) and generate local JSON Schema files.
  - [x] Author a QA policy and packaging script (`configs/qa_policy.yaml`, `scripts/package_qa_policy.py`).
  - [ ] Publish schemas to ADK artifact registry and record artifact URIs in `configs/schema_artifacts.yaml`.
  - [ ] Add a small `scripts/publish_schemas.py` helper (CLI + guarded SDK/CLI support).
  - [ ] Add a local-only publish option: copy `schemas/*.schema.json` into `artifacts/schemas/` and pin `file://` (or repo-relative) URIs in `configs/schema_artifacts.yaml` for isolated servers without ADK credentials. Note: local-only artifacts are developer caches and not published to the ADK control plane.

- Deliverable: canonical MoviePlan, AssetRefs, QAReport, and Checkpoint schemas
  plus QA policy bundles published as ADK artifacts.
- Subtasks:
  - Revise the Pydantic models as needed, export JSON Schemas, and push them to
    `artifact://sparkle-motion/schemas/...` with semantic versioning.
  - Publish QA policy YAML + schema bundles; document how WorkflowAgent loads
    them at runtime.
  - Update ScriptAgent prompt templates to fetch schemas from ADK artifacts
    instead of embedding local copies.

## 3) FunctionTool packaging — STATUS: in-progress

- Deliverable: packaged, versioned FunctionTools for every stage (ScriptAgent,
  SDXL, Wan, Chatterbox TTS, Wav2Lip, Assemble, QA) with metadata suitable for
  registration in ADK ToolRegistry.
- Current state: scaffolding exists — `function_tools/` doc, initial Dockerfile
  contexts (dev-only), `scripts/run_function_tool.py` (Colab runner), and
  `configs/tool_registry.yaml` with `local-colab` endpoints and ports. A small
  `src/sparkle_motion/tool_registry.py` helper was added to load these configs.
- Subtasks (actionable):
  - [x] Add per-tool scaffolding & README (`function_tools/*`).
  - [x] Provide a Colab-friendly runner `scripts/run_function_tool.py` that resolves host/port from `configs/tool_registry.yaml`.
  - [ ] Implement `scripts/register_tools.py` to register each tool in ADK ToolRegistry (guarded SDK/CLI, idempotent).
  - [ ] Harden per-tool entrypoints to emit health, /invoke contract, and telemetry hooks.
  - [ ] Add per-tool test harnesses (smoke test producing artifacts under tmp runs).
  - [ ] (Optional) Produce IaC manifests (Cloud Run / Vertex) for future hosted deployments.

Note: The Colab runtime cannot run Docker. For the `local-colab` profile start
FunctionTools in-process using the runner script. Container builds target
future hosted runtimes and are optional for local development.

## 4) WorkflowAgent definition — STATUS: not started

- Deliverable: WorkflowAgent YAML/JSON encoding the stage graph `script ->
  images -> videos -> tts -> lipsync -> assemble -> qa` with retry policies,
  event actions, and resume semantics.
- Subtasks:
  - [ ] Draft WorkflowAgent YAML with stage nodes referencing FunctionTool IDs and schema artifact URIs.
  - [ ] Add retry/backoff + jitter semantics and QA-driven auto-regenerate rules.
  - [ ] Implement deployment helper `scripts/register_workflow.py` that calls `adk workflows deploy` (guarded SDK/CLI).
  - [ ] Add small unit/integration tests that validate resume/retry behavior against a local-colab sandbox.

## 5) Human + QA governance — STATUS: completed (2025-11-26)

- Deliverable: ADK-native human review and QA escalation flows for script and
  visual assets.
- Current state: `configs/qa_policy.yaml` and `scripts/package_qa_policy.py` are present; the QA tool scaffold exists in `function_tools/`.
- Subtasks:
  - [x] Author baseline QA policy and packaging script.
  - [ ] Implement `qa_qwen2vl` FunctionTool to emit `QAReport` artifacts and policy decisions.
  - [ ] Wire WorkflowAgent to auto-regenerate or escalate based on QA decisions.
  - [ ] Ensure MemoryService logs reviewer decisions and seeds (add tests).

## 6) Operator experience — STATUS: in-progress

- Deliverable: CLI + Colab notebook that authenticate against ADK, launch
  workflow runs, stream status, and download artifacts — no local orchestration.
- Current state: `notebooks/sparkle_motion.ipynb` contains basic Drive mount and dotenv helper cells; `scripts/push_prompt_template.py` exists for prompt publishing.
- Subtasks:
  - [ ] Implement CLI wrapper `sparkle-motion` that uses ADK SDK/CLI to run workflows and tail timelines.
  - [x] Ensure Colab notebook mounts Drive, loads secrets, and can call local runners; upgrade it to call `adk workflows run` when credentials are available.
  - [ ] Document operator credential setup and run lifecycle in `docs/ORCHESTRATOR.md`.

## 7) Observability, seeds, and resume — STATUS: in-progress

- Deliverable: unified observability + determinism story driven entirely by ADK
  telemetry and resume APIs.
- Current state: runner emits local logs; MemoryService wiring exists in
  `src/sparkle_motion/services.py` (session/memory usage). Resume/retry helpers
  are planned as WorkflowAgent features.
- Subtasks:
  - [x] Keep telemetry export disabled in `local-colab` profile and document the notebook/CLI logging flow.
  - [ ] Ensure seeds propagate via session metadata and add tests for determinism.
  - [ ] Implement `workflow_runs.resume` validation tests against the local sandbox.

## 8) Data egress + asset delivery — STATUS: not started

- Deliverable: artifact lifecycle covering final movies, per-shot assets, and
  run manifests served straight from ADK ArtifactService.
- Subtasks:
  - [ ] Finalize artifact naming and versioning conventions and document them.
  - [ ] Implement `scripts/download_artifacts.py` to fetch artifacts by ADK URI.
  - [ ] Ensure artifact outputs are downloadable as signed URLs when hosted.

## 9) Verification & deprecation of legacy code — STATUS: not started

- Deliverable: proof that the ADK workflow supersedes the old runner and that
  all tests/docs reference the new path.
- Subtasks:
  - [ ] Author integration tests that call WorkflowAgent (SDK stubs/local sandbox) to validate full runs and QA regenerate paths.
  - [ ] Update docs and onboarding to point exclusively at ADK workflows; remove legacy runner docs.
  - [ ] Archive/delete obsolete legacy modules and ensure `scripts/check_no_legacy.py` flags regressions in CI.

---

Tracking expectations
- Use this file to capture high-level deliverables and rationale.
- Mirror task status in the workspace tracker so automation stays accurate.
- When a section is completed, summarize the proof (deployments, test IDs,
  doc links) before flipping the status.
