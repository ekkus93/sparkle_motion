# Sparkle Motion release notes

## 2025-12-01 — Agent naming cleanup

- **What changed:** Only `script_agent` (MoviePlan generation) and `production_agent`
  (WorkflowAgent runtime) retain the `_agent` suffix. Every other runtime component
  now appears as a stage module or FunctionTool, and the canonical mapping is:

  | Old name | New name | Notes |
  | --- | --- | --- |
  | `images_agent` | `images_stage` | FunctionTool shim over `function_tools/images_sdxl`; telemetry prefixed `images_stage.*`. |
  | `videos_agent` | `videos_stage` | Orchestrates Wan chunking plus retry telemetry under the `videos_stage.*` prefix. |
  | `tts_agent` | `tts_stage` | Provider-selection stage calling `function_tools/tts_chatterbox`; emits `tts_stage.*` telemetry. |

  FunctionTools such as `images_sdxl`, `videos_wan`, `tts_chatterbox`,
  `lipsync_wav2lip`, and `assemble_ffmpeg` keep their adapter
  names. Finalize now emits deliverables directly after assemble, and the
  notebook control panels no longer surface the old reviewer prompts. Tool IDs, telemetry namespaces, notebooks, and CLI output now match
  this terminology, and `docs/ARCHITECTURE.md#_agent-naming-matrix` is the only
  place that references the legacy suffixes for historical context.
- **Why it matters:** Earlier docs mixed agents and FunctionTools, which caused
  orchestration confusion and inconsistent port map entries. The cleanup makes it
  explicit that only two ADK agents exist; everything else is a callable tool or
  stage invoked by `production_agent`.
- **Action required for users:**
  - If you hard-coded `*_agent` identifiers in local notebooks, CLI scripts, or
    ToolRegistry overrides, replace them with the stage/FunctionTool names above
    so only `script_agent`/`production_agent` retain the suffix.
  - Confirm your `configs/tool_registry.yaml` and `configs/workflow_agent.yaml`
    pull the updated IDs (`script_agent`, `production_agent`, `images_stage`,
    `videos_stage`, `tts_stage`, `images_sdxl`, etc.) before re-registering the
    workflow.
  - Re-run the notebook control panel quickstart so widget labels, status panes,
    and artifact viewers pick up the clarified naming.
- **Related documentation:** See `README.md` (§Agents vs FunctionTools),
  `docs/ARCHITECTURE.md#_agent-naming-matrix`, and
  `docs/NOTEBOOK_AGENT_INTEGRATION.md` (Agent + FunctionTool port map) for the
  canonical explanations that mirror this release note. Full-suite regression
  status: `PYTHONPATH=.:src pytest -q` (326 passed, 1 skipped, 2025-12-01).
