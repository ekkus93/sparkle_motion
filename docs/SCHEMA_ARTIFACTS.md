# Schema Artifact Reference

This document enumerates the canonical schema and policy artifacts referenced by
`sparkle_motion.schema_registry` and `configs/schema_artifacts.yaml`. Update this
file whenever a schema URI or version changes so agents/tools stay aligned.

| Contract | Artifact URI | Local fallback (dev/test) | Status / Notes |
| --- | --- | --- | --- |
| MoviePlan | `artifact://sparkle-motion/schemas/movie_plan/v1` | `artifacts/schemas/MoviePlan.schema.json` | Canonical schema for `script_agent` output + production validation. |
| AssetRefs | `artifact://sparkle-motion/schemas/asset_refs/v1` | `artifacts/schemas/AssetRefs.schema.json` | Used by `adk_helpers.publish_artifact()` metadata and downstream tooling. |
| QAReport | `artifact://sparkle-motion/schemas/qa_report/v1` | `artifacts/schemas/QAReport.schema.json` | Consumed by `qa_qwen2vl` + QA policy checks. |
| StageEvent | `artifact://sparkle-motion/schemas/stage_event/v1` | `artifacts/schemas/StageEvent.schema.json` | WorkflowAgent + `production_agent` timeline payload schema. |
| Checkpoint | `artifact://sparkle-motion/schemas/checkpoint/v1` | `artifacts/schemas/Checkpoint.schema.json` | Optional per-stage checkpoint manifests for resume flows. |
| QA policy bundle | `artifact://sparkle-motion/qa_policy/v1` | `artifacts/qa_policy/qa_policy_v1.tar.gz` (manifest: `artifacts/qa_policy/v1/manifest.json`) | Policy pack consumed by QA + human-review gates; update manifest when policy revisions ship. |

## Next steps

1. **Publish confirmations** – ensure each URI above resolves inside the ADK
   ArtifactService for the current environment (Colab local profile). Document
   publish commands or runbook links when we next rotate versions.
2. **Version bumps** – when schemas evolve, bump both the `configs/schema_artifacts.yaml`
   entry and this table. Include migration/compatibility notes inline so agents
   know whether mixed-version operation is supported.
3. **Schema registry tests** – add/extend unit tests under
   `tests/unit/test_schema_registry.py` (or equivalent) that load
   `configs/schema_artifacts.yaml`, verify required keys, and assert the table
   above stays in sync (e.g., by parsing this file or a shared data source).
4. **Doc linkage** – `sparkle_motion.schema_registry` docstrings and the
   onboarding guide (`docs/HowToRunLocalTools.md`) both point back to this table.
   Keep those references updated whenever you change artifact URIs so new
   contributors land on the canonical schema list immediately.
