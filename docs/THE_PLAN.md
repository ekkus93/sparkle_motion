# The Plan — UPDATED (2025-11-27)

This document is the authoritative plan for converting Sparkle Motion to an
ADK-native, WorkflowAgent-driven system. It reflects the enforced decision that
the ADK Python SDK is required for application entrypoints that instantiate
agents. This plan excludes `resources/` (samples) from runtime counts; those are
references only.

---

Authoritative runtime counts (application code only — `resources/` excluded)

- ADK agents (application runtime): 7
   - Rationale: Option A (per-tool agents) is the authoritative architecture for
      this branch. Each FunctionTool process that needs an LLM or ADK-managed
      model will instantiate its own `LlmAgent` at process startup. After
      normalizing a case-duplicate scaffold (`ScriptAgent` / `script_agent`),
      the canonical set of application FunctionTools is seven unique tools.
- FunctionTools (application-level directories under `function_tools/`): 7
   - Canonical tool directories: `script_agent`, `images_sdxl`, `videos_wan`,
      `tts_chatterbox`, `lipsync_wav2lip`, `assemble_ffmpeg`, `qa_qwen2vl`.
   - Note: there is a case-duplicate pair `ScriptAgent` / `script_agent` in
      the tree; the duplicate scaffold will be removed during normalization.

Notes:
- The `resources/` tree contains many ADK sample projects and examples. Those
  are vendor/sample code for reference only and are explicitly excluded from
  the counts above.
- ADK SDK usage in application entrypoints is mandatory: any attempt to
  instantiate an agent without the SDK or required credentials will raise and
  stop the process. No silent fallback behavior for agent construction is
  permitted in application runtime code.

---

Objectives (short)

1. Workflow-first orchestration: keep the stage graph as a WorkflowAgent YAML
   and run via ADK control plane.
2. ADK-first runtime: application entrypoints that construct agents must use
   the `google.adk` SDK and fail loudly on import/creation errors.
3. Schema-first contracts: publish canonical schemas as ADK artifacts and use
   them at runtime; allow local file fallbacks only for isolated developer
   runs.
4. Tool catalog: package heavy stages as FunctionTools with metadata and
   telemetry. Each FunctionTool process either instantiates its own agent via
   a centralized factory or raises on missing SDK.

---

Immediate actionable workstream (highest priority)

1. Design and add `src/sparkle_motion/adk_factory.py` (supporting per-tool mode)
    - Purpose: centralize ADK SDK probing, credentials checks, model selection,
       and agent construction **but support Option A (per-tool agents)**.
    - API: factory should expose `get_agent(tool_name, model_spec, mode="per-tool")`,
       `create_agent(...)`, and explicit `close_agent(name)` semantics.
    - Behavior: when `mode="per-tool"` the factory constructs a fresh
       `LlmAgent` per `get_agent` call scoped to the calling process; failures
       raise clear RuntimeErrors. The factory also documents a `shared` mode
       for future toggles but Option A is the default for this rollout.

# The Plan — UPDATED (2025-11-27)

This document is the authoritative rollout plan for converting Sparkle Motion
into an ADK-native, WorkflowAgent-driven system. It has been updated to align
with the changes made in `docs/ARCHITECTURE.md` and
`docs/IMPLEMENTATION_TASKS.md` (2025-11-27). Key updates include:

- The explicit `production_agent` role (runtime orchestrator for validated
  MoviePlan execution).
- The formal Agent vs FunctionTool separation and the decision to use Option A
  (per-tool agents) during the initial rollout.
- The mandatory `gpu_utils.model_context` pattern for adapters that load
  models, with OOM normalization to `ModelOOMError` and telemetry points.
- Per-tool gating via `SMOKE_*` flags to avoid accidental heavy installs and
  model loads in normal CI or developer runs.
- The governance requirement: any runtime dependency or manifest change must be
  proposed via `proposals/pyproject_adk.diff` and approved before edits.

---

## Authoritative runtime summary (application code only — `resources/` excluded)

- ADK agents (application runtime): 7 (Option A — per-tool agents by default)
- FunctionTools (compute adapters): 7 canonical directories
  - `script_agent`, `images_sdxl`, `videos_wan`, `tts_chatterbox`,
    `lipsync_wav2lip`, `assemble_ffmpeg`, `qa_qwen2vl`

Notes:
- `resources/` remains reference/sample code and is not counted in runtime
  totals.
- Application entrypoints that instantiate agents must require the ADK SDK
  (`require_adk()` behavior) and fail loudly when missing credentials or the
  SDK. No silent fallbacks in application runtime code.

---

## Objectives (concise)

1. WorkflowAgent-first orchestration: keep the stage graph as a WorkflowAgent
   definition and execute via ADK control plane.
2. ADK-first runtime: application entrypoints that construct agents must use
   the ADK SDK and raise on missing SDK/credentials.
3. Schema-first contracts: publish MoviePlan, AssetRefs, QAReport, and
   Checkpoint schemas as ADK artifacts and consume them at runtime.
4. Clean Agent vs FunctionTool separation: Agents = decision/orchestration;
   FunctionTools = compute adapters invoked by Agents/production_agent.
5. Deterministic testability: seeds propagate through WorkflowAgent session
   metadata; stub harnesses and torch.Generator seeding for unit tests.

---

## Immediate actionable workstream (highest priority)

1. Add `src/sparkle_motion/adk_factory.py` (per-tool agent factory)
   - Centralize SDK probing, credentials checks, and agent construction.
   - API: `get_agent(tool_name, model_spec, mode='per-tool')`, `create_agent(...)`,
     `close_agent(name)`.

2. Normalize `function_tools/` directory names
   - Remove duplicates (e.g., `ScriptAgent` vs `script_agent`) and ensure one
     canonical `entrypoint.py` per tool.

3. Implement `production_agent` at runtime
   - Responsibility: accept validated `MoviePlan` objects and execute them
     with `dry` (simulate) and `run` (execute) semantics, orchestrate steps,
     enforce policy, manage retries/backoff, and emit progress events.

4. Implement `gpu_utils.model_context` (minimal, robust pattern)
   - Context manager for guarded model load/unload, device_map presets,
     metrics snapshotting, and OOM normalization to `ModelOOMError` with
     structured metadata (`stage='load'|'inference'`, memory_snapshot).

5. Per-tool pilot & rollout
   - Pilot `videos_wan` first (largest VRAM footprint). Validate device
     mapping presets, retry/shrink strategies and fail-loud behavior.

6. Proposal & gating
   - Prepare `proposals/pyproject_adk.diff` for any runtime dependency
     additions (e.g., `torch`, `diffusers`, `chatterbox-tts`). Do not edit
     `pyproject.toml` until the proposal is reviewed and approved.

7. Tests & gating
   - Unit tests and deterministic stubs for agents/adapters.
   - Integration smoke tests gated by `SMOKE_ADK=1` (and specific flags such
     as `SMOKE_TTS=1`, `SMOKE_LIPSYNC=1`, `SMOKE_QA=1`, `SMOKE_ADAPTERS=1`).

---

## Operational constraints & hygiene

- Single-job, single-user topology remains the operational model for local
  Colab runs: tools must page models in/out and avoid keeping multiple large
  weights resident simultaneously.
- `gpu_utils.model_context` usage is mandatory for adapters that load models.
- Every adapter must emit memory telemetry via `adk_helpers.write_memory_event()`
  at critical points (post-load, post-inference, post-cleanup).
- Any change adding heavy system dependencies or binary tools requires a
  proposal diff and explicit approval.

---

## Stage-by-stage plan (hosted / local FunctionTools)

1. ScriptAgent (ADK LlmAgent)
   - Input: idea + constraints; Output: validated `MoviePlan` artifact + memory
     timeline entry. ScriptAgent must validate against the canonical MoviePlan
     schema artifact and persist raw LLM output for audit.

2. Images (SDXL FunctionTool)
   - Agent: `images_agent` (decision layer). Adapter: `images_sdxl` (Diffusers)
   - Requirements: batching rules, token-bucket rate-limiter, pre-render QA
     (text moderation or `qa_qwen2vl` sample check), deterministic stub for
     unit tests, and `gpu_utils.model_context` for pipeline loads.

3. Videos (Wan FunctionTool)
   - Agent: `videos_agent` (orchestration); Adapter: `videos_wan` (Wan2.1)
   - Requirements: chunking/sharding/reassembly semantics, overlap defaults,
     multi-GPU device_map presets, OOM fallback precedence and adaptive
     shrink heuristics. Gate heavy runs with `SMOKE_ADAPTERS`/`SMOKE_ADK`.

4. TTS (Chatterbox FunctionTool)
   - Agent: `tts_agent` (provider selection, retries, policy enforcement);
     Adapter: `tts_chatterbox` (Chatterbox / hosted fallbacks).
   - Requirements: VoiceMetadata schema, watermark awareness, provider
     registry, and gated integration tests (`SMOKE_TTS=1`).

5. Lipsync (Wav2Lip FunctionTool)
   - Adapter: `lipsync_wav2lip` with a small, testable wrapper around the
     Wav2Lip pipeline or subprocess invocation.

6. Assemble (ffmpeg FunctionTool)
   - Deterministic ffmpeg wrapper used to produce final movie artifact. Use a
     safe subprocess helper and publish as ADK artifacts.

7. QA (Qwen-2-VL FunctionTool)
   - `qa_qwen2vl` inspects frames and prompts, emits `QAReport` artifacts,
     and triggers human review or automated regenerate actions.

---

## Cross-cutting implementation rules and notes

- Agents must be small, testable, and segregate policy decisions from heavy
  compute. Agents orchestrate adapters (FunctionTools); adapters are plain,
  testable callables that use `gpu_utils.model_context`.
- Deterministic test harnesses: provide stubbed pipelines that produce
  deterministic artifacts (seed-based PNGs, predictable pHash values).
- Duplicate detection: use perceptual hashing (pHash) and a simple LRU cache
  (or Redis) to dedupe recent artifacts; `images_agent` should support
  `dedupe=True` semantics.
- OOM handling: adapters should normalize OOMs to `ModelOOMError` with
  structured metadata (stage, suggested_shrink_hint, memory_snapshots).

---

## Governance & sign-off (must be completed before code-phase)

Before changing manifests, adding large wheels, or committing runtime binary
requirements you must approve a proposal. The implementation phase will not
begin until the following checklist is signed:

- [ ] You confirm Option A (per-tool agents) and the canonical tool list above.
- [ ] Resource sizing & rollout wave owners are approved (see `docs/ROLLOUT_PLAN.md`).
- [ ] Dependency proposal (`proposals/pyproject_adk.diff`) approved for review.
- [ ] Security checklist accepted (`docs/SECURITY_CHECKLIST.md`) or owner assigned.

Reply `approve-docs` to approve these document changes and permit the next
code-phase, or `revise-docs` to request edits to this plan.

---

## Immediate next steps (practical)

1. Finalize and publish canonical schema artifacts (MoviePlan, AssetRefs,
   QAReport) and ensure `configs/schema_artifacts.yaml` points to artifact URIs.
2. Implement a minimal `gpu_utils.model_context` in `src/sparkle_motion/gpu_utils.py`
   (a light, dependency-free implementation suitable for unit tests) and the
   `ModelOOMError` domain exception.
3. Create deterministic stub harnesses for `images_agent` and `videos_agent`
   unit tests to validate batching, dedupe, chunking, and OOM fallback logic.
4. Prepare `proposals/pyproject_adk.diff` that lists proposed runtime
   dependencies for review (do not apply until approved).

If you want, I can now scaffold one of these items locally (no manifest edits):
- Option A: `videos_agent` chunk-splitting utility + unit tests (deterministic)
- Option B: `tts_agent` voice registry helper + VoiceMetadata model and tests
- Option C: minimal `gpu_utils.model_context` implementation + tests

---

This document is now aligned to the authoritative architecture and the
implementation tasks; it is intended to be the single source of truth for the
planned rollout and the sign-off workflow.