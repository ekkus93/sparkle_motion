# Pipeline Architecture v2 — Orchestrated, Pluggable Workflow

## Objectives
- Decouple agents from concrete tools (e.g., Wav2Lip) so any stage can be swapped, disabled, or iterated independently.
- Represent the workflow as an explicit DAG/state machine with typed contracts between stages.
- Centralize configuration (ports, models, provider selection) so ToolRegistry + env vars remain the single source of truth.
- Improve debuggability: per-stage logging, retries, health signals, and artifact lineage.
- Enable partial pipelines (e.g., skip lipsync) and future task insertion without rewriting agents.

## Pain Points in v1
- **Tight coupling**: ScriptAgent/ProductionAgent import Wav2Lip helpers directly and assume directory layouts (`lipsync_wav2lip`).
- **Magic configuration**: Ports, model paths, and flags copied into notebooks, docs, and scripts instead of reading from shared config.
- **Opaque workflow**: Execution order is implicit in agent code; it is difficult to visualize or short-circuit steps.
- **Difficult testing**: GPU-heavy dependencies are dragged into every code path, making CI and local dev brittle.

## Target Architecture Overview

```
┌──────────────┐   LipSyncTask   ┌──────────────┐   VideoTask   ┌──────────────┐
│ ScriptAgent  │  ───────┬─────▶ │ LipSync Node │ ───────┬─────▶│ Video Node   │
└──────┬───────┘          │      └──────┬──────┘         │      └──────┬──────┘
		 │          Provider Registry     │        Provider Registry     │
		 │                                 ▼                             │
		 └──────────────▶ Artifact Store ◀─┴─────────────────────────────┘
```

### Core building blocks
1. **Workflow DAG Runner**: Reads a workflow definition (YAML/JSON) that lists stages, dependencies, retry policy, and provider bindings. Executes nodes when prerequisites succeed.
2. **Task Interfaces**: Each capability (e.g., `ScriptGenerationTask`, `LipSyncTask`, `VideoAssemblyTask`, `QAReviewTask`) exposes `run(request: Contract) -> Result`. Implemented as Python Protocols or ABCs with fully typed inputs/outputs.
3. **Provider Registry**: Runtime factory that instantiates concrete implementations based on config. Defaults (e.g., `Wav2LipProvider`) live beside alternative providers; selection is config-only.
4. **Artifact Contracts**: Pydantic/dataclass schemas describing serialized artifacts (metadata JSON, media descriptors). Tasks use these contracts rather than raw filesystem conventions.
5. **Shared Infrastructure**: Artifact store (local fs or GCS), settings service (ToolRegistry), logging/metrics pipeline.

## Detailed Components

### 1. Workflow Definition
- Located at `configs/pipeline/workflow.yaml` (example).
- Declares nodes with fields: `id`, `task`, `inputs`, `outputs`, `provider`, `retries`, `timeout`, `deps`.
- Supports conditional execution (`when`, `skip_if`) and parallel branches.

### 2. Task Contracts
- Maintain `contracts/` package containing shared dataclasses/pydantic models.
- Example `LipSyncRequest`: `{ audio_track: ArtifactRef, video_stub: ArtifactRef, language: str, metadata: dict }`.
- Example `LipSyncResult`: `{ aligned_video: ArtifactRef, provider: str, metrics: dict }`.
- Contracts versioned to allow evolution (`schema_version`).

### 3. Provider Implementations
- Live under `sparkle_motion/providers/<capability>/<provider_name>.py`.
- Each provider registers itself via entry points or explicit config key.
- Example `Wav2LipProvider`:
  - Reads model paths from `ToolRegistry` (no inline constants).
  - Interacts with artifact store via `ArtifactClient` abstraction.
  - Emits `LipSyncResult` regardless of internal file layout.
- Future providers (e.g., “Audio2Face”, “Dubber”) implement the same interface; selection occurs in config.

### 4. Provider Registry API
```python
registry = ProviderRegistry.from_config("configs/tool_registry.yaml")
lipsync = registry.resolve("lipsync")  # returns LipSyncTask implementation
result = lipsync.run(my_request)
```
- Registry caches instances, injects shared dependencies (logger, artifact client, temp dirs).
- Supports feature flags (`lipsync.enabled = false`) or failover lists (`providers: [wav2lip, fallback_stub]`).

### 5. Artifact & Storage Layer
- `ArtifactClient` abstracts over local FS vs GCS.
- Artifacts referenced via `ArtifactRef` (path + metadata + checksum).
- Tasks never assume absolute paths; orchestrator passes handles and manages lifecycle (cleanup, retention).

### 6. Observability
- Each node execution emits structured event: `{node_id, provider, status, latency_ms, retries, artifacts}`.
- Metrics sink (stdout, BigQuery, Prometheus) configurable.
- DAG runner exposes `/status` endpoint showing per-node state for ongoing runs.

## Execution Flow
1. **Initialization**
	- Load workspace/env, parse workflow definition, build DAG graph.
	- Initialize ProviderRegistry + ArtifactClient; inject into nodes.
2. **Run**
	- Scheduler walks DAG, enqueues runnable nodes, tracks dependencies.
	- Each node receives validated contract data, calls provider, stores outputs back via ArtifactClient, and emits events.
	- Failures trigger configured retry policy; exhaustion marks node as failed and can halt dependent nodes.
3. **Completion**
	- Final artifacts collected; summary generated (per-node metrics, artifact map, logs pointer).

## Configuration Story
- All tool settings (ports, hostnames, model checkpoints) reside in `configs/tool_registry.yaml` + per-provider overrides.
- `.env_sample` only contains environment selectors (workspace, secret references). Actual values read by registry at runtime.
- Notebook/CLI start-up steps parse ToolRegistry and expose UI toggles for enabling/disabling tasks.

## Extensibility Scenarios
- **Swap Wav2Lip**: Add new provider module, register under `providers.lipsync.alt`, change config from `wav2lip` to `alt`, rerun pipeline.
- **Disable Lipsync**: Set `lipsync.enabled=false` or use a `NoOpLipSyncProvider` that passes through audio references.
- **Insert QA Policy Node**: Add new node to workflow definition referencing `QAReviewTask`; upstream/downstream dependencies defined declaratively.

## Migration Plan
1. **Discovery**: Document current implicit pipeline steps, inputs, outputs (done in this spec).
2. **Contracts First**: Define request/result models for every stage; add serialization helpers.
3. **Provider Extraction**: Wrap existing Wav2Lip/video assembly logic in provider classes implementing the new interfaces.
4. **Registry + Runner**: Implement ProviderRegistry and DAG runner; run pipeline in “shadow mode” where old orchestrator still executes but new runner logs outputs.
5. **Cutover**: Switch ScriptAgent/ProductionAgent to call DAG runner, retire legacy inlined calls.
6. **Cleanup**: Remove deprecated magic numbers, update docs/tests, enforce contracts via CI (schema validation + contract tests).

## Open Questions / Follow-ups
- Should workflow definitions become versioned assets stored with artifacts for reproducibility?
- Do we need multi-tenant isolation per workspace, or is a single registry sufficient?
- How do we expose partial reruns (e.g., rerun only video stage) via CLI/UI?
- What governance is needed for provider plugins (signing, review)?

Answering these will guide additional iterations, but this architecture provides the structure needed to keep agents decoupled and make tool swaps (like removing Wav2Lip) straightforward.
