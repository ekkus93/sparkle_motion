Per-stage contract — Sparkle Motion orchestrator
===============================================

This document summarizes the per-stage contract the orchestrator expects and
the checkpoint format used to support resume and retry semantics.

> **QA automation status (2025-12-07):** The automated QA stages and
> `qa_qwen2vl` FunctionTool have been fully removed from the runtime. The
> orchestrator currently ends with the `assemble` stage, produces final media
> artifacts, and relies on human review or downstream systems for any QA needs.
> This document now describes the QA-free pipeline. When QA is reintroduced we
> will restore the retired sections and call out the new gating behavior.

1) Stage callable signature
---------------------------

Each stage implementation (adapter) must implement the callable with this
signature:

    def stage_fn(movie_plan: MoviePlan, asset_refs: AssetRefs, run_dir: Path) -> AssetRefs:

Inputs:
- `movie_plan` — validated `MoviePlan` object describing the whole run.
- `asset_refs` — an `AssetRefs` object (may be partially populated) that the
  stage should update with newly created asset references (file paths, URIs).
- `run_dir` — Path to the run-specific directory where stages may persist
  temporary files, checkpoints, and final artifacts.

Output:
- Return an updated `AssetRefs` instance (or a dict compatible with it).

Side-effects:
- Each stage must write a per-stage checkpoint file at
  `runs/<run_id>/checkpoints/<stage>.json` using the `Checkpoint` schema. The
  runner relies on this file to decide whether to skip the stage during `resume`.
- Stages may also write artifact files (images, audio, videos) under `run_dir`.

### Stage contract matrix

| Stage name | Purpose / external dependency | Inputs consumed | Fields written in `AssetRefs` | Recommended checkpoint metadata |
| --- | --- | --- | --- | --- |
| `script` | Normalize `MoviePlan`, ensure every shot has an `AssetRefsShot` entry (no heavy model). | `MoviePlan.shots`, `MoviePlan.characters` | Creates `asset_refs.shots[shot_id]` dictionaries with empty `start_frame`, `end_frame`, `raw_clip`, `dialogue_audio`, `final_video_clip`. | `{"shots_initialized": <int>}` plus an `adapter` label (`"script_stub"`). |
| `images` | Call SDXL (or fallback) to render keyframe PNGs. | `MoviePlan.base_images` via `ShotSpec.start_base_image_id` / `ShotSpec.end_base_image_id`, optional `MoviePlan.metadata["seed"].` | Populates `start_frame`/`end_frame` paths per shot. | `{"frames_written": <int>, "adapter": "sdxl"}` and optionally `seed` used. |
| `videos` | Call Wan / video adapter to create motion clips. | Newly written keyframe paths, `ShotSpec.motion_prompt`, `ShotSpec.duration_sec`. | Updates `raw_clip` per shot. | `{"clips_written": <int>, "adapter": "wan"}`. |
| `tts` | Generate dialogue WAVs via TTS adapter (per-line synthesis). | `ShotSpec.dialogue` and `characters.voice_profile`. | Stores ordered per-line WAV paths in `dialogue_audio` and Step metadata for downstream tooling. | `{"lines_synthesized": <int>, "voice_model": "polyglot-v1", "line_artifacts": [...]}`. |
| `lipsync` | Wav2Lip (or stub) to merge audio + raw video. | `raw_clip`, `dialogue_audio`. | Writes `final_video_clip` per shot. | `{"clips_synced": <int>, "adapter": "wav2lip"}`. |
| `assemble` | Concatenate final clips, add BGM, output movie. | `final_video_clip` entries, `MoviePlan.metadata` (e.g., frame rate). | May record top-level extras such as `asset_refs.extras["final_movie"] = str(path)`. | `{"final_path": "movie_final.mp4", "duration": 12.4, "video_codec": "mpeg4", "audio_codec": "aac"}`. |

Stages run in the order defined above. Each stage is responsible for:
1. Reading only the inputs listed in the table (other keys are considered implementation detail and should not be relied upon).
2. Writing artifacts to subdirectories of `run_dir` (never outside of the run sandbox).
3. Updating `asset_refs` atomically (mutate then persist `runs/<run_id>/asset_refs.json`).
4. Emitting a checkpoint JSON payload that captures the stage status and enough metadata for operators to reason about retries.

**Stub adapters for local runs.** Wan, TTS, Wav2Lip, and assemble adapters ship with deterministic fallbacks backed by `src/sparkle_motion/adapters/stub_adapter.py`. These helpers generate minimal-yet-valid PNG/MP4/WAV artifacts (leveraging `ffmpeg` when available) so that `tests/test_smoke.py` and Colab dry runs produce the same filesystem layout as production, minus the heavyweight models. Replacing them with real integrations only requires swapping the adapter implementation; the orchestrator contract remains unchanged.

### Service wiring & tool catalog

- **SessionService** (`src/sparkle_motion/services.py`) issues the `run_id`, prepares `runs/<run_id>/` (artifacts, checkpoints, human-review folders), and hands the runner a `SessionContext`. Plugging in the official ADK service later is a matter of passing a different implementation into `Runner(session_service=...)`.
- **ArtifactService** records every persisted output in `runs/<run_id>/artifacts.json` so humans/agents have a canonical lookup for Drive paths or URIs. Stage adapters call `artifact_service.register(name, path)` after writing files.
- **MemoryService** appends structured events (stage begin/success/fail plus human-review notes) to `runs/<run_id>/memory_log.json`. This is the long-lived log ADK agents or dashboards can consume.
- **ToolRegistry** mirrors ADK’s FunctionTool catalog. Each stage registers its callable/metadata with the registry, and the runner resolves stages via registry lookups, which allows hot-swapping adapters (local vs. remote) without editing orchestrator logic.

These abstractions keep the runner faithful to ADK patterns while remaining lightweight for the current Colab workflow.

### Observability, logging, and retries

- The `run_manifest.retry` decorator (see `src/sparkle_motion/run_manifest.py`) records
  `begin`/`fail`/`success` events with timestamps, attempts, and optional error
  strings. Defaults: `max_attempts=3`, `base_delay=0.5s`, exponential
  backoff capped at 30s, `jitter=0.2`.
- `MemoryService.record_event(event_type, payload)` writes structured entries of
  the form `{"timestamp": <float>, "event_type": <str>, "payload": <dict>}` to
  `memory_log.json`. `MemoryService.list_events()` returns a copy for
  aggregation or API responses.
- `observability.write_run_events_log(...)` merges manifest events and memory
  entries into `run_events.json`, producing a chronological timeline with
  `source="stage"` or `source="memory"`. The runner registers this artifact as
  `run_events` so operators download a single JSON file for debugging or
  dashboards.

#### Smoke test coverage

- `tests/test_smoke.py` exercises the default stubbed adapters end-to-end using
  a tiny `MoviePlan`. It asserts the run directory contains `movie_plan.json`,
  `asset_refs.json`, generated media assets, and the merged `run_events.json`
  timeline. This test runs quickly (<1s) and serves as a CI gate before wiring
  heavier adapters or Colab workflows.

#### Operator playbook — inspecting `line_artifacts`

1. Fetch memory events: `python -m sparkle_motion.tools.memory_dump --run-id <id>` (or `adk memory-events list --run-id <id>`) to stream the structured events emitted by `production_agent`.
2. Filter for `event_type == "tts.line_synthesized"` (or check
  `StepExecutionRecord.meta['tts']['line_artifacts']` inside the
  `stage_progress` events). Each payload includes `line_index`, `voice_id`,
  `provider_id`, `artifact_uri`, `duration_s`, `sample_rate`, and the
  `watermarked` flag.
3. To trace a specific dialogue line, copy its `artifact_uri` into
  `adk artifacts download <uri>` (or open the local `runs/<run_id>/audio/...`
  path) and compare against downstream lipsync entries. Because we write one
  record per line, the ordering matches the original script text and stays
  deterministic even when runs are resumed mid-stage.

#### Stage deep-dive

Below is the canonical contract for each stage. When multiple adapters exist (e.g., SDXL vs. fallback), they must satisfy the same observable behavior and metadata.

##### `script`
- **Reads**: the entire `MoviePlan` plus referenced `CharacterSpec` entries.
- **Writes**: initializes `asset_refs.shots[shot_id]` with placeholders; no filesystem assets are produced.
- **Checkpoint metadata**: include `shots_initialized`, `characters_linked`, and the adapter name.
- **Failure modes**: invalid `ShotSpec` references, duplicate shot IDs.

##### `images`
- **Reads**: prompts from each shot plus optional global seed.
- **Writes**: PNG paths saved under `run_dir/frames/<shot_id>_{start,end}.png` and stored in `asset_refs`.
- **Metadata**: `frames_written`, `adapter`, `seed`, and `avg_latency_ms` for observability.
- **Notes**: Must detect missing prompts early and fail-fast.

##### `videos`
- **Reads**: keyframe paths, motion prompt, clip duration, optional fps.
- **Writes**: MP4 files at `run_dir/videos/<shot_id>.mp4` stored as `raw_clip`.
- **Metadata**: `clips_written`, `adapter`, `avg_decode_time_ms`.
- **Notes**: Should propagate reference to the keyframes used for traceability.

##### `tts`
- **Reads**: `ShotSpec.dialogue` text plus per-character `voice_profile`. The production agent resolves line-level `voice_config` using the plan’s `characters.voice_profile` map and passes it through to `tts_stage.synthesize()`.
- **Writes**: WAV files per line under `run_dir/audio/<shot_id>/<line>.wav`, publishes each clip via `adk_helpers.publish_artifact(artifact_type="tts_audio")`, and stores the resulting list (ordered to match dialogue indexes) in `asset_refs.shots[shot_id].dialogue_audio`.
- **Metadata**: `StepExecutionRecord.meta["tts"]` now records `line_artifacts` (line index, provider_id, voice_id, artifact_uri, duration, sample_rate, bit_depth, watermark flag) plus aggregate duration and `dialogue_paths`. Stages that depend on dialogue downstream (lipsync, assemble) should rely on those metadata entries instead of re-reading the filesystem.
- **Env gating**: Real adapters only execute when `SMOKE_TTS=1` (or `SMOKE_ADAPTERS=1`) is set; otherwise the fixture adapter produces deterministic WAV placeholders. This keeps Colab/CI runs inexpensive while still emitting artifact metadata that matches production structure.

##### `lipsync`
- **Reads**: `raw_clip` and `dialogue_audio` lists for each shot.
- **Writes**: muxed clips under `run_dir/lipsync/<shot_id>.mp4` updating `final_video_clip`.
- **Metadata**: `clips_synced`, `adapter`, `avg_rtf` (real-time factor).

##### `assemble`
- **Reads**: list of final per-shot clips plus global metadata (fps, soundtrack path).
- **Writes**: final MP4 (and optional thumbnails) stored in `asset_refs.extras`.
- **Metadata**: `final_path`, `duration_sec`, `video_codec`, `audio_codec`, `shots_used` (ordered list).


2) Checkpoint format
---------------------

Every stage writes exactly one checkpoint file named `runs/<run_id>/checkpoints/<stage>.json`.
The JSON payload must include the following fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `stage` | `str` | ✅ | Stage identifier (e.g., `"images"`). |
| `status` | `"begin" | "success" | "failed"` | ✅ | State of the most recent attempt. |
| `timestamp` | `float` (epoch seconds) | ✅ | When the status was recorded. |
| `attempt` | `int` | ✅ | 1-based attempt count for visibility when retries trigger. |
| `error` | `str | null` | ❌ | Present only when `status == "failed"`. |
| `metadata` | `dict[str, Any]` | ✅ (can be empty) | Stage-specific summary: counts, codec names, adapter id, etc. |

To keep the format machine-validated, we export the `Checkpoint` JSON Schema via `src/sparkle_motion/schemas.py`. Abbreviated schema (draft-07):

```json
{
  "$id": "sparkle_motion/checkpoint.schema.json",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["stage", "status", "timestamp", "attempt", "metadata"],
  "properties": {
    "stage": {"type": "string"},
    "status": {"type": "string", "enum": ["begin", "success", "failed"]},
    "timestamp": {"type": "number"},
    "attempt": {"type": "integer", "minimum": 1},
    "error": {"type": ["string", "null"]},
    "metadata": {"type": "object"}
  },
  "additionalProperties": false
}
```

Adapters should extend the `metadata` object with stage-specific keys but avoid removing the base properties. If a stage needs nested structures (e.g., per-shot stats), nest them under a descriptive key such as `"shots": {"s1": {...}}`.

Example success checkpoint:

```json
{
  "stage": "images",
  "status": "success",
  "timestamp": 169...,
  "attempt": 1,
  "error": null,
  "metadata": {
    "frames_written": 24,
    "adapter": "sdxl_stub"
  }
}
```

Example failure checkpoint:

```json
{
  "stage": "tts",
  "status": "failed",
  "timestamp": 169..., 
  "attempt": 2,
  "error": "Timeout contacting TTS service",
  "metadata": {
    "lines_requested": 12,
    "adapter": "tts-cloud"
  }
}
```

Status values:
- `begin` — stage started (optional)
- `success` — stage completed successfully
- `failed` — stage failed and should be retried or investigated

#### Checkpoint file layout

- Directory: `runs/<run_id>/checkpoints/`
- Filename: `<stage>.json` (e.g., `images.json`)
- Write strategy: write to `<stage>.json.tmp` then atomically replace to avoid half-written JSON.
- Lifecycle: checkpoints are append-only per attempt (each retry overwrites with the latest state plus incremented `attempt`).

3) Manifest events
------------------

The `RunManifest` (see `src/sparkle_motion/run_manifest.py`) records the
chronological event stream for each stage. Every invocation of a stage should
emit the following sequence, even when retries fire:

1. `begin` — inserted exactly once per attempt (attempt number is incremented).
2. `fail` — emitted when the wrapped stage raises before max attempts are exhausted.
3. `success` — emitted once when a stage completes and returns updated `asset_refs`.

### Event schema

Events are instances of `StageEvent` persisted as JSON objects with these fields:

| Field | Type | Description |
| --- | --- | --- |
| `run_id` | `str` | The run directory identifier (e.g., `run_20231124_153000`). |
| `stage` | `str` | Stage name (`script`, `images`, ...). |
| `status` | `"begin" | "success" | "fail"` | Current state of the attempt. |
| `timestamp` | `float` | Epoch seconds when the event was written. |
| `attempt` | `int` | 1-based counter that matches the checkpoint `attempt` field. |
| `error` | `str | null` | Present for `fail` events; contains a concise error string. |
| `metadata` | `dict[str, Any]` | Optional structured context (seed, adapter, inputs). |

`RunManifest.save()` writes the manifest atomically (temp file + replace) so the
event stream stays consistent even on abrupt termination.

### Example event sequence

Below is the serialized manifest fragment for a stage that failed once and then
recovered on retry:

```json
{
  "run_id": "teaser",
  "events": [
    {"run_id": "teaser", "stage": "tts", "status": "begin", "timestamp": 169..., "attempt": 1},
    {"run_id": "teaser", "stage": "tts", "status": "fail", "timestamp": 169..., "attempt": 1, "error": "Timeout contacting TTS service"},
    {"run_id": "teaser", "stage": "tts", "status": "begin", "timestamp": 169..., "attempt": 2},
    {"run_id": "teaser", "stage": "tts", "status": "success", "timestamp": 169..., "attempt": 2, "metadata": {"lines": 12}}
  ]
}
```

### Runner usage

- Resume logic: `Runner.run(..., resume=True)` consults both the manifest and
  per-stage checkpoints. If the last manifest event for a stage is `success`,
  the runner skips the stage even when the checkpoint is missing.
- Observability: since events are append-only, operators can inspect
  `manifest.json` to see exactly how many retries occurred and which error
  strings were surfaced.
- Extensibility: add new metadata keys freely, but do not remove core fields
  without updating `RunManifest` and this document.

4) Schema artifact catalog
--------------------------

Schema and policy artifacts are versioned once and consumed everywhere via
`configs/schema_artifacts.yaml`. The helper module
`sparkle_motion.schema_registry` reads that file and exposes utility methods:

- `movie_plan_schema()` / `asset_refs_schema()` / `stage_event_schema()` /
  `checkpoint_schema()` / `run_context_schema()` / `stage_manifest_schema()` — typed helpers returning the
  `SchemaArtifact` (with both `.uri` and `.local_path`) for the canonical
  artifacts.
- `resolve_schema_uri(name, prefer_local=None)` — return either the artifact
  URI or local fallback (with warnings when fixture mode forces a local path).
- `list_schema_names()` — enumerate MoviePlan/AssetRefs/StageEvent/Checkpoint/RunContext/StageManifest.

ScriptAgent prompts should call `schema_registry.movie_plan_schema().uri` when
populating the `json_schema` parameter in ADK PromptTemplates, while
WorkflowAgent tooling injects the same URIs into stage validators via the typed
helpers. This keeps every environment on the same schema versions without
copying JSON inline.

Use `sparkle_motion.prompt_templates.build_script_agent_prompt_template()` to
assemble the prompt metadata in-process, or run
`PYTHONPATH=src python scripts/render_script_agent_prompt.py` to emit a JSON
payload suitable for `adk llm-prompts push`. Both paths look up
`artifact://sparkle-motion/schemas/movie_plan/v1` via the registry so the
WorkflowAgent and ScriptAgent stay in sync regardless of runtime.

5) Manual QA + signoff placeholder
-----------------------------------

Automated QA is currently disabled, so the orchestrator publishes its final
artifacts immediately after the `assemble` stage. Operators must perform any
content, safety, or continuity review outside of the pipeline. Recommended
stopgaps until the `qa_qwen2vl` stack returns:

1. **Human review checkpoints.** Continue using the existing
   `runs/<run_id>/human_review/*.json` markers described later in this document.
   They pause the pipeline after major stages so editors can inspect artifacts
   before spending additional GPU time.
2. **Notebook/UI badges.** Notebook helpers should display a prominent "QA
   automation disabled" warning near final artifact previews so downstream users
   know deliverables have not been machine-validated.
3. **Run metadata.** Production runs should call
   `adk_helpers.write_memory_event(event_type="qa_automation", payload={"status": "disabled"})`
   once per run. This makes the gap obvious in `/status` output, logs, and any
   downstream dashboards that relied on QA verdicts.
4. **Manual checklists.** Teams that previously depended on `qa_publish`
   outcomes should track the equivalent checks in their own docs (e.g., finger
   counts, safety scans, audio continuity) so the gap is covered until automation
   returns.

The old QA policy files (`configs/qa_policy.yaml`, schema, packaging script) and
schema catalog entries have been removed; recreating them will be part of the
future reintroduction plan.

5) Error handling and retries
------------------------------

- Implement exponential backoff with bounded retries when interacting with
  remote/model services. The `run_manifest.retry` decorator is provided to
  record attempts in the manifest.
- On permanent failures (exhausted retries), the stage should leave a
  `failed` checkpoint and the runner stops the run; operators can inspect
  the manifest and artifacts for diagnostics.

6) Determinism and testing
--------------------------

- Stage implementations used in tests should accept and use a `seed` value
  (from `movie_plan.metadata` or environment) so smoke tests are deterministic.
- For CI-friendly, fast smoke tests, prefer lightweight deterministic stubs.

7) Contracts / Types
--------------------

- `MoviePlan` / `ShotSpec` / `AssetRefs` are defined in
  `src/sparkle_motion/schemas.py` and should be used by stage implementations
  for input validation and serialization.

8) Resume semantics (summary)
-----------------------------

- The runner loads `runs/<run_id>/manifest.json` (if present) and per-stage
  checkpoint files. When `resume=True`, a stage is skipped when either the
  checkpoint indicates `status == "success"` or the manifest's last status
  for the stage is `success`.

### Helper APIs for selective resume/retry

- `Runner.resume_from_stage(run_id=..., stage="images", movie_plan=None)` — reruns the
  specified stage and every subsequent stage while skipping earlier ones. When
  `movie_plan` is omitted the runner loads `runs/<run_id>/movie_plan.json`.
  Useful after manual edits to assets or when a stage failed mid-run.
- `Runner.retry_stage(run_id=..., stage="assemble", movie_plan=None)` — reruns only the
  provided stage (no other stages execute). This forces a new checkpoint even if
  the previous run succeeded, which is helpful after fixing final-assembly
  assets referenced by a single stage.
- Both helpers validate the stage name and internally delegate to
  `Runner.run()` with the appropriate `resume/start_stage/only_stage` values, so
  service wiring, manifest updates, and human-review hooks continue to work
  identically to a full run.

9) Example minimal stage
------------------------

```
def stage_images(movie_plan, asset_refs, run_dir):
    # generate or simulate frames, update asset_refs, write checkpoint
    asset_refs['shots']['s1']['start_frame'] = str(run_dir / 's1_start.png')
    checkpoint = Checkpoint(stage='images', status='success', timestamp=time.time())
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints' / 'images.json').write_text(checkpoint.json())
    return asset_refs
```

This document is a short contract; implementations should keep checkpoints
small and resilient to partial writes (the runner uses atomic manifest writes
and the checkpoint files are small JSON objects).

10) Human review checkpoints
-----------------------------

- **Script review**: if `runs/<run_id>/human_review/script.json` exists with `{"decision": "revise"}`, the runner halts after writing `movie_plan.json`, logs the decision via MemoryService, and waits for the reviewer to clear/update the file before resuming.
- **Images/clip review**: `human_review/images.json` blocks after the `images` stage so humans can inspect keyframes before paying for Wan or later stages. Decisions (`approve`, `revise`, optional notes) are mirrored into `memory_log.json` for auditability.
- **Usage pattern**: reviewers edit the JSON via Colab/Drive, the runner polls between stages, and ArtifactService ensures referenced assets are easy to find.

These checkpoints keep the prototype human-in-the-loop friendly while aligning with ADK’s expectation that Session/Artifact/Memory services provide a consistent audit trail.

For additional context, see:
- `src/sparkle_motion/orchestrator.py` for the reference runner implementation and fallback behavior.
- `src/sparkle_motion/schemas.py` for the canonical `MoviePlan`, `AssetRefs`, and related models.
- `docs/ORCHESTRATOR.md` (this file) for the authoritative stage contract—update it before altering stage order or checkpoint semantics.
