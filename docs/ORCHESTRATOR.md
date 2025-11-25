Per-stage contract — Sparkle Motion orchestrator
===============================================

This document summarizes the per-stage contract the orchestrator expects and
the checkpoint format used to support resume and retry semantics.

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
| `images` | Call SDXL (or fallback) to render keyframe PNGs. | `ShotSpec.start_frame_prompt`, `ShotSpec.end_frame_prompt`, optional `MoviePlan.metadata["seed"].` | Populates `start_frame`/`end_frame` paths per shot. | `{"frames_written": <int>, "adapter": "sdxl"}` and optionally `seed` used. |
| `videos` | Call Wan / video adapter to create motion clips. | Newly written keyframe paths, `ShotSpec.motion_prompt`, `ShotSpec.duration_sec`. | Updates `raw_clip` per shot. | `{"clips_written": <int>, "adapter": "wan"}`. |
| `tts` | Generate dialogue WAVs via TTS adapter. | `ShotSpec.dialogue` and `characters.voice_profile`. | Stores a list of WAV paths in `dialogue_audio`. | `{"lines_synthesized": <int>, "voice_model": "polyglot-v1"}`. |
| `lipsync` | Wav2Lip (or stub) to merge audio + raw video. | `raw_clip`, `dialogue_audio`. | Writes `final_video_clip` per shot. | `{"clips_synced": <int>, "adapter": "wav2lip"}`. |
| `assemble` | Concatenate final clips, add BGM, output movie. | `final_video_clip` entries, `MoviePlan.metadata` (e.g., frame rate). | May record top-level extras such as `asset_refs.extras["final_movie"] = str(path)`. | `{"final_path": "movie_final.mp4", "duration": 12.4, "video_codec": "mpeg4", "audio_codec": "aac"}`. |
| `qa` | Run automated QA, persist report, and evaluate policy gates. | `MoviePlan`, `AssetRefs`, QA policy. | Writes QA report to `run_dir/qa_report.json` and `qa_actions.json` (gating summary); `asset_refs` untouched. | `{"qa_report": "qa_report.json", "issues_found": <int>, "decision": "approve", "qa_policy_action": "approve"}`. |

Stages run in the order defined above. Each stage is responsible for:
1. Reading only the inputs listed in the table (other keys are considered implementation detail and should not be relied upon).
2. Writing artifacts to subdirectories of `run_dir` (never outside of the run sandbox).
3. Updating `asset_refs` atomically (mutate then persist `runs/<run_id>/asset_refs.json`).
4. Emitting a checkpoint JSON payload that captures the stage status and enough metadata for operators to reason about retries.

### Service wiring & tool catalog

- **SessionService** (`src/sparkle_motion/services.py`) issues the `run_id`, prepares `runs/<run_id>/` (artifacts, checkpoints, human-review folders), and hands the runner a `SessionContext`. Plugging in the official ADK service later is a matter of passing a different implementation into `Runner(session_service=...)`.
- **ArtifactService** records every persisted output in `runs/<run_id>/artifacts.json` so humans/agents have a canonical lookup for Drive paths or URIs. Stage adapters call `artifact_service.register(name, path)` after writing files.
- **MemoryService** appends structured events (stage begin/success/fail, QA verdicts, human-review notes) to `runs/<run_id>/memory_log.json`. This is the long-lived log ADK agents or dashboards can consume.
- **ToolRegistry** mirrors ADK’s FunctionTool catalog. Each stage registers its callable/metadata with the registry, and the runner resolves stages via registry lookups, which allows hot-swapping adapters (local vs. remote) without editing orchestrator logic.

These abstractions keep the runner faithful to ADK patterns while remaining lightweight for the current Colab workflow.

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
- **Reads**: `ShotSpec.dialogue` text and per-character `voice_profile`.
- **Writes**: WAV files per line under `run_dir/audio/<shot_id>/<line>.wav` and list stored in `dialogue_audio`.
- **Metadata**: `lines_synthesized`, `voice_model`, `characters_processed`.

##### `lipsync`
- **Reads**: `raw_clip` and `dialogue_audio` lists for each shot.
- **Writes**: muxed clips under `run_dir/lipsync/<shot_id>.mp4` updating `final_video_clip`.
- **Metadata**: `clips_synced`, `adapter`, `avg_rtf` (real-time factor).

##### `assemble`
- **Reads**: list of final per-shot clips plus global metadata (fps, soundtrack path).
- **Writes**: final MP4 (and optional thumbnails) stored in `asset_refs.extras`.
- **Metadata**: `final_path`, `duration_sec`, `video_codec`, `audio_codec`, `shots_used` (ordered list).

##### `qa`
- **Reads**: canonical QA policy, `MoviePlan`, `AssetRefs`.
- **Writes**: `qa_report.json` under `run_dir`, checkpoint with decision, optional HTML summary.
- **Metadata**: `qa_report`, `decision`, `issues_found`, `policy_version`.
- **Human review bridge**: after QA (or earlier stages), the runner consults `runs/<run_id>/human_review/*.json` (see “Human review checkpoints”) and records reviewer decisions in `memory_log.json` so manual approvals integrate cleanly with automation.

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

4) QA policy & thresholds
-------------------------

The QA stage consumes a declarative policy to determine whether a run is
approved, needs regeneration, or must be escalated. The canonical policy lives
at `configs/qa_policy.yaml` and is versioned with the repo. A JSON Schema export
(`configs/qa_policy.schema.json`) documents the structure and can be used for
validation in tooling.

### Policy structure

```yaml
# configs/qa_policy.yaml
version: 1
thresholds:
  prompt_match_min: 0.75          # cosine similarity between prompt and caption
  max_finger_issue_ratio: 0.10    # fraction of frames allowed to flag fingers
  allow_missing_audio: false
  rerun_on_artifact_notes: true
actions:
  approve_if:
    - prompt_match >= threshold(prompt_match_min)
    - finger_issues <= threshold(max_finger_issue_ratio)
    - artifact_notes == []
  regenerate_if:
    - prompt_match < threshold(prompt_match_min)
    - finger_issues > threshold(max_finger_issue_ratio)
  escalate_if:
    - missing_audio_detected == true
    - artifact_notes contains "policy_violation"
```

Key requirements:

- `thresholds` defines numeric/boolean knobs the QA adapter can reuse across
  shots.
- `actions` lists mutually exclusive predicates. Evaluation order is
  `approve_if`, then `regenerate_if`, then `escalate_if`; the first matching
  block determines the outcome.

### QA report payload

The QA adapter must emit JSON conforming to `QAReport` in
`src/sparkle_motion/schemas.py` (exported to `schemas/QAReport.schema.json`)
and save it at `runs/<run_id>/qa_report.json`.
Field overview:

| Field | Notes |
| --- | --- |
| `movie_title` | Optional copy of `MoviePlan.title` for completeness. |
| `per_shot` | List of entries: `shot_id`, numeric `prompt_match`, boolean `finger_issues`, `artifact_notes` (list of strings). Extra metadata is allowed if backward compatible. |
| `summary` | Free-form text summarizing QA outcome. |
| `decision` (optional extension) | `"approve" | "regenerate" | "escalate" | "pending"`; recommended for clarity. |
| `missing_audio_detected` | Shot-level flag used by gating predicates. |
| `safety_violation` | Shot-level flag used by gating predicates. |
| `finger_issue_ratio` | Optional ratio (0-1) reported per shot; runner also derives run-level ratios. |

Per-shot entries should also include any measurements referenced by the policy
(e.g., `missing_audio_detected`, `hands_detected`).

### Gating outcomes

| Decision | Trigger | Runner response |
| --- | --- | --- |
| `approve` | All `approve_if` predicates satisfied. | Orchestrator marks run complete; outputs remain on disk. |
| `regenerate` | Any `regenerate_if` predicate matches. | Runner may requeue specific stages (e.g., images/videos) when `auto_regenerate_on_qa_fail` / `--auto-qa-regenerate` is enabled, or leave a TODO for operators. |
| `escalate` | Any `escalate_if` predicate matches. | Runner halts and notifies humans; no automatic retries. |

### Integration contract

1. `qa_adapter.run_qa(movie_plan, asset_refs, run_dir)` evaluates every shot,
  and writes `qa_report.json` plus a checkpoint.
2. The checkpoint metadata should include `{"qa_report": "qa_report.json", "decision": "approve"}`.
3. After a successful QA stage the runner loads `configs/qa_policy.yaml`,
  validates it against `configs/qa_policy.schema.json`, validates the QA
  report against `schemas/QAReport.schema.json`, and evaluates policy
  predicates to compute an action (approve/regenerate/escalate).
4. The resulting gating plan is persisted to `qa_actions.json`, registered via
  `ArtifactService`, and mirrored into `memory_log.json` as
  `qa_gating_decision` events (plus `qa_regenerate_required` or
  `qa_escalated` when relevant).
5. Operators can either rely on the `--auto-qa-regenerate` CLI flag (or
  `auto_regenerate_on_qa_fail=True` when instantiating `Runner`) to rehearse the
  recommended stages automatically, or call
  `Runner.resume_from_stage(..., stage=<first regenerate stage>)` manually to
  apply the fix and review the recorded reasons.

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

- `MoviePlan` / `ShotSpec` / `AssetRefs` / `QAReport` are defined in
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
- `Runner.retry_stage(run_id=..., stage="qa", movie_plan=None)` — reruns only the
  provided stage (no other stages execute). This forces a new checkpoint even if
  the previous run succeeded, which is helpful after fixing QA policies or
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
- `src/sparkle_motion/schemas.py` for the canonical `MoviePlan`, `AssetRefs`, and QA models.
- `docs/ORCHESTRATOR.md` (this file) for the authoritative stage contract—update it before altering stage order or checkpoint semantics.
