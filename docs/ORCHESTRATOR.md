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
| `images` | Call SDXL (or fallback) to render keyframe PNGs. | `ShotSpec.start_frame_prompt`, `ShotSpec.end_frame_prompt`, optional `MoviePlan.metadata["seed"]`. | Populates `start_frame`/`end_frame` paths per shot. | `{"frames_written": <int>, "adapter": "sdxl"}` and optionally `seed` used. |
| `videos` | Call Wan / video adapter to create motion clips. | Newly written keyframe paths, `ShotSpec.motion_prompt`, `duration_sec`. | Updates `raw_clip` per shot. | `{"clips_written": <int>, "adapter": "wan"}`. |
| `tts` | Generate dialogue WAVs via TTS adapter. | `ShotSpec.dialogue` and `characters.voice_profile`. | Stores a list of WAV paths in `dialogue_audio`. | `{"lines_synthesized": <int>}` and `voice_model`. |
| `lipsync` | Wav2Lip (or stub) to merge audio + raw video. | `raw_clip`, `dialogue_audio`. | Writes `final_video_clip` per shot. | `{"clips_synced": <int>}` plus adapter identifier. |
| `assemble` | Concatenate final clips, add BGM, output movie. | `final_video_clip` entries, `MoviePlan.metadata` (e.g., frame rate). | May record top-level extras such as `asset_refs.extras["final_movie"] = str(path)`. | `{"final_path": "movie_final.mp4", "duration": 12.4, "video_codec": "mpeg4", "audio_codec": "aac"}`. |
| `qa` | Run automated QA (policy TBD) and persist report. | `MoviePlan`, `AssetRefs`, QA policy. | Writes QA report to `run_dir/qa_report.json`; `asset_refs` untouched. | `{"qa_report": "qa_report.json", "issues_found": <int>}`. |

Stages run in the order defined above. Each stage is responsible for:
1. Reading only the inputs listed in the table (other keys are considered implementation detail and should not be relied upon).
2. Writing artifacts to subdirectories of `run_dir` (never outside of the run sandbox).
3. Updating `asset_refs` atomically (mutate then persist `runs/<run_id>/asset_refs.json`).
4. Emitting a checkpoint JSON payload that captures the stage status and enough metadata for operators to reason about retries.

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
| `error` | `str  optional` | ❌ | Present only when `status == "failed"`. |
| `metadata` | `dict[str, Any]` | ✅ (can be empty) | Stage-specific summary: counts, codec names, adapter id, etc. |

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
`src/sparkle_motion/schemas.py` and save it at `runs/<run_id>/qa_report.json`.
Field overview:

| Field | Notes |
| --- | --- |
| `movie_title` | Optional copy of `MoviePlan.title` for completeness. |
| `per_shot` | List of entries: `shot_id`, numeric `prompt_match`, boolean `finger_issues`, `artifact_notes` (list of strings). Extra metadata is allowed if backward compatible. |
| `summary` | Free-form text summarizing QA outcome. |
| `decision` (optional extension) | `"approve" | "regenerate" | "escalate"`; recommended for clarity. |

Per-shot entries should also include any measurements referenced by the policy
(e.g., `missing_audio_detected`, `hands_detected`).

### Gating outcomes

| Decision | Trigger | Runner response |
| --- | --- | --- |
| `approve` | All `approve_if` predicates satisfied. | Orchestrator marks run complete; outputs remain on disk. |
| `regenerate` | Any `regenerate_if` predicate matches. | Runner may requeue specific stages (e.g., images/videos) or leave a TODO for operators. |
| `escalate` | Any `escalate_if` predicate matches. | Runner halts and notifies humans; no automatic retries. |

### Integration contract

1. `qa_adapter.run_qa(movie_plan, asset_refs, run_dir)` loads the policy,
   evaluates every shot, and writes `qa_report.json` plus a checkpoint.
2. The checkpoint metadata should include `{"qa_report": "qa_report.json", "decision": "approve"}`.
3. The runner reads the QA decision to decide whether to rerun stages or exit.
4. Future dashboards can read both the policy and report to render per-shot
   findings.

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

For additional context, see:
- `src/sparkle_motion/orchestrator.py` for the reference runner implementation and fallback behavior.
- `src/sparkle_motion/schemas.py` for the canonical `MoviePlan`, `AssetRefs`, and QA models.
- `docs/ORCHESTRATOR.md` (this file) for the authoritative stage contract—update it before altering stage order or checkpoint semantics.
