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

2) Checkpoint format
---------------------

The `Checkpoint` Pydantic model (see `src/sparkle_motion/schemas.py`) describes
the JSON checkpoint shape. Example:

```json
{
  "stage": "images",
  "status": "success",
  "timestamp": 169...,
  "attempt": 1,
  "error": null,
  "metadata": {}
}
```

Status values:
- `begin` — stage started (optional)
- `success` — stage completed successfully
- `failed` — stage failed and should be retried or investigated

3) Manifest events
------------------

The `RunManifest` (see `src/sparkle_motion/run_manifest.py`) records stage
events (`begin` / `fail` / `success`) with timestamps and attempt numbers. The
runner writes the manifest atomically and the manifest plus per-stage
checkpoints are used together to decide resume behavior.

4) Error handling and retries
-----------------------------

- Implement exponential backoff with bounded retries when interacting with
  remote/model services. The `run_manifest.retry` decorator is provided to
  record attempts in the manifest.
- On permanent failures (exhausted retries), the stage should leave a
  `failed` checkpoint and the runner stops the run; operators can inspect
  the manifest and artifacts for diagnostics.

5) Determinism and testing
--------------------------

- Stage implementations used in tests should accept and use a `seed` value
  (from `movie_plan.metadata` or environment) so smoke tests are deterministic.
- For CI-friendly, fast smoke tests, prefer lightweight deterministic stubs.

6) Contracts / Types
--------------------

- `MoviePlan` / `ShotSpec` / `AssetRefs` / `QAReport` are defined in
  `src/sparkle_motion/schemas.py` and should be used by stage implementations
  for input validation and serialization.

7) Resume semantics (summary)
----------------------------

- The runner loads `runs/<run_id>/manifest.json` (if present) and per-stage
  checkpoint files. When `resume=True`, a stage is skipped when either the
  checkpoint indicates `status == "success"` or the manifest's last status
  for the stage is `success`.

8) Example minimal stage
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
