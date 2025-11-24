"""Minimal orchestrator prototype: stage runner with checkpointing and retries.

This module is intentionally dependency-light and operates on plain JSON/dict shapes.
It provides a Runner class that executes configured stages in order and writes
checkpoints to disk so runs can be resumed.

Usage (programmatic):
    from sparkle_motion.orchestrator import Runner
    runner = Runner(runs_root="./runs")
    runner.run(movie_plan={...}, run_id="test-run", resume=False)

The stage implementations here are placeholders that simulate work and write
dummy asset paths into the asset_refs structure. Replace these with real
implementations that call SDXL/Wan/TTS/Wav2Lip later.
"""
from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .adapters.common import MissingDependencyError
from .adapters import sdxl_adapter, wan_adapter, tts_adapter, wav2lip_adapter, assemble_adapter, qa_adapter
from .run_manifest import RunManifest, retry as manifest_retry


def retry_with_backoff(attempts: int = 3, base_delay: float = 0.5, factor: float = 2.0, jitter: float = 0.2):
    """Decorator that retries the wrapped function with exponential backoff + jitter.

    The wrapped function should raise exceptions on failure.
    """

    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == attempts:
                        raise
                    # sleep with jitter
                    sleep_t = delay + random.uniform(0, jitter)
                    print(f"[retry] attempt {attempt} failed: {e!r}; sleeping {sleep_t:.2f}s")
                    time.sleep(sleep_t)
                    delay *= factor

        return wrapper

    return decorator


class Runner:
    """Simple stage runner.

    Stages are a list of tuples (stage_name, stage_callable).
    Each stage_callable receives (movie_plan: dict, asset_refs: dict, run_dir: Path)
    and must return an updated asset_refs dict.
    """

    def __init__(self, runs_root: str = "runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

        # default stages order (placeholders)
        self.stages: List[tuple[str, Callable[[Dict[str, Any], Dict[str, Any], Path], Dict[str, Any]]]] = [
            ("script", self.stage_script),
            ("images", self.stage_images),
            ("videos", self.stage_videos),
            ("tts", self.stage_tts),
            ("lipsync", self.stage_lipsync),
            ("assemble", self.stage_assemble),
            ("qa", self.stage_qa),
        ]

    def _run_dir(self, run_id: str) -> Path:
        p = self.runs_root / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _checkpoint_path(self, run_dir: Path, stage: str) -> Path:
        cp_dir = run_dir / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        return cp_dir / f"{stage}.json"

    def _write_json(self, path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def run(self, movie_plan: Dict[str, Any], run_id: Optional[str] = None, resume: bool = False) -> Dict[str, Any]:
        """Run the pipeline for the given movie_plan.

        Args:
            movie_plan: plain dict representing the plan.
            run_id: optional run id; if None, a timestamp-based id is created.
            resume: if True, skip stages that have successful checkpoints.

        Returns:
            final asset_refs dict (may be empty if nothing generated).
        """

        if run_id is None:
            run_id = time.strftime("run_%Y%m%d_%H%M%S")

        run_dir = self._run_dir(run_id)
        movie_plan_path = run_dir / "movie_plan.json"
        asset_refs_path = run_dir / "asset_refs.json"

        # persist the input plan
        self._write_json(movie_plan_path, movie_plan)

        # load existing asset_refs if present
        if asset_refs_path.exists():
            asset_refs = json.loads(asset_refs_path.read_text(encoding="utf-8"))
        else:
            asset_refs = {"shots": {}}

        # create or load run manifest and save it at runs/<run_id>/manifest.json
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = RunManifest.load(manifest_path)
            except Exception:
                # if manifest is corrupted, start a fresh one but keep path
                print("[runner] warning: failed to load existing manifest; creating a new one")
                manifest = RunManifest(run_id=run_id, path=manifest_path)
        else:
            manifest = RunManifest(run_id=run_id, path=manifest_path)

        for stage_name, stage_fn in self.stages:
            cp_path = self._checkpoint_path(run_dir, stage_name)

            # If resume requested, consult both checkpoint file and manifest events
            if resume:
                skipped = False
                if cp_path.exists():
                    try:
                        cp = json.loads(cp_path.read_text(encoding="utf-8"))
                        if cp.get("status") == "success":
                            print(f"[runner] skipping stage {stage_name} (checkpoint success)")
                            skipped = True
                    except Exception:
                        print(f"[runner] warning: failed to read checkpoint for {stage_name}; re-running")

                # consult manifest events for a successful completion as well
                try:
                    last_status = manifest.last_status_for_stage(stage_name)
                    if last_status == "success":
                        print(f"[runner] skipping stage {stage_name} (manifest indicates success)")
                        skipped = True
                except Exception:
                    # if manifest helper fails for any reason, ignore and proceed
                    pass

                if skipped:
                    continue

            print(f"[runner] running stage {stage_name}")

            # We need a small wrapper so the retry decorator can receive the manifest kwarg
            def _stage_wrapper(mp, ar, rd, *, manifest=None):
                # delegate to actual stage implementation; ignore manifest here
                return stage_fn(mp, ar, rd)

            # decorate wrapper with manifest-aware retry (records events)
            wrapped = manifest_retry(max_attempts=3, base_delay=0.5, jitter=0.2, stage_name=stage_name)(_stage_wrapper)

            # run the stage with retries and manifest recording
            try:
                asset_refs = wrapped(movie_plan, asset_refs, run_dir, manifest=manifest)
                # persist asset_refs and checkpoint
                self._write_json(asset_refs_path, asset_refs)
                cp_obj = {
                    "stage": stage_name,
                    "status": "success",
                    "timestamp": time.time(),
                }
                self._write_json(cp_path, cp_obj)
                # persist manifest
                try:
                    manifest.save()
                except Exception:
                    print(f"[runner] warning: failed to save manifest for {run_id}")
            except Exception as e:
                print(f"[runner] stage {stage_name} failed permanently: {e!r}")
                cp_obj = {
                    "stage": stage_name,
                    "status": "failed",
                    "timestamp": time.time(),
                    "error": repr(e),
                }
                self._write_json(cp_path, cp_obj)
                try:
                    manifest.save()
                except Exception:
                    print(f"[runner] warning: failed to save manifest for failed stage {run_id}")
                # stop the run on failure
                break

        return asset_refs

    # --- stage implementations (call adapters when available; fallback to simulation) ---
    def stage_script(self, movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Ensure shots exist in asset_refs."""
        shots = movie_plan.get("shots", [])
        for s in shots:
            sid = s.get("id")
            if sid not in asset_refs["shots"]:
                asset_refs["shots"][sid] = {
                    "start_frame": None,
                    "end_frame": None,
                    "raw_clip": None,
                    "dialogue_audio": [],
                    "final_video_clip": None,
                }
        return asset_refs

    def stage_images(self, movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Try to use SDXL adapter; on MissingDependencyError fall back to simulation."""
        try:
            return sdxl_adapter.generate_frames(movie_plan, asset_refs, run_dir)
        except MissingDependencyError as e:
            print(f"[adapter] SDXL adapter unavailable: {e}")
            # fallback: simulate images
            for sid, info in asset_refs.get("shots", {}).items():
                start = run_dir / f"{sid}_start.png"
                end = run_dir / f"{sid}_end.png"
                start.write_text("start-image", encoding="utf-8")
                end.write_text("end-image", encoding="utf-8")
                info["start_frame"] = str(start)
                info["end_frame"] = str(end)
            return asset_refs

    def stage_videos(self, movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Try Wan adapter; fallback to simulation."""
        try:
            return wan_adapter.generate_video(movie_plan, asset_refs, run_dir)
        except MissingDependencyError as e:
            print(f"[adapter] Wan adapter unavailable: {e}")
            for sid, info in asset_refs.get("shots", {}).items():
                raw = run_dir / f"{sid}_raw.mp4"
                raw.write_text("raw-clip", encoding="utf-8")
                info["raw_clip"] = str(raw)
            return asset_refs

    def stage_tts(self, movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Try TTS adapter; fallback to simulation."""
        try:
            return tts_adapter.generate_audio(movie_plan, asset_refs, run_dir)
        except MissingDependencyError as e:
            print(f"[adapter] TTS adapter unavailable: {e}")
            for s in movie_plan.get("shots", []):
                sid = s.get("id")
                dialogue = s.get("dialogue", [])
                if not dialogue:
                    continue
                wavs = []
                for i, line in enumerate(dialogue):
                    wav = run_dir / f"{sid}_line_{i}.wav"
                    wav.write_text(line.get("text", ""), encoding="utf-8")
                    wavs.append(str(wav))
                asset_refs["shots"][sid]["dialogue_audio"] = wavs
            return asset_refs

    def stage_lipsync(self, movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Try Wav2Lip adapter; fallback to simulation."""
        try:
            return wav2lip_adapter.lipsync(movie_plan, asset_refs, run_dir)
        except MissingDependencyError as e:
            print(f"[adapter] Wav2Lip adapter unavailable: {e}")
            for sid, info in asset_refs.get("shots", {}).items():
                final = run_dir / f"{sid}_final.mp4"
                final.write_text("final-video", encoding="utf-8")
                info["final_video_clip"] = str(final)
            return asset_refs

    def stage_assemble(self, movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Try assemble adapter; fallback to simulation."""
        try:
            return assemble_adapter.assemble(movie_plan, asset_refs, run_dir)
        except MissingDependencyError as e:
            print(f"[adapter] Assemble adapter unavailable: {e}")
            out = run_dir / "movie_final.mp4"
            out.write_text("movie-final", encoding="utf-8")
            return asset_refs

    def stage_qa(self, movie_plan: Dict[str, Any], asset_refs: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Try QA adapter; fallback to simulation."""
        try:
            qa_result = qa_adapter.run_qa(movie_plan, asset_refs, run_dir)
            qa_path = run_dir / "qa_report.json"
            self._write_json(qa_path, qa_result)
            return asset_refs
        except MissingDependencyError as e:
            print(f"[adapter] QA adapter unavailable: {e}")
            qa = {"movie_title": movie_plan.get("title"), "per_shot": []}
            for sid in asset_refs.get("shots", {}).keys():
                qa["per_shot"].append({"shot_id": sid, "prompt_match": "unknown", "finger_issues": False, "artifact_notes": []})
            qa_path = run_dir / "qa_report.json"
            self._write_json(qa_path, qa)
            return asset_refs


if __name__ == "__main__":
    # quick CLI for manual testing
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal Sparkle Motion orchestrator run")
    parser.add_argument("--run-id", help="run id (directory name)", default=None)
    parser.add_argument("--runs-root", help="root runs dir", default="runs")
    parser.add_argument("--resume", help="resume existing run if checkpoints exist", action="store_true")
    args = parser.parse_args()

    # example minimal movie plan
    example = {
        "title": "Example Run",
        "shots": [{"id": "shot_001", "duration_sec": 4.0, "visual_description": "Test"}],
    }
    runner = Runner(runs_root=args.runs_root)
    out = runner.run(movie_plan=example, run_id=args.run_id, resume=args.resume)
    print("Final asset_refs:", out)
