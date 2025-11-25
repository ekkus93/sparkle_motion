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
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .adapters.common import MissingDependencyError
from .adapters import sdxl_adapter, wan_adapter, tts_adapter, wav2lip_adapter, assemble_adapter, qa_adapter
from . import qa_policy
from .run_manifest import RunManifest, retry as manifest_retry
from .services import ArtifactService, MemoryService, SessionContext, SessionService, ToolRegistry
from .human_review import HumanReviewCoordinator


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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA_POLICY_PATH = PROJECT_ROOT / "configs" / "qa_policy.yaml"
DEFAULT_QA_POLICY_SCHEMA_PATH = PROJECT_ROOT / "configs" / "qa_policy.schema.json"


class Runner:
    """Simple stage runner that plugs into lightweight ADK-style services."""

    def __init__(
        self,
        runs_root: str = "runs",
        *,
        session_service: Optional[SessionService] = None,
        tool_registry: Optional[ToolRegistry] = None,
        artifact_service_factory: Optional[Callable[[SessionContext], ArtifactService]] = None,
        memory_service_factory: Optional[Callable[[SessionContext], MemoryService]] = None,
        human_review_factory: Optional[Callable[[MemoryService], HumanReviewCoordinator]] = None,
        qa_policy_path: Optional[Path | str] = None,
        qa_policy_schema_path: Optional[Path | str] = None,
        auto_regenerate_on_qa_fail: bool = False,
    ) -> None:
        self.runs_root = Path(runs_root)
        self.session_service = session_service or SessionService(self.runs_root)
        self.tool_registry = tool_registry or ToolRegistry()
        self.artifact_service_factory = artifact_service_factory or (lambda session: ArtifactService(session))
        self.memory_service_factory = memory_service_factory or (lambda session: MemoryService(session))
        self.human_review_factory = human_review_factory or (lambda memory: HumanReviewCoordinator(memory))
        self.qa_policy_path = Path(qa_policy_path) if qa_policy_path else DEFAULT_QA_POLICY_PATH
        self.qa_policy_schema_path = (
            Path(qa_policy_schema_path) if qa_policy_schema_path else DEFAULT_QA_POLICY_SCHEMA_PATH
        )
        self.auto_regenerate_on_qa_fail = auto_regenerate_on_qa_fail
        self._auto_regen_active = False

        self._stages: List[tuple[str, Callable[[Dict[str, Any], Dict[str, Any], Path], Dict[str, Any]]]] = []
        self.stage_order: List[str] = []
        # default stage registration
        self.stages = [
            ("script", self.stage_script),
            ("images", self.stage_images),
            ("videos", self.stage_videos),
            ("tts", self.stage_tts),
            ("lipsync", self.stage_lipsync),
            ("assemble", self.stage_assemble),
            ("qa", self.stage_qa),
        ]

    @property
    def stages(self) -> List[tuple[str, Callable[[Dict[str, Any], Dict[str, Any], Path], Dict[str, Any]]]]:
        return list(self._stages)

    @stages.setter
    def stages(self, new_stages: List[tuple[str, Callable[[Dict[str, Any], Dict[str, Any], Path], Dict[str, Any]]]]) -> None:
        self._stages = list(new_stages)
        self.stage_order = [name for name, _ in self._stages]
        for stage_name, stage_fn in self._stages:
            self.tool_registry.register(
                name=stage_name,
                description=f"Stage tool: {stage_name}",
                func=stage_fn,
                config={"category": "stage"},
            )

    def _run_dir(self, run_id: str) -> Path:
        p = self.runs_root / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _checkpoint_path(self, run_dir: Path, stage: str) -> Path:
        cp_dir = run_dir / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        return cp_dir / f"{stage}.json"

    def _load_movie_plan(self, movie_plan_path: Path) -> Dict[str, Any]:
        if not movie_plan_path.exists():
            raise FileNotFoundError(
                f"Movie plan not found at {movie_plan_path}. Provide a movie_plan or run the full pipeline first."
            )
        return json.loads(movie_plan_path.read_text(encoding="utf-8"))

    def _select_stages(self, *, start_stage: Optional[str], only_stage: Optional[str]) -> List[str]:
        if start_stage and only_stage:
            raise ValueError("start_stage and only_stage are mutually exclusive")
        if only_stage:
            if only_stage not in self.stage_order:
                raise ValueError(f"Stage '{only_stage}' is not registered")
            return [only_stage]
        if start_stage:
            if start_stage not in self.stage_order:
                raise ValueError(f"Stage '{start_stage}' is not registered")
            idx = self.stage_order.index(start_stage)
            return self.stage_order[idx:]
        return self.stage_order

    def _assert_stage_registered(self, stage: str) -> None:
        if stage not in self.stage_order:
            raise ValueError(f"Stage '{stage}' is not registered")

    def _atomic_write_json(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(obj, indent=2, ensure_ascii=False)
        dir_path = str(path.parent)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=dir_path)
        tmp_file = None
        try:
            tmp_file = os.fdopen(fd, "w", encoding="utf-8")
            tmp_file.write(data)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_file.close()
            tmp_file = None
            os.replace(tmp_path, str(path))
            tmp_path = None
        finally:
            if tmp_file is not None:
                try:
                    tmp_file.close()
                except Exception:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _normalize_stage_result(self, result: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        asset_refs = result
        if isinstance(result, tuple) and len(result) == 2:
            asset_refs = result[0]
            maybe_meta = result[1]
            if isinstance(maybe_meta, dict):
                metadata = maybe_meta
        return asset_refs, metadata

    def _finalize_stage_metadata(
        self,
        *,
        stage_name: str,
        stage_fn: Callable[[Dict[str, Any], Dict[str, Any], Path], Dict[str, Any]],
        asset_refs: Dict[str, Any],
        run_dir: Path,
        duration_sec: float,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "adapter": getattr(stage_fn, "__module__", stage_name).split(".")[-1],
            "duration_sec": round(max(duration_sec, 0.0), 4),
            "shots_total": len(asset_refs.get("shots", {})),
        }

        extras = extra_metadata or {}
        for key, value in extras.items():
            if value is None:
                continue
            metadata[key] = value

        if stage_name == "qa":
            metadata.setdefault("qa_report", str(run_dir / "qa_report.json"))
        if stage_name == "assemble":
            final_movie = asset_refs.get("extras", {}).get("final_movie") if isinstance(asset_refs, dict) else None
            if final_movie:
                metadata.setdefault("final_movie", final_movie)

        return metadata

    def _checkpoint_payload(
        self,
        *,
        stage: str,
        status: str,
        attempt: Optional[int],
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        attempt_val = attempt if isinstance(attempt, int) and attempt > 0 else 1
        payload = {
            "stage": stage,
            "status": status,
            "timestamp": time.time(),
            "attempt": attempt_val,
            "error": error,
            "metadata": metadata or {},
        }
        return payload

    def _earliest_regeneration_stage(self, candidate_stages: List[str]) -> Optional[str]:
        if not candidate_stages:
            return None
        order_index = {name: idx for idx, name in enumerate(self.stage_order)}
        best: Optional[str] = None
        best_idx: Optional[int] = None
        for stage in candidate_stages:
            idx = order_index.get(stage)
            if idx is None:
                continue
            if best_idx is None or idx < best_idx:
                best = stage
                best_idx = idx
        return best

    def _auto_regenerate_from_stage(
        self,
        *,
        movie_plan: Dict[str, Any],
        run_id: str,
        start_stage: str,
    ) -> Dict[str, Any]:
        """Re-run the pipeline starting from ``start_stage`` once for QA auto-regenerate."""

        self._auto_regen_active = True
        try:
            return self.run(
                movie_plan=movie_plan,
                run_id=run_id,
                resume=True,
                start_stage=start_stage,
                only_stage=None,
            )
        finally:
            self._auto_regen_active = False

    def _apply_qa_policy(
        self,
        *,
        run_dir: Path,
        artifact_service: ArtifactService,
        memory_service: MemoryService,
        metadata: Dict[str, Any],
    ) -> Optional[qa_policy.QAGatingDecision]:
        qa_report_path = metadata.get("qa_report")
        if qa_report_path:
            qa_report_path = Path(qa_report_path)
        else:
            qa_report_path = run_dir / "qa_report.json"

        if not qa_report_path.exists():
            memory_service.record_event(
                "qa_policy_skipped",
                {"reason": f"QA report missing at {qa_report_path}"},
            )
            return None

        policy_path = self.qa_policy_path
        if not policy_path or not policy_path.exists():
            memory_service.record_event(
                "qa_policy_skipped",
                {"reason": f"QA policy missing at {policy_path}"},
            )
            return None

        schema_path = self.qa_policy_schema_path if self.qa_policy_schema_path.exists() else None

        try:
            decision = qa_policy.evaluate_report(
                report_path=qa_report_path,
                policy_path=policy_path,
                policy_schema_path=schema_path,
            )
        except qa_policy.QAPolicyError as exc:
            memory_service.record_event(
                "qa_policy_error",
                {"error": str(exc)},
            )
            return None

        metadata["qa_policy_action"] = decision.action
        memory_service.record_event(
            "qa_gating_decision",
            {
                "action": decision.action,
                "reasons": decision.reasons,
                "regenerate_stages": decision.regenerate_stages,
                "metrics": decision.metrics,
            },
        )
        actions_path = run_dir / "qa_actions.json"
        self._atomic_write_json(
            actions_path,
            {
                "decision": decision.action,
                "reasons": decision.reasons,
                "regenerate_stages": decision.regenerate_stages,
                "metrics": decision.metrics,
                "policy_path": str(policy_path),
                "schema_path": str(schema_path) if schema_path else None,
                "qa_report": str(qa_report_path),
            },
        )
        artifact_service.register(
            name="qa_actions",
            path=actions_path,
            metadata={"decision": decision.action},
        )
        if decision.action == "regenerate":
            memory_service.record_event(
                "qa_regenerate_required",
                {"stages": decision.regenerate_stages, "reasons": decision.reasons},
            )
        elif decision.action == "escalate":
            memory_service.record_event(
                "qa_escalated",
                {"reasons": decision.reasons},
            )

        return decision

    def run(
        self,
        movie_plan: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        resume: bool = False,
        *,
        start_stage: Optional[str] = None,
        only_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline for the given movie_plan."""

        session_ctx = self.session_service.create_session(run_id)
        run_id = session_ctx.run_id
        run_dir = session_ctx.run_dir
        artifact_service = self.artifact_service_factory(session_ctx)
        memory_service = self.memory_service_factory(session_ctx)
        human_review = self.human_review_factory(memory_service) if self.human_review_factory else None

        movie_plan_path = run_dir / "movie_plan.json"
        asset_refs_path = run_dir / "asset_refs.json"

        if movie_plan is None:
            movie_plan = self._load_movie_plan(movie_plan_path)
        else:
            self._atomic_write_json(movie_plan_path, movie_plan)
        artifact_service.register(name="movie_plan", path=movie_plan_path)

        # load existing asset_refs if present
        if asset_refs_path.exists():
            asset_refs = json.loads(asset_refs_path.read_text(encoding="utf-8"))
        else:
            asset_refs = {"shots": {}}
            self._atomic_write_json(asset_refs_path, asset_refs)
        artifact_service.register(name="asset_refs", path=asset_refs_path, metadata={"stage": "init"})

        # create or load run manifest and save it at runs/<run_id>/manifest.json
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = RunManifest.load(manifest_path)
            except Exception:
                print("[runner] warning: failed to load existing manifest; creating a new one")
                manifest = RunManifest(run_id=run_id, path=manifest_path)
        else:
            manifest = RunManifest(run_id=run_id, path=manifest_path)

        selected_stages = self._select_stages(start_stage=start_stage, only_stage=only_stage)
        force_rerun_stages = set(selected_stages) if (start_stage or only_stage) else set()

        for stage_name in selected_stages:
            try:
                stage_spec = self.tool_registry.get(stage_name)
            except KeyError as exc:
                raise RuntimeError(f"Stage '{stage_name}' is not registered in the tool catalog") from exc
            stage_fn = stage_spec.func
            cp_path = self._checkpoint_path(run_dir, stage_name)

            # If resume requested, consult both checkpoint file and manifest events
            if resume and stage_name not in force_rerun_stages:
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
                stage_start_time = time.time()
                stage_result = wrapped(movie_plan, asset_refs, run_dir, manifest=manifest)
                stage_end_time = time.time()
                asset_refs, stage_metadata = self._normalize_stage_result(stage_result)
                self._atomic_write_json(asset_refs_path, asset_refs)
                artifact_service.register(name="asset_refs", path=asset_refs_path, metadata={"stage": stage_name})
                attempts_used = manifest.last_attempt_for_stage(stage_name) if manifest else 1
                metadata = self._finalize_stage_metadata(
                    stage_name=stage_name,
                    stage_fn=stage_fn,
                    asset_refs=asset_refs,
                    run_dir=run_dir,
                    duration_sec=stage_end_time - stage_start_time,
                    extra_metadata=stage_metadata,
                )
                cp_obj = self._checkpoint_payload(
                    stage=stage_name,
                    status="success",
                    attempt=attempts_used,
                    metadata=metadata,
                )
                self._atomic_write_json(cp_path, cp_obj)
                memory_service.record_event(
                    "stage_success",
                    {
                        "stage": stage_name,
                        "attempt": attempts_used,
                        "metadata": metadata,
                    },
                )
                # persist manifest
                try:
                    manifest.save()
                except Exception:
                    print(f"[runner] warning: failed to save manifest for {run_id}")

                # route QA decisions + artifacts
                if stage_name == "qa":
                    qa_report = run_dir / "qa_report.json"
                    if qa_report.exists():
                        artifact_service.register(name="qa_report", path=qa_report, metadata=metadata)
                    decision = metadata.get("decision") or metadata.get("qa_decision")
                    if decision:
                        memory_service.record_event(
                            "qa_decision",
                            {"decision": decision, "issues_found": metadata.get("issues_found")},
                        )
                    qa_policy_decision = self._apply_qa_policy(
                        run_dir=run_dir,
                        artifact_service=artifact_service,
                        memory_service=memory_service,
                        metadata=metadata,
                    )
                    if (
                        self.auto_regenerate_on_qa_fail
                        and qa_policy_decision
                        and qa_policy_decision.action == "regenerate"
                    ):
                        if self._auto_regen_active:
                            memory_service.record_event(
                                "qa_auto_regenerate_skipped",
                                {
                                    "reason": "auto_regen_already_running",
                                    "requested_stages": qa_policy_decision.regenerate_stages,
                                },
                            )
                        else:
                            target_stage = self._earliest_regeneration_stage(qa_policy_decision.regenerate_stages)
                            if target_stage:
                                memory_service.record_event(
                                    "qa_auto_regenerate_triggered",
                                    {
                                        "target_stage": target_stage,
                                        "requested_stages": qa_policy_decision.regenerate_stages,
                                        "reasons": qa_policy_decision.reasons,
                                    },
                                )
                                return self._auto_regenerate_from_stage(
                                    movie_plan=movie_plan,
                                    run_id=run_id,
                                    start_stage=target_stage,
                                )
                            else:
                                memory_service.record_event(
                                    "qa_auto_regenerate_skipped",
                                    {
                                        "reason": "no_valid_stage",
                                        "requested_stages": qa_policy_decision.regenerate_stages,
                                    },
                                )

                # handle human review gates
                if human_review:
                    if stage_name == "script":
                        decision = human_review.require_script_signoff()
                        if decision.status == "revise":
                            memory_service.record_event(
                                "human_review_blocked",
                                {"stage": "script", "notes": decision.notes},
                            )
                            break
                        if decision.status == "approved":
                            memory_service.record_event(
                                "human_review_approved",
                                {"stage": "script", "notes": decision.notes},
                            )
                    if stage_name == "images":
                        decision = human_review.require_images_signoff()
                        if decision.status == "revise":
                            memory_service.record_event(
                                "human_review_blocked",
                                {"stage": "images", "notes": decision.notes},
                            )
                            break
                        if decision.status == "approved":
                            memory_service.record_event(
                                "human_review_approved",
                                {"stage": "images", "notes": decision.notes},
                            )
            except Exception as e:
                print(f"[runner] stage {stage_name} failed permanently: {e!r}")
                stage_end_time = time.time()
                attempts_used = manifest.last_attempt_for_stage(stage_name) if manifest else 1
                metadata = self._finalize_stage_metadata(
                    stage_name=stage_name,
                    stage_fn=stage_fn,
                    asset_refs=asset_refs,
                    run_dir=run_dir,
                    duration_sec=stage_end_time - stage_start_time,
                    extra_metadata={},
                )
                cp_obj = self._checkpoint_payload(
                    stage=stage_name,
                    status="failed",
                    attempt=attempts_used,
                    metadata=metadata,
                    error=repr(e),
                )
                self._atomic_write_json(cp_path, cp_obj)
                memory_service.record_event(
                    "stage_failure",
                    {
                        "stage": stage_name,
                        "attempt": attempts_used,
                        "error": repr(e),
                    },
                )
                try:
                    manifest.save()
                except Exception:
                    print(f"[runner] warning: failed to save manifest for failed stage {run_id}")
                # stop the run on failure
                break

        return asset_refs

    def resume_from_stage(
        self,
        *,
        run_id: str,
        stage: str,
        movie_plan: Optional[Dict[str, Any]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Resume a run starting at ``stage`` (rerunning that stage and everything after)."""

        self._assert_stage_registered(stage)
        return self.run(
            movie_plan=movie_plan,
            run_id=run_id,
            resume=resume,
            start_stage=stage,
            only_stage=None,
        )

    def retry_stage(
        self,
        *,
        run_id: str,
        stage: str,
        movie_plan: Optional[Dict[str, Any]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Retry a single stage (rerun only ``stage`` without touching others)."""

        self._assert_stage_registered(stage)
        return self.run(
            movie_plan=movie_plan,
            run_id=run_id,
            resume=resume,
            start_stage=None,
            only_stage=stage,
        )

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
            self._atomic_write_json(qa_path, qa_result)
            metadata = {
                "qa_report": str(qa_path),
                "decision": qa_result.get("decision"),
                "issues_found": qa_result.get("issues_found"),
            }
            return asset_refs, metadata
        except MissingDependencyError as e:
            print(f"[adapter] QA adapter unavailable: {e}")
            qa = {"movie_title": movie_plan.get("title"), "per_shot": []}
            for sid in asset_refs.get("shots", {}).keys():
                qa["per_shot"].append(
                    {
                        "shot_id": sid,
                        "prompt_match": 0.0,
                        "finger_issues": False,
                        "artifact_notes": [],
                        "missing_audio_detected": False,
                        "safety_violation": False,
                    }
                )
            qa_path = run_dir / "qa_report.json"
            self._atomic_write_json(qa_path, qa)
            return asset_refs, {"qa_report": str(qa_path), "decision": "pending"}


if __name__ == "__main__":
    # quick CLI for manual testing
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal Sparkle Motion orchestrator run")
    parser.add_argument("--run-id", help="run id (directory name)", default=None)
    parser.add_argument("--runs-root", help="root runs dir", default="runs")
    parser.add_argument("--resume", help="resume existing run if checkpoints exist", action="store_true")
    parser.add_argument(
        "--auto-qa-regenerate",
        help="automatically resume from the first QA-recommended stage when QA policy requests regeneration",
        action="store_true",
    )
    args = parser.parse_args()

    # example minimal movie plan
    example = {
        "title": "Example Run",
        "shots": [{"id": "shot_001", "duration_sec": 4.0, "visual_description": "Test"}],
    }
    runner = Runner(runs_root=args.runs_root, auto_regenerate_on_qa_fail=args.auto_qa_regenerate)
    out = runner.run(movie_plan=example, run_id=args.run_id, resume=args.resume)
    print("Final asset_refs:", out)
