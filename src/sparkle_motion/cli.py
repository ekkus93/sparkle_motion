from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _tool_module_from_tool_id(tool_id: str) -> str:
    # map e.g. 'images_sdxl:local-colab' -> 'sparkle_motion.function_tools.images_sdxl.entrypoint'
    name = tool_id.split(":", 1)[0]
    return f"sparkle_motion.function_tools.{name}.entrypoint"


def validate_workflow_dryrun(path: Path) -> bool:
    from scripts import register_workflow_local as validator

    wf = load_yaml(path)
    try:
        return bool(validator.validate_workflow(wf))
    except TypeError:
        # older/newer signatures may accept more args
        return bool(validator.validate_workflow(wf, tool_registry=None, schema_artifacts=None))


def run_workflow(path: Path, outdir: Path, *, dry_run: bool = True) -> int:
    wf = load_yaml(path)
    if dry_run:
        ok = validate_workflow_dryrun(path)
        print("Dry-run OK" if ok else "Dry-run FAILED")
        return 0 if ok else 1

    # run in-process: sequential simple runner for local development
    os.environ.setdefault("ADK_USE_FIXTURE", "1")
    os.environ.setdefault("DETERMINISTIC", "1")

    outdir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"workflow": str(path), "stages": []}
    stage_outputs: Dict[str, Dict[str, Any]] = {}

    for idx, stage in enumerate(wf.get("stages", [])):
        sid = stage.get("id")
        sid_slug = _safe_stage_id(sid, idx)
        tool_id = stage.get("tool_id")
        print(f"Running stage: {sid_slug} -> {tool_id}")
        mod_path = _tool_module_from_tool_id(tool_id)
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except Exception as e:
            print(f"Failed to import tool module {mod_path}: {e}")
            return 2

        app = mod.make_app() if hasattr(mod, "make_app") else getattr(mod, "app")

        # use TestClient to exercise invoke endpoint in-process
        try:
            from fastapi.testclient import TestClient

            client = TestClient(app)
            payload = _build_stage_payload(stage, stage_outputs, sid_slug)
            r = client.post("/invoke", json=payload)
            if r.status_code != 200:
                print(f"Stage {sid_slug} failed: {r.status_code} {r.text}")
                return 3
            data = r.json()
            response_path = _persist_stage_response(outdir, idx, sid_slug, data)
            artifact_uri = data.get("artifact_uri")
            artifact_payload = _load_artifact_payload(artifact_uri)
            artifact_path: Optional[Path] = None
            if artifact_payload is not None:
                artifact_path = _persist_artifact_payload(outdir, idx, sid_slug, artifact_payload)

            entry: Dict[str, Any] = {
                "id": sid or sid_slug,
                "artifact_uri": artifact_uri,
                "response_path": str(response_path),
            }
            if artifact_path is not None:
                entry["artifact_payload_path"] = str(artifact_path)
            manifest["stages"].append(entry)
            stage_outputs[sid or sid_slug] = {
                "response": data,
                "artifact_uri": artifact_uri,
                "artifact_payload": artifact_payload,
            }
            print(f"Stage {sid_slug} completed, artifact={artifact_uri}")
        except Exception as e:
            print(f"Stage {sid} runtime error: {e}")
            return 4

        time.sleep(0.01)

    # write manifest
    mf = outdir / "run_manifest.json"
    with mf.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Workflow run completed. Manifest: {mf}")
    return 0


def _persist_stage_response(outdir: Path, idx: int, stage_id: str, payload: Dict[str, Any]) -> Path:
    path = outdir / f"{idx:02d}_{stage_id}_response.json"
    _write_json(path, payload)
    return path


def _persist_artifact_payload(outdir: Path, idx: int, stage_id: str, payload: Any) -> Path:
    path = outdir / f"{idx:02d}_{stage_id}_artifact.json"
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_artifact_payload(uri: Optional[str]) -> Optional[Any]:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        path = Path(parsed.path)
    else:
        path = Path(uri)
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_stage_id(stage_id: Optional[str], idx: int) -> str:
    if stage_id:
        return stage_id
    return f"stage-{idx}"


def _build_stage_payload(stage: Dict[str, Any], stage_outputs: Dict[str, Dict[str, Any]], sid_slug: str) -> Dict[str, Any]:
    tool_id = stage.get("tool_id") or ""
    tool_name = tool_id.split(":", 1)[0].lower() if tool_id else ""
    stage_name = (stage.get("id") or "").lower()
    if _is_production_stage(stage, tool_id):
        plan_payload = _extract_movie_plan(stage_outputs)
        if plan_payload is None:
            plan_payload = _fallback_movie_plan()
            print(f"Warning: No MoviePlan detected from previous stages; using fallback plan for stage {sid_slug}")
        mode = stage.get("mode") or "run"
        return {"plan": plan_payload, "mode": mode}
    if stage_name == "qa" or tool_name == "qa_qwen2vl":
        return _build_qa_stage_payload(sid_slug, stage_outputs)
    return {"prompt": f"operator-run:{sid_slug}"}


def _is_production_stage(stage: Dict[str, Any], tool_id: str) -> bool:
    stage_id = (stage.get("id") or "").lower()
    tool_name = tool_id.split(":", 1)[0].lower() if tool_id else ""
    return stage_id == "production" or tool_name == "production_agent"


def _build_qa_stage_payload(sid_slug: str, stage_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    frames = _extract_frames_for_qa(stage_outputs)
    if not frames:
        frames = [_default_qa_frame_bytes()]
    payload_frames = [
        {
            "id": f"{sid_slug}-frame-{idx:04d}",
            "data_b64": _encode_b64(data),
        }
        for idx, data in enumerate(frames)
    ]
    return {"prompt": f"operator-run:{sid_slug}", "frames": payload_frames}


def _extract_frames_for_qa(stage_outputs: Dict[str, Dict[str, Any]]) -> List[bytes]:
    production = stage_outputs.get("production") or {}
    payload = production.get("artifact_payload") or {}
    frames: List[bytes] = []
    if isinstance(payload, dict):
        # Heuristic: look for embedded base64 preview fields or raw data bytes
        candidates = payload.get("shots") or payload.get("frames")
        if isinstance(candidates, list):
            for shot in candidates:
                if not isinstance(shot, dict):
                    continue
                data_b64 = shot.get("preview_b64") or shot.get("data_b64")
                if isinstance(data_b64, str):
                    try:
                        frames.append(base64.b64decode(data_b64))
                        continue
                    except Exception:
                        pass
                raw = shot.get("raw_bytes")
                if isinstance(raw, str):
                    frames.append(raw.encode("utf-8"))
    return frames


def _default_qa_frame_bytes() -> bytes:
    return b"sparkle-motion-qa-fixture-frame"


def _encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _extract_movie_plan(stage_outputs: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    priority_ids = ("script",)
    for pid in priority_ids:
        plan = _coerce_plan(stage_outputs.get(pid))
        if plan is not None:
            return plan
    for entry in stage_outputs.values():
        plan = _coerce_plan(entry)
        if plan is not None:
            return plan
    return None


def _coerce_plan(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entry:
        return None
    artifact_payload = entry.get("artifact_payload")
    plan = _plan_from_payload(artifact_payload)
    if plan is not None:
        return plan
    response_payload = entry.get("response")
    plan = _plan_from_payload(response_payload)
    if plan is not None:
        return plan
    return None


def _plan_from_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    if _has_required_shot_fields(payload):
        return payload
    for key in ("validated_plan", "plan", "movie_plan", "payload"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and _has_required_shot_fields(candidate):
            return candidate
    return None


def _has_required_shot_fields(plan: Dict[str, Any]) -> bool:
    shots = plan.get("shots")
    if not isinstance(shots, list) or not shots:
        return False
    first = next((shot for shot in shots if isinstance(shot, dict)), None)
    if first is None:
        return False
    required = {"duration_sec", "visual_description", "start_frame_prompt", "end_frame_prompt"}
    return required.issubset(first.keys())


def _fallback_movie_plan() -> Dict[str, Any]:
    candidate = Path("artifacts/Test_Movie.json")
    if candidate.exists():
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            plan = _plan_from_payload(payload)
            if plan is not None:
                return plan
        except Exception:
            pass
    # Synthetic plan keeps local workflow unblocked when upstream stages fail to produce one.
    return {
        "title": "CLI fallback plan",
        "metadata": {"source": "cli_fallback", "reason": "missing_plan"},
        "characters": [
            {
                "id": "char_narrator",
                "name": "Narrator",
                "description": "Explains the scene transitions",
                "voice_profile": {},
            },
            {
                "id": "char_hero",
                "name": "Hero",
                "description": "Adventurer exploring luminous canyon",
                "voice_profile": {},
            },
        ],
        "shots": [
            {
                "id": "fallback-shot-1",
                "duration_sec": 4,
                "setting": "Sunrise plateau",
                "visual_description": "Wide sunrise shot over a glowing canyon with floating dust motes.",
                "start_frame_prompt": "Golden hour light over a sandstone plateau, cinematic lighting, 35mm, volumetric rays",
                "end_frame_prompt": "Camera settles on glowing canyon rim with dust particles sparkling in the air",
                "motion_prompt": "Slow dolly forward from wide to medium",
                "is_talking_closeup": False,
                "dialogue": [
                    {"character_id": "char_narrator", "text": "A new journey begins at first light."}
                ],
            },
            {
                "id": "fallback-shot-2",
                "duration_sec": 5,
                "setting": "Crystal cavern",
                "visual_description": "Medium shot of the hero touching bioluminescent crystals inside a cavern.",
                "start_frame_prompt": "Hero steps into a teal-lit cavern, crystals casting caustic reflections",
                "end_frame_prompt": "Closeup of the hero's hand leaving streaks of light across the wall",
                "motion_prompt": "Handheld push-in with gentle camera shake",
                "is_talking_closeup": True,
                "dialogue": [
                    {"character_id": "char_hero", "text": "I can hear the canyon breathe."}
                ],
            },
        ],
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("file", help="Path to workflow YAML")
    p.add_argument("--out", default="./out", help="Output directory for artifacts/manifest")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    raise SystemExit(run_workflow(Path(args.file), Path(args.out), dry_run=args.dry_run))
