# Notebook ↔ Agent Integration (working notes)

This document captures how users interact with the Sparkle Motion agents from a
Google Colab notebook. It is intentionally iterative; we will refine the flow as
the UI and orchestration harden.

> **Delivery status (2025-12-07):** Automated QA tooling (`qa_qwen2vl`) has been
> removed from the runtime. Production runs now terminate at the `finalize`
> stage, which publishes the `video_final` artifact and relies on manual review
> outside the control panel. The instructions below describe this
> finalize-first, QA-free state.

### Manual review flag (`QA_AUTOMATION_REMOVED`)

- QA automation remains disabled by default. Export `QA_AUTOMATION_REMOVED=1`
	in notebook + CLI environments to make the manual-review banner visible and
	to document that finalize artifacts require human sign-off. Leave the flag at
	`1` until the QA tool is stable again.
- When QA tooling returns, set `QA_AUTOMATION_REMOVED=0` before launching the
	control panel or CLI. Doing so removes the banner and lets us re-enable
	automated QA stages without editing notebooks.
- Regardless of the flag, operators must review the `video_final` artifact and
	emit `finalize_manual_review` events (see `docs/THE_PLAN.md`) before sharing
	deliverables. The control panel mirrors this requirement by inserting a
	manual review banner whenever `QA_AUTOMATION_REMOVED=1`.

## Colab-first workflow snapshot

1. User opens the shared Colab notebook (single-user profile).
2. Notebook cells start the two ADK agents (`script_agent`, `production_agent`)
	 plus the required FunctionTools (`images_sdxl`, `videos_wan`, `tts_chatterbox`,
	 etc.) inside the Colab VM.
6. The user provides a story prompt/title via notebook UI, which calls the
	 `script_agent` agent entrypoint to produce a `MoviePlan` artifact.
4. Once the plan exists, the notebook can trigger `production_agent` to execute
	 the pipeline (images → videos → TTS → lipsync → assemble) and surface
	 artifact URIs back to the user for manual review.

### Agent vs FunctionTool naming

- Only `script_agent` (plan generation) and `production_agent` (WorkflowAgent
	 orchestrator) retain the `_agent` suffix because they are true ADK agents.
- Every other runtime component is a FunctionTool hosted under
	 `function_tools/<tool_id>` (e.g., `images_sdxl`, `videos_wan`,
	 `tts_chatterbox`, `lipsync_wav2lip`, `assemble_ffmpeg`).
- Stage modules inside `src/sparkle_motion/*_stage.py` coordinate retries and
	 policy enforcement before invoking the FunctionTool endpoints. When this
	 document references “stages,” it is describing those orchestration modules;
	 when it references FunctionTools it is describing the FastAPI entrypoints
	Each envelope is light enough to travel via HTTP/JSON, while the referenced
	artifacts remain in storage. Resume requests simply point at the same
	`asset_uri`s so downstream stages can pick up without recomputing.

### Artifact backend selection

- The runtime now validates `ARTIFACTS_BACKEND` via
	`sparkle_motion.utils.env.resolve_artifacts_backend()`. Leaving the env unset
	defaults to real ADK (`adk`), while setting `ARTIFACTS_BACKEND=filesystem`
	activates the shim. Any other value raises immediately during helper/CLI
	startup so misconfigured notebooks fail fast.
- When the filesystem backend is selected, set
	`ARTIFACTS_FS_ROOT=/content/sparkle_motion/artifacts_fs` (or a Drive path) and
	`ARTIFACTS_FS_INDEX=/content/sparkle_motion/artifacts_fs/index.db` in the same
	env cell. These variables are consumed directly by
	`FilesystemArtifactsConfig.from_env()`, which now reads `os.environ` and
	mkdirs the directories automatically.
- The `Filesystem artifact retention helper` cell already checks that
	`ARTIFACTS_BACKEND=filesystem` before exposing destructive actions; with the
	new validation helpers, the CLI emits a descriptive error when the backend is
	unset or typoed.
- Toggle guidance:
	- Use `ARTIFACTS_BACKEND=adk` for production or when ArtifactService is
		reachable.
	- Use `ARTIFACTS_BACKEND=filesystem` for Colab/offline runs when you rely on
		local disk + SQLite manifests. Remember to export the same env vars before
		running notebooks, CLI retention tasks, or local smoke tests so every
		process points to the same shim root.
- `scripts/filesystem_artifacts.py` now ships `env`, `serve`, and `health`
	commands to keep CLI usage in sync with the notebook helpers:
	- `python scripts/filesystem_artifacts.py env --root /content/artifacts_fs --index /content/artifacts_fs/index.db --token $ARTIFACTS_FS_TOKEN --emit-token` prints shell exports for Colab/Drive sessions.
	- `python scripts/filesystem_artifacts.py serve --host 127.0.0.1 --port 7077`
		launches the FastAPI shim via uvicorn and picks up the same env vars.
	- `python scripts/filesystem_artifacts.py health --url http://127.0.0.1:7077/healthz`
		probes readiness so operators can gate the control panel on a healthy shim.
- The notebook now includes a "Filesystem ArtifactService shim controls" cell
	(just below the Workflow Agent server controls) that wraps those CLI commands
	with ipywidgets buttons: export env vars, start/stop the shim, and run `/healthz`
	before pointing the control panel at the filesystem backend.
- The control panel displays the current artifact backend (ADK vs. filesystem)
	and enables a "Check filesystem health" button whenever
	`ARTIFACTS_BACKEND=filesystem`. This keeps operators aware of which storage
	backend is active when switching between ADK and the local shim.

#### Filesystem artifact evacuation workflow (Colab)

Filesystem artifacts vanish as soon as the Colab VM shuts down unless the shim
root lives on Drive. Operators must decide *where* `ARTIFACTS_FS_ROOT` points at
session start and then follow the matching evacuation flow before teardown.

1. **Prefer Drive-backed roots.** When Drive is mounted, set
	`ARTIFACTS_FS_ROOT=/content/drive/MyDrive/sparkle_motion/artifacts_fs` and keep
	`ARTIFACTS_FS_INDEX` inside the same tree. Running `python
	scripts/filesystem_artifacts.py env --root /content/drive/...` prints the exact
	exports so every tool (CLI, notebook, FastAPI shim) references the persistent
	path. When the root lives on Drive no extra copy step is required—the shim writes
	directly to persistent storage.
2. **Detect ephemeral roots early.** If `ARTIFACTS_FS_ROOT` resolves under
	`/content` (or any non-Drive path), log a warning immediately so the team does
	not forget to copy artifacts off the VM. The retention helper cell now checks
	this and prints a bright banner; production_agent should also emit a
	`write_memory_event` entry each time it publishes to the filesystem backend with
	`{"storage_backend": "filesystem", "root": ARTIFACTS_FS_ROOT}` so the run log
	reminds operators that assets are ephemeral.
3. **Copy artifacts before shutdown.** Run `rsync` (preferred for incremental
	syncs) or `tar` (single archive) to move the entire shim tree onto Drive or your
	local machine *before* ending the session. Example commands from Colab once
	Drive is mounted:

	```bash
	!rsync -avh --progress /content/sparkle_motion/artifacts_fs/ \
	    /content/drive/MyDrive/sparkle_motion/artifacts_fs/
	```

	```bash
	!tar -czf /content/drive/MyDrive/sparkle_motion/artifacts_fs_$(date +%Y%m%d%H%M).tgz \
	    -C /content/sparkle_motion artifacts_fs
	```

	Keep the destination folder in Drive so the next session can point
	`ARTIFACTS_FS_ROOT` back at the synced directory and replay manifests without
	rendering again.
4. **Emit evacuation telemetry.** The evacuation step must be discoverable in
	run history. When a copy starts (either from the notebook helper or a CLI), call
	`adk_helpers.write_memory_event()` with an explicit event type so observers know
	which runs still need manual action. Example snippet for notebook cells:

	```python
	from pathlib import Path
	from sparkle_motion import adk_helpers

	def log_fs_evacuate(run_id: str, src: Path, dest: Path, *, status: str) -> None:
	    adk_helpers.write_memory_event(
	        run_id=run_id,
	        event_type="filesystem_artifacts.evacuate",
	        payload={
	            "artifacts_fs_root": str(src),
	            "destination": str(dest),
	            "status": status,
	            "backend": "filesystem",
	        },
	    )

	log_fs_evacuate(current_run_id, Path(src_root), Path(dest_root), status="pending_copy")
	# ... copy logic ...
	log_fs_evacuate(current_run_id, Path(src_root), Path(dest_root), status="completed")
	```

	Use `status="pending_copy"` before invoking `rsync`/`tar` and `status` values
	like `completed` or `failed` afterward so dashboards can flag unfinished
	evacuations.
5. **Warn before notebook teardown.** Add a final cell that re-runs the retention
	helper plus a lightweight existence check:

	```python
	import os
	from pathlib import Path

	root = Path(os.environ["ARTIFACTS_FS_ROOT"])
	if root.exists() and not str(root).startswith("/content/drive/"):
	    print("WARNING: Filesystem artifacts still live under /content. Copy them to Drive before disconnecting!")
	```

	Pair the warning with `log_fs_evacuate(..., status="warning_emitted")` so
	`/status` and MemoryService timelines reflect that the run still depends on
	manual backup.

Following this checklist ensures filesystem artifacts survive Colab restarts and
that every evacuation shows up in the same telemetry stream as the rest of the
run.

## Detailed design: end-to-end notebook workflow

### 1. Notebook scaffolding

1. **Session bootstrap**
	- Mount Google Drive (if desired) and set `SPARKLE_DB_PATH`, `ARTIFACTS_DIR`,
		and other env vars inside the notebook runtime.
	- Install dependencies via `pip install -r requirements-ml.txt` plus any
		extras (Wan/SDXL torch wheels) the user opts into.
2. **Tool startup cells**
	- Each runtime gets a helper cell: start the two agents (`script_agent`,
		 `production_agent`) first, then launch the FunctionTools (`images_sdxl`,
		 `videos_wan`, `tts_chatterbox`, `lipsync_wav2lip`, `assemble_ffmpeg`) so they bind to `localhost:<port>` via `uvicorn.run` (or the
		 `scripts/run_function_tool.py` helper) inside the notebook session.
	- Cells display readiness indicators by hitting `/ready` endpoints before
		exposing UI controls.
	**Control panel cell**
	- Builds a widget layout (HBox/VBox/GridBox) with the prompt/title inputs plus
		new helpers: a `Plan URI` field paired with a `Load Plan JSON` button, a run
		mode dropdown (`Mode`), a `Send plan inline` checkbox, and
		`Run ID`/`Stage` filters for the artifact viewer. All controls sit above the
		same `Generate Plan`, `Run Production`, `Pause`, `Resume`, and `Stop`
		buttons.
	- The plan viewer area now auto-renders summary counts (plan_id, shots,
		base-images, dialogue entries) whenever a plan artifact is parsed, so
		operators can confirm schema health before triggering production.
	- Button callbacks call helper coroutines that invoke:
		- `POST http://localhost:8101/invoke` to generate a MoviePlan.
		- `POST http://localhost:8200/invoke` with the plan or artifact URI to run
			production_agent.
	- A lightweight polling task (async or timer-based) refreshes `/ready` and
		`/status` results and pushes the data into widget displays. Users can also
		toggle a streaming status pane, auto-fetch artifacts, or manually refresh
		specific stages through the same control surface.

### 2. Plan generation + review loop

1.	User enters a prompt/title and clicks `Generate Plan`.
2.	The `Generate Plan` button callback posts the payload via Python, captures
	`artifact_uri`, and caches the validated plan in a notebook variable while also
	autofilling the `Plan URI` text field.
3.	Users click `Load Plan JSON` to parse the artifact (supports `file://` URIs)
	and the `Plan Details` pane prints a summary with plan_id, shot/base-image
	counts, and dialogue timeline entries. This feedback loop happens before any
	production call so schema regressions are caught immediately.
4.	Widgets (table, accordion, or text area) reflect the plan JSON so users can
	inspect/edit before production. Checking “Send plan inline” embeds the cached
	payload in the next `production_agent` request; leaving it unchecked causes the
	panel to reference the artifact URI instead.
5.	UI exposes a “Review Plan” section showing shots, the script_agent-generated
	dialogue timeline, and the base-image thumbnails; user can edit inline or
	download/edit/upload before approving production.
6.	If edits are made, notebook cell updates the plan payload before handing it
	off to production_agent. Manual edits run through the same MoviePlan schema
	validator before any downstream call, so malformed changes are surfaced to the
	user immediately. (Future work: expose “Refine with ScriptAgent” actions so
	users can request targeted regenerations directly from the notebook UI.)

#### Run modes

- The `Mode` dropdown mirrors the production_agent `mode` flag (`dry` vs.
	`run`). Dry-runs stop after validation so operators can iterate quickly, while
	`run` drives the full clip rendering + lipsync path.
- The control panel surfaces a static finalize-stage reminder so operators
	perform manual review of the published `video_final` artifact before sharing
	anything externally.

#### Plan artifact loader vs. inline submission

- The `Plan URI` field is set automatically after `script_agent` responds but
	can also accept pasted URIs from prior runs. The `Load Plan JSON` button reads
	the artifact from disk (or `file://` URIs) and caches the `validated_plan`
	payload for reuse.
- When “Send plan inline” is checked, the cached payload is embedded directly in
	the `production_agent` request. Otherwise the run references the artifact URI.
	This makes it easy to hand-edit plans in the notebook and send them without
	writing a new artifact.
- The plan summary output documents how many shots, base images, and dialogue
	entries the payload contains so discrepancies are obvious before a run starts.

#### Status + artifact monitoring

- `Poll Status` spins up an `asyncio` task that alternates between `/ready` and
	`/status` calls. The `Interval (s)` widget adjusts cadence without editing
	code.
- Enabling `Auto-fetch artifacts` causes every poll iteration to call
	`/artifacts` for the current Run ID, optionally filtered by the `Stage` text
	box (e.g., `assemble`).
- The `Refresh Artifacts` button triggers an immediate fetch using the same
	filters so operators can pause polling but still inspect new files on demand.
- Separate `widgets.Output` panes display one-shot control responses, the live
	status stream, and artifact JSON. This keeps run telemetry, artifacts, and plan
	state visible without scrolling between notebook cells.

#### Control panel quickstart cell

- The Colab notebook now includes a dedicated "Quickstart: Launch the control
	panel" section. Run the accompanying code cell to execute:

	```python
	from notebooks.control_panel import create_control_panel
	control_panel = create_control_panel()
	control_panel
	```

	This cell imports `create_control_panel`, instantiates the widget stack using
	the `local-colab` profile, and assigns the resulting object to a global
	`control_panel` variable so downstream helper cells (final deliverable preview,
	plan inspectors, etc.) can reuse the same run metadata.
- When the quickstart cell completes you should see the ipywidgets surface
	directly in the notebook, matching the screenshots earlier in this document.
	If the FastAPI endpoints are not running, the cell displays the HTTP error in
	the Script Agent or Production Agent output panes so you can restart the tool
	process.
- Operators who need custom timeouts or alternative endpoint profiles can skip
	the quickstart cell and instead run the "Notebook control panel prototype"
	section immediately below it, which demonstrates manual `PanelEndpoints`
	creation and `ControlPanel` instantiation.

### Agent + FunctionTool port map (local-colab profile)

The control panel assumes the `local-colab` ToolRegistry profile defined in
`configs/tool_registry.yaml`. Launch each runtime (agents first, then
FunctionTools) inside the Colab VM with the command shown below. Only override
ports when another process already occupies the default.

| Runtime type | ID | Default port | Launch snippet | Fixture / smoke env vars |
| --- | --- | --- | --- | --- |
| Agent | `script_agent` | `5001` | `PYTHONPATH=src python scripts/run_function_tool.py --tool script_agent --port 5001` | `SCRIPT_AGENT_MODEL` (LLM to call), `ADK_USE_FIXTURE=1` to keep deterministic outputs |
| FunctionTool | `images_sdxl` | `5002` | `PYTHONPATH=src python scripts/run_function_tool.py --tool images_sdxl --port 5002` | `SMOKE_IMAGES=1` / `SMOKE_ADAPTERS=1` to exercise SDXL, `IMAGES_SDXL_FIXTURE_ONLY=1` to pin fixtures, `IMAGES_SDXL_MODEL` + `IMAGES_SDXL_DEVICE` select checkpoints |
| FunctionTool | `videos_wan` | `5003` | `PYTHONPATH=src python scripts/run_function_tool.py --tool videos_wan --port 5003` | `SMOKE_VIDEOS=1`, `SMOKE_ADAPTERS=1`, or `SMOKE_ADK=1` for Wan2.1, `VIDEOS_WAN_FIXTURE_ONLY=1` for fixtures, `VIDEOS_WAN_MODEL` / `VIDEOS_WAN_DEVICE_PRESET` override model/device maps |
| FunctionTool | `tts_chatterbox` | `5004` | `PYTHONPATH=src python scripts/run_function_tool.py --tool tts_chatterbox --port 5004` | `SMOKE_TTS=1` / `SMOKE_ADAPTERS=1` for real synthesis, `TTS_CHATTERBOX_MODEL`, `TTS_CHATTERBOX_DEVICE`, `TTS_CHATTERBOX_CACHE_TTL_S` tune the backend |
| FunctionTool | `lipsync_wav2lip` | `5005` | `PYTHONPATH=src python scripts/run_function_tool.py --tool lipsync_wav2lip --port 5005` | `SMOKE_LIPSYNC=1` (or `SMOKE_ADAPTERS=1`) to invoke Wav2Lip, `LIPSYNC_WAV2LIP_FIXTURE_ONLY=1` to force fixtures, `WAV2LIP_REPO` / `WAV2LIP_CHECKPOINT` point at local assets |
| FunctionTool | `assemble_ffmpeg` | `5006` | `PYTHONPATH=src python scripts/run_function_tool.py --tool assemble_ffmpeg --port 5006` | `SMOKE_ASSEMBLE=1` / `SMOKE_ADAPTERS=1` to run ffmpeg, `ASSEMBLE_FFMPEG_FIXTURE_ONLY=1` to clamp fixtures, `FFMPEG_PATH` selects the binary |
| Agent | `production_agent` | `5008` | `PYTHONPATH=src python scripts/run_function_tool.py --tool production_agent --port 5008` | Stage-level flags (`SMOKE_TTS`, `SMOKE_LIPSYNC`, `SMOKE_VIDEOS`, `SMOKE_IMAGES`, `SMOKE_ASSEMBLE`, `SMOKE_ADAPTERS`) control which adapters run for real; `SPARKLE_DB_PATH` picks the SQLite store; `ADK_USE_FIXTURE=1` keeps every FunctionTool on fixtures |

All commands assume the `sparkle_motion` conda environment is active and that
your notebook session exported `SPARKLE_DB_PATH`/`ARTIFACTS_DIR` before launching
any tools. If you change ports or environment variables, update
`configs/tool_registry.yaml` and rerun the quickstart cell so the control panel
points at the correct endpoints.

##### Dialogue timeline structure (script_agent output)

- `MoviePlan` now carries a top-level `dialogue_timeline` array instead of
	embedding dialogue per shot. Each entry is either `{"type": "dialogue", ...}`
	or `{"type": "silence", ...}` with inclusive `start_time_sec` and
	`duration_sec`.
- Dialogue entries contain `character_id`, `text`, and an estimated
	`duration_sec` generated by script_agent (words-per-second heuristic). Silence
	entries mark gaps between dialogue segments so downstream stages see a single
	continuous schedule.
- The total runtime of the dialogue timeline (dialogue + silence durations)
	must equal the sum of `duration_sec` across all shots. Production_agent and
	the notebook will enforce this invariant after any edits.
- After TTS synthesis completes we may revise `duration_sec` values to match
	actual clip lengths, but the structure (ordering + silence markers) remains the
	same so replays stay deterministic.

##### Base images inventory

- Script agent emits a top-level `base_images` array sized to `len(shots) + 1`.
	Each entry includes an `id`, `prompt`, and optional metadata (seed, guidance,
	style cues). The extra image ensures the final video end frame is represented.
- Every shot references the precomputed assets via
	`start_base_image_id` / `end_base_image_id` rather than duplicating prompts.
	For shot *i*, the start reference must match `base_images[i]` and the end
	reference must match `base_images[i+1]` so continuity is guaranteed.
- Images agents can iterate over `base_images` to generate the full stack of
	keyframes up front. Production_agent then hands those URIs to `videos_wan`
	when rendering motion between the linked start/end frames. Once generated, the
	base images are treated as immutable artifacts; video synthesis tools read
	them but never mutate or overwrite the originals.
- After each base image renders, operators should manually inspect the cached
	preview (finger counts, prompt adherence). If an image looks off, use the
	`images` resume helper to regenerate before paying for video or TTS; this keeps
	later steps fed with clean inputs even without automated QA.
- When editors adjust shot durations they must also ensure the base image count
	still equals `len(shots) + 1`; otherwise notebook validation will block the
	run with a timeline/keyframe mismatch error.

##### Render profile configuration (new requirement)

- The plan now carries a mandatory `render_profile` object that locks in which
	model family each stage must use (video, TTS, SDXL variant, etc.). This keeps
	replayability and consistency in step with how other stages already expose
	model choices.
- For the video stage we start with a single entry:

	```json
	"render_profile": {
		"video": {
			"model_id": "wan-2.1",
			"max_fps": 24,
			"notes": "Default WAN 2.1 checkpoint"
		}
	}
	```

- Script agent (or the notebook UI) sets the defaults; production_agent simply
	validates and enforces what is declared. There is no backward-compatibility
	requirement, so runs fail fast if the new block is missing.
- The MoviePlan schema + notebook validators must be updated to treat
	`render_profile.video.model_id` as required before users can trigger
	production_agent.

#### Script agent JSON payloads

- The immediate HTTP response from `/invoke` is a lightweight envelope with
	`status`, `artifact_uri`, and `request_id`. The actual plan lives inside the
	artifact JSON file that the tool writes to `ARTIFACTS_DIR` (and uploads via
	`publish_artifact`).
- Each artifact contains both the original request body and the validated
	`MoviePlan`, so notebooks can reload plans without rerunning the model.
- Sample artifact payload:

```json
{
	"request": {
		"title": "Rooftop Revelations",
		"prompt": "Two friends on a neon rooftop reconcile before sunrise."
	},
	"validated_plan": {
		"title": "Rooftop Revelations",
		"characters": [
			{
				"id": "ava",
				"name": "Ava",
				"description": "Street photographer processing a hard truth",
				"voice_profile": {
					"style": "confessional",
					"language": "en-US"
				}
			},
			{
				"id": "miko",
				"name": "Miko",
				"description": "DJ finishing a final set before leaving town",
				"voice_profile": {
					"style": "calm",
					"language": "en-US"
				}
			}
		],
		"base_images": [
			{
				"id": "frame_000",
				"prompt": "Night skyline, pink neon, rain streaks on lens",
				"metadata": {"seed": 99123}
			},
			{
				"id": "frame_001",
				"prompt": "Camera settles behind Ava as she faces Miko",
				"metadata": {"seed": 99124}
			},
			{
				"id": "frame_002",
				"prompt": "Closeup on clasped hands lit by cyan signage",
				"metadata": {"seed": 99125}
			}
		],
		"render_profile": {
			"video": {
				"model_id": "wan-2.1",
				"max_fps": 24,
				"runtime": "wan_v2.1_default"
			}
		},
		"shots": [
			{
				"id": "shot_001",
				"duration_sec": 8.0,
				"setting": "Neon rooftop with rain pooling on glass tiles",
				"visual_description": "Wide establishing view of the skyline",
				"start_base_image_id": "frame_000",
				"end_base_image_id": "frame_001",
				"motion_prompt": "Slow crane down while rain intensifies",
				"is_talking_closeup": false
			},
			{
				"id": "shot_002",
				"duration_sec": 9.5,
				"setting": "Same rooftop, tighter framing near a glowing billboard",
				"visual_description": "Medium shot as the friends talk through tears",
				"start_base_image_id": "frame_001",
				"end_base_image_id": "frame_002",
				"motion_prompt": "Handheld push-in that ends on their hands",
				"is_talking_closeup": true
			}
		],
		"dialogue_timeline": [
			{
				"type": "dialogue",
				"character_id": "ava",
				"text": "I was scared of losing us.",
				"start_time_sec": 8.4,
				"duration_sec": 2.6
			},
			{
				"type": "dialogue",
				"character_id": "miko",
				"text": "You never had to be.",
				"start_time_sec": 11.2,
				"duration_sec": 2.2
			},
			{
				"type": "silence",
				"start_time_sec": 13.4,
				"duration_sec": 4.1
			}
		],
		"metadata": {
			"source": "script_agent.generate_plan",
			"shot_count": "2",
			"seed": "12345"
		}
	},
	"generated_at": "2025-11-29T03:41:12.714218+00:00",
	"metadata": {
		"tool": "script_agent",
		"request_id": "4d5413f6c0984b14a38c4d92658a4a2f",
		"shot_count": 2
	},
	"schema_uri": "gs://sparkle-motion-schemas/MoviePlan.schema.json"
}
```

- Notebooks can either persist `artifact_uri` directly (for production_agent to
	consume) or parse the `validated_plan` block above to prefill UI tables,
	start/end frame thumbnails, and dialogue timeline builders.

### 3. Dialogue timeline + TTS synthesis

1. **Timeline builder**
	- Consume the plan’s `dialogue_timeline` directly. Each dialogue line already
		has an estimated `start_time_sec` + `duration_sec`; silence entries mark the
		gaps. If the user edits the script, the notebook recalculates timings so the
		total still equals the cumulative shot runtime.
2. **TTS adapter call**
	- `production_agent` (or a dedicated timeline stage) sends the timeline to
		tts_stage with a request to synthesize each line.
	- Adapter returns per-line artifacts (`line_artifacts`) plus raw WAVs and
		updated durations based on the rendered speech.
3. **Stitch + silence insertion**
	- Helper reconciles estimated vs. actual durations, then creates a single WAV
		by concatenating line clips and inserting (or trimming) silence so each entry
		still matches the plan’s absolute timestamps.
	- Metadata records the offsets so downstream stages (lipsync, assemble) can
		rely on exact timing.
	- The stage writes `dialogue_timeline_audio.json`, which contains each
		`line_artifact` plus `start_time_actual_s`, `end_time_actual_s`,
		`duration_audio_raw_s`, `timeline_padding_s`, and `timeline_trimmed_s`. A
		`timeline_offsets` map mirrors these values so `/artifacts` consumers can
		rebuild the stitched WAV alignment without reprocessing audio.

### 4. Video clip generation + continuity

1. For each shot, `production_agent` calls `images_sdxl` once to render the
	full immutable `base_images` list. Operators should spot-check the resulting
	frames (finger counts, prompt adherence) before continuing. Once the stack is
	approved, the relevant start/end assets go to `videos_wan` with shot-specific
	`num_frames`, `fps`, and continuity inputs:
	- `start_frame`: `base_images[start_base_image_id]` (which also equals the
		previous shot’s end frame) to seed continuity.
	- `end_frame`: `base_images[end_base_image_id]`, stored for use by the next
		shot.
2. Clips are typically 5–10 seconds; metadata tracks clip length, chunk index,
	and prompts.
3. AssetRefs stores references so assemble/lipsync stages can read the correct
	files.

### 5. Manual clip review + retries

Clip-level automation is paused inside the notebook workflow. After each
`videos_wan` invocation completes, operators should:

1. Inspect the emitted MP4 plus thumbnails via the artifact viewer to confirm
	prompt adherence, continuity, and safety requirements.
2. Use `Resume from stage` helpers to rerun `images`/`videos` selectively when a
	shot needs regeneration. The control panel logs these manual retries next to
	the StepExecutionRecord snapshot so history stays auditable.
3. Document any manual overrides (e.g., “shot_002 regenerations: 2”) in the run
	notes field or via `adk_helpers.write_memory_event()` so `/status` reflects the
	manual review coverage while automation is paused.

### 6. Assembly + lipsync integration

1. `assemble_ffmpeg` (or a new pipeline stage) concatenates raw clips in shot
	order to produce a silent master video, preserving per-shot durations.
2. The stitched WAV from the TTS stage is aligned to this master video (they
	share the same total duration + silence padding).
3. `lipsync_wav2lip` receives both the silent video and the full WAV, producing
	a final MP4 where the stitched audio drives lip movements across clip
	boundaries.
4. Artifacts emitted:
	- `tts_timeline.wav` (single audio track with silence).
	- `video_raw_concat.mp4` (pre-lipsync video).
	- `video_final.mp4` (post-lipsync, final deliverable).

#### production_agent stage inputs & outputs

The notebook needs to know exactly what data each production_agent stage
consumes and emits so it can cache artifacts, surface validation errors early,
and offer resume controls. The table below expands the high-level bullets from
the user prompt into concrete inputs/outputs:

| Stage | Required inputs | Primary outputs | Notes |
| --- | --- | --- | --- |
| **Plan intake** | MoviePlan JSON (inline or fetched via `artifact_uri`), schema hash, policy gate config, environment paths (`ARTIFACTS_DIR`, etc.), `render_profile` block | Canonical `RunContext` (shot order, dialogue timeline, base-image map, selected render profiles), validation report, policy audit log | Fails fast if schema mismatch, missing base-image references, or render_profile absent/unsupported. |
| **Dialogue & audio** | `dialogue_timeline`, character voice profiles, TTS model settings, optional pronunciation overrides | Per-line WAVs + metadata (durations, viseme hints), stitched `tts_timeline.wav`, `dialogue_timeline_audio.json` with measured offsets/duration deltas | Stitched file length must equal cumulative shot duration; offsets stored for assemble/lipsync. |
| **Base images** | `base_images` array, SDXL config (prompt, seed, guidance), regeneration policy | PNG/WEBP URIs (one per base image), render metadata, optional reviewer notes | Operators manually inspect frames after rendering and re-run `images_sdxl` when continuity or safety issues appear. |
| **Video clips** | Shot list (start/end base-image URIs, duration, motion prompt, fps, frame count), video `render_profile` entry (model_id, caps), continuity constraints | Per-shot MP4/PNG sequences, motion metadata (num frames, seed), continuity confirmation records | Uses shot *N* end frame as shot *N+1* start frame; stores asset refs for assembly; honoring the declared model is mandatory. |
| **Assembly & lipsync** | Ordered clip URIs, per-shot durations, stitched WAV, lipsync model config | `video_raw_concat.mp4`, lip-synced `video_final.mp4`, alignment map between audio and frames | Ensures total runtime parity between audio and video before invoking lipsync. |
| **Manual review & publish** | Final video/audio artifacts, run metadata, reviewer checklist | Notebook/Drive notes, `video_final` artifact manifest, memory events documenting review outcome | Teams perform safety/continuity signoff here until automated review returns; use `/artifacts` + inline previews to guide decisions. |

##### Manual review expectations

- Because automated checks have been removed, every production run should include a
	manual review note before sharing outputs. The control panel surfaces a
	reminder banner near the final deliverable helper; update it with reviewer
	initials and date when signoff is complete.
- The notebook honors `QA_AUTOMATION_REMOVED=1` by default. Leaving it set keeps
	the warning banner visible, forces CLI/control-panel payloads to send
	`qa_mode="disabled"`, and records `manual_review_required=True` in manifests.
- Known limitations while the flag remains enabled: final deliverable cells no
	longer show QA badges, `/status` timelines omit QA entries, and downstream
	automation must rely on the manual review memory events described above.
- If a run bypasses manual review (e.g., fixture-only experimentation), log the
	decision via `adk_helpers.write_memory_event(event_type="finalize_manual_review",
	payload={"status": "skipped"})` so `/status` and artifacts history call out the
	exception.

##### Stage-to-stage payload examples

All stage RPCs remain JSON-based envelopes. Heavy assets (images, audio, video)
are never embedded directly; we pass strongly typed `asset_uri` references that
point at files in `ARTIFACTS_DIR` or cloud storage. Each stage adds metadata and
artifact manifests so the notebook (or a retry) can rehydrate state without
rerunning upstream work.

###### Plan intake

**Input envelope**

```json
{
	"stage": "plan_intake",
	"run_id": "run_9c29599b",
	"inputs": {
		"plan_uri": "gs://sparkle-motion-artifacts/run_9c29599b/movie_plan.json",
		"inline_plan": {"title": "Rooftop Revelations", "shots": [...], "base_images": [...], "dialogue_timeline": [...]},
		"schema_hash": "sha256:7f3c...",
		"policy_profile": "default",
		"render_profile": {
			"video": {
				"model_id": "wan-2.1",
				"max_fps": 24
			}
		}
	}
}
```

**Output envelope**

```json
{
	"stage": "plan_intake",
	"status": "succeeded",
	"outputs": {
		"run_context": {
			"shot_order": ["shot_001", "shot_002"],
			"dialogue_timeline": "artifact://run_9c29599b/dialogue_timeline.json",
			"base_image_map": {
				"frame_000": "gs://sparkle-motion-artifacts/run_9c29599b/base_images/frame_000_prompt.json"
			},
			"render_profile": {
				"video": {
					"model_id": "wan-2.1",
					"max_fps": 24
				}
			}
		},
		"validation_report_uri": "gs://.../plan_validation.json"
	},
	"artifacts": []
}
```

###### Dialogue & audio

**Input envelope**

```json
{
	"stage": "dialogue_audio",
	"run_id": "run_9c29599b",
	"inputs": {
		"dialogue_timeline_uri": "artifact://run_9c29599b/dialogue_timeline.json",
		"voice_profiles": {
			"ava": {"style": "confessional", "language": "en-US"},
			"miko": {"style": "calm", "language": "en-US"}
		},
		"tts_config": {"model": "chatterbox-v1", "temperature": 0.4}
	}
}
```

**Output envelope**

```json
{
	"stage": "dialogue_audio",
	"status": "succeeded",
	"outputs": {
		"line_artifacts": [
			{
				"line_id": "line_000",
				"audio_uri": "gs://.../tts/line_000.wav",
				"duration_sec": 2.58,
				"viseme_stream_uri": "gs://.../tts/line_000.viseme.json"
			}
		],
		"stitched_audio_uri": "gs://.../tts/tts_timeline.wav",
		"timeline_with_actuals_uri": "gs://.../tts/dialogue_timeline_actuals.json"
	},
	"artifacts": ["gs://.../tts/tts_timeline.wav"]
}
```

###### Base images

**Input envelope**

```json
{
	"stage": "base_images",
	"run_id": "run_9c29599b",
	"inputs": {
		"base_image_prompts": [
			{"id": "frame_000", "prompt": "Night skyline...", "metadata": {"seed": 99123}},
			{"id": "frame_001", "prompt": "Camera settles..."}
		],
		"sdxl_config": {"scheduler": "dpm++", "guidance": 7.5}
	}
}
```

**Output envelope**

```json
{
	"stage": "base_images",
	"status": "succeeded",
	"outputs": {
		"base_image_assets": [
			{"id": "frame_000", "asset_uri": "gs://.../base_images/frame_000.png", "mime_type": "image/png"},
			{"id": "frame_001", "asset_uri": "gs://.../base_images/frame_001.png", "mime_type": "image/png"}
		],
	},
	"artifacts": ["gs://.../base_images/base_images_manifest.json"]
}
```

###### Video clips

**Input envelope**

```json
{
	"stage": "video_clips",
	"run_id": "run_9c29599b",
	"inputs": {
		"shots": [
			{
				"shot_id": "shot_001",
				"duration_sec": 8.0,
				"motion_prompt": "Slow crane down...",
				"fps": 24,
				"num_frames": 192,
				"start_frame_uri": "gs://.../base_images/frame_000.png",
				"end_frame_uri": "gs://.../base_images/frame_001.png"
			}
		],
		"renderer_profile": {
			"model_id": "wan-2.1",
			"max_fps": 24,
			"cfg_scale": 6.5
		}
	}
}
```

**Output envelope**

```json
{
	"stage": "video_clips",
	"status": "succeeded",
	"outputs": {
		"clip_assets": [
			{"shot_id": "shot_001", "asset_uri": "gs://.../video_clips/shot_001.mp4", "duration_sec": 8.0, "continuity_verified": true}
		],
		"motion_metadata_uri": "gs://.../video_clips/motion_manifest.json"
	},
	"artifacts": ["gs://.../video_clips/shot_001.mp4"]
}
```

###### Assembly & lipsync

**Input envelope**

```json
{
	"stage": "assembly_lipsync",
	"run_id": "run_9c29599b",
	"inputs": {
		"clip_uris": ["gs://.../video_clips/shot_001.mp4", "gs://.../video_clips/shot_002.mp4"],
		"stitched_audio_uri": "gs://.../tts/tts_timeline.wav",
		"clip_durations": [8.0, 9.5],
		"lipsync_config": {"model": "wav2lip-hq", "face_detector": "s3fd"}
	}
}
```

**Output envelope**

```json
{
	"stage": "assembly_lipsync",
	"status": "succeeded",
	"outputs": {
		"video_raw_uri": "gs://.../assembly/video_raw_concat.mp4",
		"video_final_uri": "gs://.../assembly/video_final.mp4",
		"alignment_map_uri": "gs://.../assembly/alignment_map.json"
	},
	"artifacts": ["gs://.../assembly/video_final.mp4"]
}
```

Each envelope is light enough to travel via HTTP/JSON, while the referenced
artifacts remain in storage. Resume requests simply point at the same
`asset_uri`s so downstream stages can pick up without recomputing.

### 7. Notebook UI feedback

1. Progress log displays each stage’s StepExecutionRecord as it completes (e.g.,
	via `widgets.Output` or `widgets.Accordion`).
2. Artifact table lists URIs + local paths for plan, audio, video, assemble outputs, and any reviewer notes.
3. Optional preview section uses `IPython.display.Video` or
	`google.colab.widgets.TabBar` to show the final MP4 when ready.
4. Error states show the stage name, error message, and a “retry stage” action
	that calls production_agent with `resume_from=<stage>`.

#### Production run dashboard + controls

- **Live status polling**: production_agent now persists a `StepExecutionRecord`
	per stage transition and surfaces the rolling history via
	`GET /status?run_id=<id>`. The response describes `current_stage`,
	`status`, `started_at`, `completed_at`, percent complete, and the last N log
	lines so the notebook can render a streaming timeline. Poll every 3–5 seconds
	or upgrade to server-sent events later; stream updates into a shared
	`widgets.Output` so the Colab UI stays responsive.
- **Immediate asset previews**: every stage that emits media exposes a short
	manifest through `GET /artifacts?run_id=<id>&stage=<name>`. Entries include
	`asset_uri`, `media_type`, labels, and optional thumbnails. The control panel
	renders these with `widgets.Image`, `widgets.Audio`, and `IPython.display.Video`
	(or `google.colab.widgets.TabBar`) so previews stay entirely Python-driven.
	Include `stage="dialogue_audio"` in the sample notebook snippet so operators immediately
	see the stitched `tts_timeline.wav` row alongside per-line dialogue manifests without
	guessing which filter to use. The `/artifacts` response now returns a `stages` array with
	per-stage summaries (`count`, `artifact_types`, `media_types`) plus the flattened
	`artifacts` list, so helpers should lean on that metadata instead of re-filtering
	by hand. Example helper:

	- Shot-scoped manifests now include `shot_frames`, `shot_dialogue_audio`, `shot_video`, `shot_lipsync_video`,
	  and `assembly_plan`. Surface any `metadata.review_notes`, `review_status`, or similar manual-review fields
	  next to their corresponding media rows so operators can immediately see what still needs attention. Reviewer
	  attachments (screenshots, transcripts) live alongside the manifest within the same
	  `artifacts/runs/<run_id>/<plan_id>/<stage>/<shot_id>/` tree; link to those files when operators expand a row
	  so they can capture context before deciding whether to resume a stage.

	```python
	import requests
	from IPython.display import Audio, display

	def fetch_dialogue_audio(run_id: str) -> tuple[list[dict], dict]:
	    resp = requests.get(
	        f"{PROD_BASE}/artifacts",
	        params={"run_id": run_id, "stage": "dialogue_audio"},
	        timeout=10,
	    )
	    resp.raise_for_status()
	    payload = resp.json()
	    stage_section = payload["stages"][0]  # stage filter ensures exactly one section
	    return stage_section["artifacts"], {
	        "count": stage_section["count"],
	        "artifact_types": stage_section["artifact_types"],
	        "media_types": stage_section["media_types"],
	    }

	dialogue_rows, dialogue_meta = fetch_dialogue_audio(RUN_ID)
	print(f"Loaded {dialogue_meta['count']} dialogue artifacts ({', '.join(dialogue_meta['media_types'])})")
	timeline = next(entry for entry in dialogue_rows if entry["artifact_type"] == "tts_timeline_audio")
	display(Audio(filename=timeline["local_path"], autoplay=False))
	```

	The same rows drive whatever UI widget lists the individual per-line clips (the manifest’s
	`metadata.entry_count` clarifies how many to expect). Each `/artifacts` response now includes
	per-stage `preview` metadata with the first image/audio/video entries plus aggregates such as
	`media_summary` (counts, total duration, playback readiness) and optional `review_summary`. Use those
	fields to show quick badges (e.g., “2 audio clips · 44s total”) or pick the default player without
	scanning the full manifest manually.
- **Pause / resume / stop**: add control endpoints to production_agent so the
	notebook can orchestrate long-running jobs:
	- `POST /control/pause` → `{ "run_id": "..." }`
	- `POST /control/resume` → `{ "run_id": "..." }`
	- `POST /control/stop` → `{ "run_id": "..." }`
	Internally these map to an asyncio Event gate—pause holds the next stage
	while preserving artifacts, resume releases the gate, and stop tears down the
	run gracefully while marking the status feed with `status: "stopped"`.
- **Event schema**: widget callbacks consume the `/status` payload in this shape
	(abbreviated):

	```json
	{
	  "run_id": "run_9c29599b",
	  "current_stage": "assemble",
	  "status": "running",
	  "progress": 0.62,
	  "log": [
	    {
	      "stage": "lipsync",
	      "status": "succeeded",
	      "started_at": "2025-11-29T03:52:10Z",
	      "completed_at": "2025-11-29T03:54:02Z",
	      "artifacts": ["gs://.../lipsync/lipsync_manifest.json"]
	    }
	  ]
	}
	```

- **Colab control cell**: reuse the ipywidgets pattern from the script_agent
	cell. The layout includes:
	- Header row with “Start production”, “Pause”, “Resume”, “Stop” buttons wired
	  to the endpoints above (disabled/enabled based on current status).
	- Stage progress table that binds the `/status` response to rows with
	  timestamps, durations, and pass/fail badges.
	- Asset accordion with tabs for base images, dialogue/TTS clips, video clips,
	  assembly outputs, and reviewer notes/manual checklists. Each tab fetches the matching manifest and
	  renders thumbnails/players via `widgets.Image`, `widgets.Audio`, or
	  `IPython.display.Video`.
	- Alert banner that flips red if the latest StepExecutionRecord reports
	  `status in {"failed", "stopped"}` and exposes a “resume from stage” button
	  that posts back to `/invoke` with `resume_from` populated.

This richer dashboard keeps the user informed about which stage is running,
lets them preview every artifact as soon as it exists, and gives them agency to
pause/resume/stop long productions without leaving the notebook.

	#### Artifact preview recipes (images, audio, video)

	Once `/artifacts` begins returning the richer `stages` payload, the notebook should
	follow a consistent recipe for inline previews so operators never have to inspect
	raw JSON. Every helper below assumes the response already filtered to a specific
	stage (by passing `stage=...` in the query) which means `payload["stages"][0]`
	is the manifest section to render.

	1. **Extract the stage manifest** (count, artifact/media summaries, preview hints)
	   and keep the flattened `artifacts` list for widget rendering:

	   ```python
	   from pathlib import Path
	   from typing import Any, Dict, List

	   def load_stage_manifest(run_id: str, stage: str) -> Dict[str, Any]:
		   resp = requests.get(
			   f"{PROD_BASE}/artifacts",
			   params={"run_id": run_id, "stage": stage},
			   timeout=15,
		   )
		   resp.raise_for_status()
		   data = resp.json()
		   if not data["stages"]:
			   raise RuntimeError(f"Stage {stage} returned no artifacts")
		   return data["stages"][0]
	   ```

	2. **Images**: prefer `widgets.Image` so previews stay in-line without blocking
	   autoplay. Use manifest metadata (width/height, format) when present but fall
	   back to the filename extension.

	   ```python
	   import base64
	   import mimetypes
	   import ipywidgets as widgets

	   def render_image_row(entry: Dict[str, Any], *, width: int = 320) -> widgets.Widget:
		   local_path = Path(entry["local_path"])
		   mime = entry.get("metadata", {}).get("mime_type") or mimetypes.guess_type(local_path.name)[0] or "image/png"
		   encoded = base64.b64encode(local_path.read_bytes())
		   return widgets.Image(value=encoded, format=mime.split("/")[-1], width=width)
	   ```

	   Wrap these in a `widgets.HBox` / `widgets.Accordion` when stages emit many
	   thumbnails (e.g., `base_images`). If `local_path` is missing, download the
	   file to `/content/artifacts/<run_id>/` before rendering and always update the
	   manifest row with the cached path so future renders stay local.

	3. **Audio**: rely on `IPython.display.Audio` to embed waveforms in the same
	   output panel. Respect the manifest’s `media_summary.audio.total_duration_s`
	   when labeling clips so users can see runtime at a glance.

	   ```python
	   from IPython.display import Audio, display

	   def render_audio_stage(manifest: Dict[str, Any]) -> None:
		   rows = manifest["artifacts"]
		   print(f"Loaded {len(rows)} audio artifacts ({manifest['media_summary']['audio']['total_duration_s']:.1f}s)")
		   for row in rows:
			   label = row.get("label") or row.get("artifact_type", "audio")
			   display(Audio(filename=row["local_path"], autoplay=False))
			   print(label, "→", row["local_path"])
	   ```

	4. **Video**: `IPython.display.Video` keeps the UX consistent with the final
	   deliverable helper. Use the manifest’s `preview.video.playback_ready` flag to
	   decide whether to show the inline player or emit a warning that the clip must
	   be downloaded manually first.

	   ```python
	   from IPython.display import Video

	   def render_video_entry(entry: Dict[str, Any], *, width: int = 640) -> None:
		   if not Path(entry.get("local_path", "")).exists():
			   raise FileNotFoundError("Video local_path missing; download the artifact first")
		   display(Video(filename=entry["local_path"], embed=True, width=width))
	   ```

	5. **Finalize status banner**: every preview widget should read the manifest’s
	   finalize metadata (e.g., `finalize_summary` or the stage-level `manual_review`
	   flags) and add a banner when the clip still needs manual validation. Keep the
	   banner even when the media renders successfully so operators remember to log a
	   `finalize_manual_review` result before sharing assets.

	These helpers live in regular notebook cells (or a lightweight
	`notebooks/preview_helpers.py` module) so the control panel outputs can call
	`render_image_row`, `render_audio_stage`, or `render_video_entry` whenever the
	user toggles the Artifacts pane. Keeping the recipes centralized also makes it
	easy to evolve the UX (e.g., switch to carousels) without rewriting each cell.

	#### Final deliverable preview & download

	- The `finalize` stage must emit an `/artifacts` manifest entry with
		`artifact_type: "video_final"`, `artifact_uri`, `local_path` (if production_agent
		downloaded the file into the Colab runtime), and `download_url` (signed ADK URL
		when the asset only lives in ArtifactService). The dashboard surfaces this entry
		in a dedicated “Final Video” tab so operators immediately see the latest
		deliverable and whether manual review is outstanding.
	- Provide a convenience notebook cell that fetches the manifest, previews the
		video inline, and offers a download button. Example:

		```python
		import subprocess
		import requests
		from pathlib import Path
		from IPython.display import Video, display
		from google.colab import files

		PROD_BASE = "http://localhost:8200"
		RUN_ID = current_run_id  # captured when production started

		def fetch_final_entry(run_id: str) -> dict:
		    resp = requests.get(
		        f"{PROD_BASE}/artifacts",
		        params={"run_id": run_id, "stage": "finalize"},
		        timeout=10,
		    )
		    resp.raise_for_status()
		    payload = resp.json()
		    stage_section = payload["stages"][0]  # stage filter keeps this scoped to finalize
		    preview_video = stage_section["preview"]["video"]
		    media_summary = stage_section["media_summary"].get("video", {})
		    print(
		        "finalize emitted"
		        f" {stage_section['count']} artifact(s): {stage_section['artifact_types']} |"
		        f" video preview ready={preview_video['playback_ready']}"
		    )
		    return next(item for item in stage_section["artifacts"] if item.get("artifact_type") == "video_final")

		final_entry = fetch_final_entry(RUN_ID)
		local_path = Path(final_entry.get("local_path", "/tmp/video_final.mp4"))
		if not local_path.exists():
		    # Fallback: download from ADK or signed URL when production_agent only published remotely.
		    adk_uri = final_entry["artifact_uri"]
		    subprocess.run(["adk", "artifacts", "download", adk_uri, str(local_path)], check=True)
	
		display(Video(filename=str(local_path), embed=True))
		files.download(str(local_path))
		```

	- When manual review is still pending, the helper should render the video
		inline **and** show the finalize banner (see step #5) so users know to log their
		`finalize_manual_review` decision before distributing the file.
	- If `/artifacts` ever returns without a `video_final` entry (e.g., finalize
		failed or was never executed), treat it as a blocking error: raise a
		toast/banner explaining that finalize has not produced a deliverable yet and
		offer a button that re-invokes production_agent with `resume_from="finalize"`
		so users can retry the stage immediately. This mirrors THE_PLAN’s guidance to
		treat missing `video_final` manifests as terminal.
	- Because Colab downloads rely on `google.colab.files.download`, keep the final
		video under the default Drive mount or `/content` so the helper can expose it
		without extra filesystem plumbing.

## Next steps (open items)

- Implement the ipywidgets prototype cell and commit sample code snippets.
- Finalize the timeline builder API (where in production_agent vs. a helper).
- Document port assignments/env vars for each FunctionTool once the UI is wired.
- Capture artifact preview patterns (inline video/audio players) after the first
	UI prototype is tested.
