ADK Test Fixtures
=================

Purpose
-------
This directory contains lightweight, file-backed shim implementations of
the ADK `ArtifactService` used by tests. They live under
`tests/fixtures/google/adk/artifacts/` so tests can explicitly load them
without risking shadowing an installed `google` package in developer
environments.

How tests use the fixtures
--------------------------
- Tests that need the shim explicitly insert `tests/fixtures` at the
  front of `sys.path` before importing the code under test. This ensures
  that `import google.adk.artifacts...` resolves to the fixture modules.

Example (in tests):

```py
# repo-root/tests/test_...py
repo_root = Path(__file__).resolve().parents[2]
fixtures_dir = str(repo_root / "tests" / "fixtures")
if fixtures_dir not in sys.path:
    sys.path.insert(0, fixtures_dir)
    importlib.invalidate_caches()
```

CI guidance
-----------
- Do not add `tests/fixtures` to `PYTHONPATH` globally in CI jobs where
  the real ADK SDK is expected to be available; doing so will force the
  fixtures to shadow the real SDK. Only tests that intentionally use the
  shim should modify `sys.path` locally.

ADK_USE_FIXTURE
---------------
- Keep `ADK_USE_FIXTURE=1` (default) whenever you want every test and
  entrypoint to use the lightweight filesystem shim defined in this
  directory. This is the expected setting for local development and CI.
- Set `ADK_USE_FIXTURE=0` **only** when you have installed and
  authenticated the real `google.adk` SDK and want the runtime to talk to
  a live ADK control plane. When doing so, ensure `tests/fixtures` is not
  inserted ahead of the real SDK on `sys.path` and export
  `ADK_PROJECT`/`ADK_API_KEY` as required by the SDK.

Notes
-----
- The fixtures are intentionally minimal and synchronous-friendly for
  test simplicity. They are not intended to be production replacements
  for the real ADK SDK.

- Platform note: the fixture shims use POSIX advisory file locks (via
  `fcntl`) for atomic `.rev` updates and have been tested on Linux and
  macOS. If you expect contributors or CI to run on Windows, consider
  using a cross-platform library such as `portalocker` or switch the
  shims to a maintained portable implementation. The project currently
  defaults to the POSIX implementation to avoid adding a runtime
  dependency for test fixtures.

Deterministic media assets
--------------------------
- `tests/fixtures/assets/sample_image.png` — 32×32 RGB gradient used for
  image/QA tests.
- `tests/fixtures/assets/sample_audio.wav` — 0.35 s mono sine wave at
  16 kHz for audio/TTS flows.
- `tests/fixtures/assets/sample_video.mp4` — Wan adapter fixture payload
  (seeded) suitable for video/publish tests.
- `tests/fixtures/assets/sample_plan.json` — tiny MoviePlan-style JSON
  referencing the media assets above.

To regenerate the media assets run (from repo root):

```bash
PYTHONPATH=src python scripts/generate_fixture_assets.py
```
