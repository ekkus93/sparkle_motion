from __future__ import annotations

import pytest

from sparkle_motion.function_tools.videos_wan import adapter


@pytest.fixture(autouse=True)
def _force_fixture(monkeypatch, tmp_path):
	monkeypatch.setenv("ADK_USE_FIXTURE", "1")
	monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))


def test_render_clip_fixture_creates_mp4(tmp_path):
	result = adapter.render_clip(
		prompt="Test prompt",
		num_frames=16,
		fps=8,
		width=320,
		height=240,
		seed=42,
		metadata={"shot": "A1"},
	)

	assert result.engine == "wan_fixture"
	assert result.frame_count == 16
	assert result.path.exists()
	assert result.metadata["shot"] == "A1"
	header = result.path.read_bytes()[:16]
	assert b"ftyp" in header


def test_render_clip_validates_dimensions():
	with pytest.raises(ValueError):
		adapter.render_clip(prompt="bad", num_frames=0, fps=8, width=320, height=240)


def test_should_use_real_engine_respects_fixture_env(monkeypatch):
	monkeypatch.setenv("ADK_USE_FIXTURE", "1")
	assert not adapter.should_use_real_engine(env={"ADK_USE_FIXTURE": "1"})
	assert adapter.should_use_real_engine(env={"SMOKE_VIDEOS": "1"})
