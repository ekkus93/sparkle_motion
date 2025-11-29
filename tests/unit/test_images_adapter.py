from __future__ import annotations

from pathlib import Path

import pytest

from sparkle_motion.function_tools.images_sdxl import adapter


def test_should_use_real_engine_prefers_fixture_flag() -> None:
    env = {"ADK_USE_FIXTURE": "1", "SMOKE_IMAGES": "1", "SMOKE_ADAPTERS": "1"}
    assert adapter.should_use_real_engine(env) is False


def test_should_use_real_engine_respects_smoke_flags() -> None:
    env = {"SMOKE_IMAGES": "true"}
    assert adapter.should_use_real_engine(env) is True


def test_render_images_rejects_non_positive_dimensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    with pytest.raises(ValueError):
        adapter.render_images("negative-width", {"count": 1, "width": 0, "height": 64}, output_dir=tmp_path)
    with pytest.raises(ValueError):
        adapter.render_images("negative-height", {"count": 1, "width": 64, "height": -1}, output_dir=tmp_path)


def test_render_images_fixture_metadata_contains_expected_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    results = adapter.render_images(
        "forest",
        {"count": 1, "width": 32, "height": 32, "seed": 123, "cfg_scale": 8.0, "negative_prompt": "night"},
        output_dir=tmp_path,
    )
    assert len(results) == 1
    result = results[0]
    meta = result.metadata
    assert meta["engine"] == "fixture"
    assert meta["seed"] == 123
    assert meta["width"] == 32
    assert meta["height"] == 32
    assert meta["cfg_scale"] == 8.0
    assert meta["negative_prompt"] == "night"
    assert isinstance(meta["phash"], str) and len(meta["phash"]) == 16
    assert result.path.exists()
    assert result.path.parent == tmp_path


def test_render_images_respects_batch_start_for_indices_and_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    results = adapter.render_images("city", {"count": 2, "width": 16, "height": 16, "batch_start": 4}, output_dir=tmp_path)
    indexes = [item.metadata["index"] for item in results]
    assert indexes == [4, 5]
    filenames = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".png")
    assert any(name.endswith("0004.png") for name in filenames)
    assert any(name.endswith("0005.png") for name in filenames)
