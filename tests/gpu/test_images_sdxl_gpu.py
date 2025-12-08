from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from sparkle_motion.function_tools.images_sdxl import adapter as sdxl_adapter

from . import helpers


def _render_real_sdxl(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: Path,
    *,
    prompt: str,
    opts: Mapping[str, Any],
    subdir: str = "sdxl_images",
) -> list[sdxl_adapter.ImageRenderResult]:
    helpers.ensure_real_adapter(
        monkeypatch,
        flags=["SMOKE_ADAPTERS", "SMOKE_IMAGES"],
        disable_keys=["IMAGES_SDXL_FIXTURE_ONLY", "ADK_USE_FIXTURE"],
    )
    helpers.set_env(
        monkeypatch,
        {
            "IMAGES_SDXL_FIXTURE_ONLY": "0",
        },
    )

    output_dir = tmp_path / subdir
    with helpers.temp_output_dir(monkeypatch, "IMAGES_SDXL_OUTPUT_DIR", output_dir):
        return sdxl_adapter.render_images(prompt=prompt, opts=dict(opts))


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:4] != b"\x89PNG":
        raise AssertionError("Artifact is not a PNG file")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


@pytest.mark.gpu
def test_sdxl_render_single_image_real(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    results = _render_real_sdxl(
        monkeypatch,
        tmp_path,
        prompt="Cinematic sunrise over mountains",
        opts={
            "count": 1,
            "seed": 42,
            "steps": 20,
            "cfg_scale": 7.5,
            "width": 1024,
            "height": 1024,
        },
    )

    assert len(results) == 1, "SDXL adapter should return exactly one image"
    result = results[0]
    assert result.path.exists(), "Rendered image path should exist on disk"
    assert result.path.stat().st_size > 100 * 1024, "Real SDXL render should be larger than 100KB"

    file_header = result.path.read_bytes()[:4]
    assert file_header == b"\x89PNG", "Rendered artifact must be a valid PNG"

    metadata = result.metadata
    assert metadata.get("engine") == "sdxl", "Real engine should be reported as 'sdxl'"
    assert metadata.get("seed") == 42
    assert metadata.get("width") == 1024
    assert metadata.get("height") == 1024
    phash = metadata.get("phash")
    assert isinstance(phash, str) and phash, "Metadata must include a perceptual hash"


@pytest.mark.gpu
def test_sdxl_render_batch(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    results = _render_real_sdxl(
        monkeypatch,
        tmp_path,
        prompt="Hero portrait",
        opts={
            "count": 4,
            "batch_start": 0,
            "seed": 100,
            "width": 1024,
            "height": 1024,
        },
        subdir="sdxl_batch",
    )

    assert len(results) == 4, "Batch render must return four images"

    seeds = [res.metadata.get("seed") for res in results]
    assert seeds == [100, 101, 102, 103], "Seeds should increment sequentially per batch"

    phashes = [res.metadata.get("phash") for res in results]
    assert len({phash for phash in phashes if isinstance(phash, str)}) == len(phashes), "Each batch image must have a unique phash"

    dimensions = {(res.metadata.get("width"), res.metadata.get("height")) for res in results}
    assert len(dimensions) == 1, "All batch images should share identical dimensions"


@pytest.mark.gpu
def test_sdxl_negative_prompt(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    results = _render_real_sdxl(
        monkeypatch,
        tmp_path,
        prompt="Forest",
        opts={
            "count": 1,
            "seed": 99,
            "negative_prompt": "dark, gloomy",
        },
        subdir="sdxl_negative",
    )

    result = results[0]
    assert result.path.exists(), "Negative prompt run should persist an artifact"
    assert result.path.read_bytes()[:4] == b"\x89PNG", "Artifact must be a PNG"

    metadata = result.metadata
    assert metadata.get("prompt") == "Forest"
    assert metadata.get("negative_prompt") == "dark, gloomy"


@pytest.mark.gpu
def test_sdxl_custom_dimensions(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    width, height = 1280, 720
    results = _render_real_sdxl(
        monkeypatch,
        tmp_path,
        prompt="Wide landscape",
        opts={
            "count": 1,
            "seed": 50,
            "width": width,
            "height": height,
        },
        subdir="sdxl_custom_dims",
    )

    result = results[0]
    metadata = result.metadata
    assert metadata.get("width") == width
    assert metadata.get("height") == height

    png_width, png_height = _png_dimensions(result.path)
    assert (png_width, png_height) == (width, height), "PNG dimensions must match requested size"


@pytest.mark.gpu
def test_sdxl_determinism(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
    opts = {
        "count": 1,
        "seed": 7,
        "steps": 15,
        "cfg_scale": 7.5,
        "width": 1024,
        "height": 1024,
    }

    first = _render_real_sdxl(
        monkeypatch,
        tmp_path,
        prompt="Test consistency",
        opts=opts,
        subdir="sdxl_determinism_a",
    )[0]
    second = _render_real_sdxl(
        monkeypatch,
        tmp_path,
        prompt="Test consistency",
        opts=opts,
        subdir="sdxl_determinism_b",
    )[0]

    assert first.metadata.get("phash") == second.metadata.get("phash"), "Perceptual hashes should match for identical parameters"
    assert first.path.read_bytes() == second.path.read_bytes(), "SDXL renders should be byte-identical for the same prompt/seed"
