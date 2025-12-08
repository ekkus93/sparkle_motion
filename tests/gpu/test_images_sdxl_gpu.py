from __future__ import annotations

from pathlib import Path

import pytest

from sparkle_motion.function_tools.images_sdxl import adapter as sdxl_adapter

from . import helpers


@pytest.mark.gpu
def test_sdxl_render_single_image_real(monkeypatch: "pytest.MonkeyPatch", tmp_path: Path) -> None:
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

    output_dir = tmp_path / "sdxl_images"
    with helpers.temp_output_dir(monkeypatch, "IMAGES_SDXL_OUTPUT_DIR", output_dir):
        results = sdxl_adapter.render_images(
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
