from __future__ import annotations

from sparkle_motion.function_tools.images_sdxl import adapter


def test_fixture_render_is_deterministic(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    opts = {"count": 2, "seed": 42, "width": 64, "height": 64}
    first = adapter.render_images("fixture", opts, output_dir=tmp_path)
    second = adapter.render_images("fixture", opts, output_dir=tmp_path)

    assert len(first) == len(second) == 2
    assert first[0].data[:8] == b"\x89PNG\r\n\x1a\n"
    assert first[0].data == second[0].data
    assert first[1].metadata["phash"] == second[1].metadata["phash"]
    for result in first:
        assert result.path.exists()


def test_fixture_metadata_merges(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADK_USE_FIXTURE", "1")
    extra_meta = {"plan_id": "plan-123", "engine": "override"}
    results = adapter.render_images(
        "meta",
        {"count": 1, "metadata": extra_meta, "width": 64, "height": 64},
        output_dir=tmp_path,
    )
    assert results[0].metadata["plan_id"] == "plan-123"
    assert results[0].metadata["engine"] == "fixture"