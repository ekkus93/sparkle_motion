from __future__ import annotations

from pathlib import Path

from sparkle_motion.adapters import get_stub_adapter, StubAssetRef


def test_generate_images_basic():
    adapter = get_stub_adapter()
    imgs = adapter.generate_images("a test prompt", count=2)
    assert isinstance(imgs, list)
    assert len(imgs) == 2
    for a in imgs:
        assert isinstance(a, StubAssetRef)
        # PNG signature
        assert a.data.startswith(b"\x89PNG")
        assert a.meta.get("prompt") == "a test prompt"


def test_tts_roundtrip():
    adapter = get_stub_adapter()
    out = adapter.tts("hello world")
    assert out == b"hello world"
    assert out.decode("utf-8") == "hello world"


def test_lipsync_metadata():
    adapter = get_stub_adapter()
    meta = adapter.lipsync(b"audio-bytes", video_ref="video123")
    assert meta["status"] == "ok"
    assert meta["audio_len"] == len(b"audio-bytes")
    assert meta["video_ref"] == "video123"


def test_assemble_creates_file(tmp_path: Path):
    adapter = get_stub_adapter()
    res = adapter.assemble([], out_path=tmp_path / "out.mp4")
    out_path = res.path
    assert out_path.exists()
    assert isinstance(res.audio_included, bool)
    # stub now creates a non-empty placeholder file
    assert out_path.stat().st_size > 0
    # minimal MP4 header should include the 'ftyp' box
    data = out_path.read_bytes()
    assert b"ftyp" in data[:32]


def test_assemble_add_audio_flag_false(tmp_path: Path):
    adapter = get_stub_adapter()
    res = adapter.assemble([], out_path=tmp_path / "no_audio.mp4", add_audio=False)
    out_path = res.path
    audio_included = res.audio_included
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert audio_included is False
