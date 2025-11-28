from pathlib import Path
import tempfile

from sparkle_motion.gpu_utils import model_context
from sparkle_motion.adapters.diffusers_adapter import DiffusersAdapter
from sparkle_motion.adapters.wan_adapter import WanAdapter
from sparkle_motion.adapters.wav2lip_adapter import Wav2LipAdapter
from sparkle_motion.adapters.tts_adapter import TTSAdapter


def test_model_context_basic():
    closed = {"flag": False}

    class DummyModel:
        def close(self):
            closed["flag"] = True

    def loader():
        return DummyModel()

    with model_context("sdxl", loader=loader, weights="dummy") as ctx:
        assert ctx.pipeline is not None

    assert closed["flag"] is True


def test_diffusers_adapter_renders(tmp_path: Path):
    adapter = DiffusersAdapter(weights="dummy")
    adapter.load()
    outs = adapter.render_images("a test prompt")
    assert len(outs) >= 1
    for p in outs:
        assert p.exists()
        assert p.stat().st_size > 0


def test_wan_adapter_produces_mp4(tmp_path: Path):
    adapter = WanAdapter(weights="dummy")
    adapter.load()
    out = tmp_path / "out.mp4"
    res = adapter.run(None, None, "a prompt", out)
    assert res.exists()
    assert res.stat().st_size > 0


def test_wav2lip_and_tts_adapters(tmp_path: Path):
    wav_adapter = Wav2LipAdapter()
    out1 = tmp_path / "lip.mp4"
    wav_adapter.run(Path("in.mp4"), Path("in.wav"), out1)
    assert out1.exists()

    tts = TTSAdapter()
    out2 = tmp_path / "speech.wav"
    tts.synthesize("hello world", out2)
    assert out2.exists()
