from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TTSAdapter:
    """Scaffold adapter for TTS (Coqui TTS / Bark / ElevenLabs).

    Real implementations should use `TTS.api.TTS` or vendor SDKs. This
    scaffold writes a tiny WAV placeholder for unit tests.
    """
    model_name: Optional[str] = None

    def synthesize(self, text: str, out_path: Path, voice_config: Optional[dict] = None) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a tiny WAV-like placeholder (not a real WAV file). Tests should
        # only verify the presence/metadata wrapper, not audio fidelity.
        out_path.write_bytes(b"RIFFFAKEWAVE")
        return out_path
