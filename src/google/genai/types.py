from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Part:
    """Minimal Part shim with helper constructors used by the shimbed services.

    Attributes are intentionally small: `data` (bytes or text) and `mime_type`.
    """

    data: Optional[bytes] = None
    mime_type: Optional[str] = None
    file_uri: Optional[str] = None

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str | None = None) -> "Part":
        return cls(data=data, mime_type=mime_type)

    @classmethod
    def from_uri(cls, file_uri: str) -> "Part":
        return cls(file_uri=file_uri)
