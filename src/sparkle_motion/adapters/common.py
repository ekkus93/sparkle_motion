"""Common helpers for adapters."""
from __future__ import annotations

class MissingDependencyError(RuntimeError):
    """Raised by adapter stubs when required libraries are not installed.

    The message should tell the user which packages are required and where to
    find implementation notes.
    """

    pass
