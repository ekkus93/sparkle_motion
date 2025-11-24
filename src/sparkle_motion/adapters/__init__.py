"""Adapters package for external heavy-model integrations.

Each adapter exposes a small function-based API. In this repository we
provide lightweight stubs that raise a clear error explaining required
dependencies and where to implement a real adapter.
"""

from .common import MissingDependencyError

__all__ = ["MissingDependencyError"]

