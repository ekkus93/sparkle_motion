"""Compatibility entrypoint module for legacy imports.

This file intentionally re-exports the canonical implementation from
`sparkle_motion.function_tools.script_agent.entrypoint` so that code and
tests referencing the old `ScriptAgent.entrypoint` path continue to work.
"""
from __future__ import annotations

from sparkle_motion.function_tools.script_agent.entrypoint import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
