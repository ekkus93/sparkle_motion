"""Compatibility shim for legacy imports using `ScriptAgent` (case-variant).

This module re-exports the canonical `script_agent` package so existing
imports/tests that reference `sparkle_motion.function_tools.ScriptAgent`
continue to work after we normalized directories to `script_agent`.

This is intentionally minimal and safe; if you prefer removal, the shim
can be deleted and tests updated to use the canonical path.
"""
from __future__ import annotations

from sparkle_motion.function_tools.script_agent import *  # re-export public symbols

__all__ = [name for name in dir() if not name.startswith("_")]
