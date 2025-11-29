#!/usr/bin/env python3
"""Convenience wrapper for the recent-index inspection CLI.

This script simply delegates to ``sparkle_motion.utils.recent_index_cli`` so
operators can run `python scripts/recent_index.py ...` alongside the other
toolkit commands without remembering the module path.
"""

from __future__ import annotations

from sparkle_motion.utils.recent_index_cli import main as recent_index_main


if __name__ == "__main__":
    raise SystemExit(recent_index_main())
