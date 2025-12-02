#!/usr/bin/env python3
"""Convenience wrapper for filesystem artifact utilities CLI."""

from __future__ import annotations

from sparkle_motion.filesystem_artifacts.cli import main as filesystem_cli_main


if __name__ == "__main__":
    raise SystemExit(filesystem_cli_main())
