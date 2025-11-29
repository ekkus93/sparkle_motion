#!/usr/bin/env python3
"""Compatibility wrapper for the new module-based publish_schemas entrypoint."""
from __future__ import annotations

from sparkle_motion.scripts.publish_schemas import main


if __name__ == "__main__":
    raise SystemExit(main())
