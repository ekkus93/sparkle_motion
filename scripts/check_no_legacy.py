#!/usr/bin/env python3
"""Fail the build if deprecated legacy runner artifacts reappear."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [ROOT / "src" / "sparkle_motion"]
BANNED_PATTERNS = ["ADK-style", "legacy runner", "LegacyRunner", "legacy_runner"]


def scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    hits: list[str] = []
    for pattern in BANNED_PATTERNS:
        if pattern in text:
            hits.append(f"{path.relative_to(ROOT)} contains banned pattern '{pattern}'")
    return hits


def scan() -> list[str]:
    violations: list[str] = []
    for target in TARGET_DIRS:
        if not target.exists():
            continue
        for path in target.rglob("*"):
            if path.is_dir():
                if path.name.lower().startswith("legacy"):
                    violations.append(
                        f"Disallowed legacy directory detected: {path.relative_to(ROOT)}"
                    )
                continue
            if path.suffix not in {".py", ".md", ".txt"}:
                continue
            violations.extend(scan_file(path))
    return violations


def main() -> None:
    violations = scan()
    if violations:
        print("Legacy guardrail violations found:")
        for issue in violations:
            print(f" - {issue}")
        print("\nRemove legacy artifacts before proceeding.")
        sys.exit(1)
    print("Legacy guardrail check passed.")


if __name__ == "__main__":
    main()
