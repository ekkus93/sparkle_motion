"""Render and push a PromptTemplate JSON to ADK.

This script prefers the `adk` CLI (recommended in Colab / local-colab).
If the CLI is not available it prints a helpful error explaining how to
install/configure the ADK tooling in the active environment.

Usage:
  # render and push default script_agent template
  PYTHONPATH=src python scripts/push_prompt_template.py

  # push an existing JSON file
  PYTHONPATH=src python scripts/push_prompt_template.py --file artifacts/prompt_templates/script_agent_movie_plan_v1.json

Note: This helper intentionally shells out to the `adk` CLI to avoid
inventing ADK SDK usage patterns in this repo; if you prefer an SDK-based
integration I can add a guarded implementation that imports `google.adk`.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

from sparkle_motion.prompt_templates import render_script_agent_prompt_template
from sparkle_motion.adk_helpers import probe_sdk, run_adk_cli


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", type=Path, help="Path to existing PromptTemplate JSON to push")
    p.add_argument("--no-render", action="store_true", help="Do not render a new file when --file is omitted")
    return p.parse_args()


def ensure_adk_cli_available() -> bool:
    return shutil.which("adk") is not None


def push_with_adk_cli(file_path: Path) -> int:
    cmd = ["adk", "llm-prompts", "push", "--file", str(file_path)]
    print("Running:", " ".join(cmd))
    # Use central runner to make tests easier to patch/match
    rc, out, err = run_adk_cli(cmd, dry_run=False)
    if out:
        print(out)
    if err:
        print(err)
    return rc


def push_with_sdk(file_path: Path) -> tuple[bool, str]:
    """Attempt to push using the `google.adk` Python SDK.

    This is a best-effort helper: ADK SDK entrypoints may vary across
    versions. The function tries a few plausible attributes and methods and
    returns (success, message). It never raises import errors to keep callers
    able to fall back to the CLI.
    """
    sdk_probe = probe_sdk()
    if not sdk_probe:
        return False, "google.adk not importable"
    adk = sdk_probe[0]

    # Try a small set of plausible client entrypoints and method names.
    candidates = ["llm_prompts", "prompts", "llm_prompts_client", "llm"]
    methods = ["push", "create", "upload", "register"]

    for cand in candidates:
        client = getattr(adk, cand, None)
        if client is None:
            continue
        for method in methods:
            fn = getattr(client, method, None)
            if not fn:
                continue
            try:
                # Some SDK methods accept a path, others accept file-like or dict.
                # We call with the file path first and let the SDK raise if wrong.
                result = fn(str(file_path))
                return True, f"Pushed via google.adk.{cand}.{method} -> {result}"
            except Exception as e:  # pragma: no cover - depends on installed SDK
                return False, f"google.adk.{cand}.{method} raised: {e}"

    return False, "google.adk imported but no known push API found; check your SDK version"


def main() -> int:
    args = parse_args()
    file_path: Optional[Path] = None

    if args.file:
        file_path = args.file
        if not file_path.exists():
            print(f"Specified file does not exist: {file_path}")
            return 2
    else:
        if args.no_render:
            print("No input file provided and --no-render was set. Nothing to do.")
            return 0
        # render the default ScriptAgent prompt template
        file_path = render_script_agent_prompt_template()
        print(f"Rendered prompt template to {file_path}")

    if not ensure_adk_cli_available():
        print("The 'adk' CLI was not found in PATH. To push a prompt template you can:")
        print("  1) Install the ADK: pip install google-adk")
        print("  2) Ensure the 'adk' CLI is available in PATH (restart your shell")
        print("  3) Or push the JSON via your preferred ADK SDK integration")
        return 3

    rc = push_with_adk_cli(file_path)
    if rc != 0:
        print(f"adk CLI returned non-zero exit code: {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
