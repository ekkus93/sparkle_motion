#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="local-colab"

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_adk_projects.sh [--profile NAME]

Only the local-colab profile is currently supported. Run this script from the
repository root inside a Colab runtime after mounting Google Drive.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export BOOTSTRAP_PROFILE="$PROFILE"

python3 <<'PY'
import json
import os
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parents[1]
manifest_path = repo_root / "resources" / "adk_projects.json"
profile_name = os.environ["BOOTSTRAP_PROFILE"]

try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    print(f"Manifest not found at {manifest_path}", file=sys.stderr)
    sys.exit(1)

profiles = data.get("profiles", [])
profile = next((p for p in profiles if p.get("name") == profile_name), None)
if profile is None:
    print(f"Profile '{profile_name}' not found in {manifest_path}", file=sys.stderr)
    sys.exit(1)

summary = []

artifact_config = profile.get("artifact_service", {})
artifact_path = pathlib.Path(artifact_config.get("path", ""))
requires_mount = artifact_config.get("requires_mount", False)
if requires_mount and not pathlib.Path("/content/drive").exists():
    print("[WARN] /content/drive is missing. Mount Google Drive before running the bootstrap script.")

session_config = profile.get("session_service", {})
memory_config = profile.get("memory_service", {})
secrets_config = profile.get("secrets", {})
service_accounts = profile.get("service_accounts", [])

sqlite_targets = [
    (session_config, "SessionService"),
    (memory_config, "MemoryService"),
]

for config, label in sqlite_targets:
    path_value = config.get("path")
    if not path_value:
        continue
    path = pathlib.Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
        summary.append(f"Created {label} SQLite database at {path}")
    else:
        summary.append(f"{label} SQLite database already exists at {path}")

if artifact_path:
    artifact_path.mkdir(parents=True, exist_ok=True)
    summary.append(f"Ensured artifact root exists at {artifact_path}")

secrets_path_value = secrets_config.get("path")
if secrets_path_value:
    secrets_path = pathlib.Path(secrets_path_value)
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    if not secrets_path.exists():
        env_lines = [
            "# Sparkle Motion local-colab secrets",
            f"ADK_PROFILE={profile_name}",
        ]
        session_path = session_config.get("path", "")
        memory_path = memory_config.get("path", "")
        artifact_str = str(artifact_path) if artifact_path else ""
        if session_path:
            env_lines.append(f"ADK_SESSION_DB={session_path}")
        if memory_path:
            env_lines.append(f"ADK_MEMORY_DB={memory_path}")
        if artifact_str:
            env_lines.append(f"ADK_ARTIFACT_ROOT={artifact_str}")
        env_lines.append(f"ADK_SECRETS_FILE={secrets_path}")
        telemetry_enabled = profile.get("telemetry", {}).get("enabled", True)
        env_lines.append(f"ADK_TELEMETRY_ENABLED={'1' if telemetry_enabled else '0'}")
        env_lines.append("ADK_API_KEY=")
        if service_accounts:
            env_lines.append("")
            env_lines.append("# Service account JSON payloads (paste path or inline JSON)")
            for sa in service_accounts:
                env_var = sa.get("env_var") or sa.get("id", "").upper()
                env_lines.append(f"{env_var}=")
        secrets_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        summary.append(f"Created secrets template at {secrets_path}")
    else:
        summary.append(f"Secrets file already exists at {secrets_path}")

telemetry_enabled = profile.get("telemetry", {}).get("enabled", True)
summary.append(f"Telemetry enabled: {telemetry_enabled}")

tools = profile.get("tools", [])
if tools:
    summary.append("Tool endpoints:")
    for tool in tools:
        summary.append(
            f"  - {tool.get('id')}: {tool.get('endpoint')} (auth={tool.get('auth', 'none')})"
        )

if service_accounts:
    summary.append("Service account env vars:")
    for sa in service_accounts:
        env_var = sa.get("env_var") or sa.get("id", "").upper()
        description = sa.get("description", "")
        summary.append(f"  - {env_var} ({description})")

print(f"Bootstrap complete for profile '{profile_name}'.")
print("\n".join(summary))
PY
