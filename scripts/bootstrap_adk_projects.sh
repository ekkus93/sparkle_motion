#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-"resources/adk_projects.json"}
ENVIRONMENT=${1:-}

usage() {
  cat <<'EOF'
Usage: bootstrap_adk_projects.sh <environment>
Environments are defined inside resources/adk_projects.json (e.g., staging, production).
Set CONFIG_PATH to override the manifest location.
EOF
}

if [[ -z "${ENVIRONMENT}" ]]; then
  usage
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

read_config() {
  python3 - "$CONFIG_PATH" "$ENVIRONMENT" <<'PY'
import json
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
environment = sys.argv[2]
data = json.loads(config_path.read_text(encoding="utf-8"))
projects = data.get("projects", [])
project = next((p for p in projects if p.get("environment") == environment), None)
if project is None:
    raise SystemExit(f"Environment '{environment}' not found in {config_path}")

required_roles = data.get("required_roles", {})

def emit(key, value):
    if isinstance(value, str):
        print(f"{key}={shlex.quote(value)}")
    else:
        print(f"{key}={shlex.quote(json.dumps(value))}")

emit("ADK_PROJECT_ID", project["adk_project_id"])
emit("GCP_PROJECT_ID", project["gcp_project_id"])
emit("REGION", project["region"])
emit("ARTIFACT_BUCKET", project["artifact_bucket"])
emit("ARTIFACT_LOCATION", project.get("artifact_location", "US"))
session = project["session"]
emit("SESSION_BACKEND", session["backend"])
emit("SESSION_LOCATION", session["location"])
memory = project["memory"]
emit("MEMORY_BACKEND", memory["backend"])
emit("MEMORY_INSTANCE", memory["instance"])
emit("MEMORY_DATABASE", memory["database"])
service_accounts = project["service_accounts"]
emit("SA_WORKFLOW", service_accounts["workflow"])
emit("SA_TOOLS", service_accounts["tool_runtime"])
emit("SA_OPERATOR", service_accounts["operator"])
for role_group, roles in required_roles.items():
    emit(f"ROLES_{role_group.upper()}", roles)
PY
}

# shellcheck disable=SC2046
eval $(read_config)

print_header() {
  echo
  echo "=== $* ==="
}

run() {
  echo "+ $*"
  "$@"
}

ensure_gcp_project() {
  if ! gcloud projects describe "${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    run gcloud projects create "${GCP_PROJECT_ID}" --name="${GCP_PROJECT_ID}"
  fi
}

enable_services() {
  local services=(
    artifactregistry.googleapis.com
    cloudresourcemanager.googleapis.com
    cloudkms.googleapis.com
    firestore.googleapis.com
    iam.googleapis.com
    logging.googleapis.com
    secretmanager.googleapis.com
    serviceusage.googleapis.com
    spanner.googleapis.com
    storage.googleapis.com
  )
  for svc in "${services[@]}"; do
    run gcloud services enable "${svc}" --project="${GCP_PROJECT_ID}"
  done
}

ensure_bucket() {
  if ! gcloud storage buckets describe "gs://${ARTIFACT_BUCKET}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    run gcloud storage buckets create "gs://${ARTIFACT_BUCKET}" \
      --project="${GCP_PROJECT_ID}" \
      --location="${ARTIFACT_LOCATION}" \
      --uniform-bucket-level-access
  fi
}

ensure_firestore() {
  if ! gcloud firestore databases describe "(default)" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    run gcloud firestore databases create "(default)" \
      --project="${GCP_PROJECT_ID}" \
      --location="${SESSION_LOCATION}" \
      --type="${SESSION_BACKEND}"
  fi
}

ensure_spanner() {
  if ! gcloud spanner instances describe "${MEMORY_INSTANCE}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    run gcloud spanner instances create "${MEMORY_INSTANCE}" \
      --project="${GCP_PROJECT_ID}" \
      --config="regional-${REGION}" \
      --description="Sparkle Motion Memory (${ENVIRONMENT})" \
      --nodes=1
  fi
  if ! gcloud spanner databases describe "${MEMORY_DATABASE}" --instance="${MEMORY_INSTANCE}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    run gcloud spanner databases create "${MEMORY_DATABASE}" \
      --instance="${MEMORY_INSTANCE}" \
      --project="${GCP_PROJECT_ID}"
  fi
}

ensure_service_account() {
  local sa_id=$1
  local display=$2
  local email="${sa_id}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "${email}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    run gcloud iam service-accounts create "${sa_id}" \
      --project="${GCP_PROJECT_ID}" \
      --display-name="${display}"
  fi
}

bind_roles() {
  local email=$1
  local roles_json=$2
  while read -r cmd; do
    run bash -c "$cmd"
  done < <(python3 - "$email" "$roles_json" "$GCP_PROJECT_ID" <<'PY'
import json
import sys
email = sys.argv[1]
roles = json.loads(sys.argv[2])
project = sys.argv[3]
for role in roles:
    print(f"gcloud projects add-iam-policy-binding {project} --member=serviceAccount:{email} --role={role}")
PY
  )
}

ensure_secrets() {
  local secrets=(workflow-agent-service-account operator-oauth-client)
  for secret in "${secrets[@]}"; do
    if ! gcloud secrets describe "${secret}" --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
      run gcloud secrets create "${secret}" \
        --project="${GCP_PROJECT_ID}" \
        --replication-policy="automatic"
    fi
  done
}

configure_adk() {
  if ! adk projects describe "${ADK_PROJECT_ID}" >/dev/null 2>&1; then
    run adk projects create "${ADK_PROJECT_ID}" \
      --gcp-project "${GCP_PROJECT_ID}" \
      --region "${REGION}"
  fi
  run adk artifacts buckets register "${ADK_PROJECT_ID}" \
    --bucket "gs://${ARTIFACT_BUCKET}" \
    --default
  run adk services configure session \
    --project "${ADK_PROJECT_ID}" \
    --backend "${SESSION_BACKEND}" \
    --location "${SESSION_LOCATION}"
  run adk services configure memory \
    --project "${ADK_PROJECT_ID}" \
    --backend "${MEMORY_BACKEND}" \
    --instance "${MEMORY_INSTANCE}" \
    --database "${MEMORY_DATABASE}"
}

print_header "Ensuring GCP project ${GCP_PROJECT_ID}"
ensure_gcp_project

enable_services
ensure_bucket
ensure_firestore
ensure_spanner

print_header "Service accounts"
ensure_service_account "${SA_WORKFLOW}" "Workflow Agent (${ENVIRONMENT})"
ensure_service_account "${SA_TOOLS}" "Tool Runtime (${ENVIRONMENT})"
ensure_service_account "${SA_OPERATOR}" "Operator CLI (${ENVIRONMENT})"

bind_roles "${SA_WORKFLOW}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" "${ROLES_WORKFLOW}"
bind_roles "${SA_TOOLS}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" "${ROLES_TOOL_RUNTIME}"
bind_roles "${SA_OPERATOR}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" "${ROLES_OPERATOR}"

ensure_secrets
configure_adk

cat <<EOF

Bootstrap complete for environment '${ENVIRONMENT}'.
Next steps:
  - Upload secret payloads via: gcloud secrets versions add <secret> --data-file <file>
  - Register ToolRuntimes once containers exist (TODO item 3).
  - Grant human operators access to ${SA_OPERATOR}@${GCP_PROJECT_ID} via workload identity federation if desired.
EOF
