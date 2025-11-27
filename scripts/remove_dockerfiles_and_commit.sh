#!/usr/bin/env bash
set -euo pipefail

# This script removes Dockerfiles from function_tools, commits the change,
# and pushes a branch to origin. Run from the repo root.

FILES=(
  "function_tools/script_agent/Dockerfile"
  "function_tools/images_sdxl/Dockerfile"
  "function_tools/videos_wan/Dockerfile"
)

COMMIT_MSG="chore: remove Dockerfiles (project no longer uses Docker containers)"

echo "Removing files on current branch $(git rev-parse --abbrev-ref HEAD)"
for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    git rm "$f"
    echo "  scheduled $f for removal"
  else
    echo "  $f not found; skipping"
  fi
done

echo "Committing"
git commit -m "$COMMIT_MSG" || echo "Nothing to commit"

echo "Running tests"
PYTHONPATH=src python -m pytest -q

echo "If tests passed, push the current branch with:"
echo "  git push origin HEAD"

echo "Done."
