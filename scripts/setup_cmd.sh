#!/usr/bin/env bash
# Run this on the GPU VM to set everything up.
#
# The one-liner clones the repo first, then runs setup.sh from the local
# copy — so you always execute the latest version (no stale curl cache).
#
# Works from any shell (bash, zsh, csh, tcsh — doesn't matter).
#
# Auto-detect first GPU (uses whoami for username):
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/scripts/setup_cmd.sh -o /tmp/setup_cmd.sh && bash /tmp/setup_cmd.sh
#
# Explicit username + GPU:
#   bash /tmp/setup_cmd.sh --user jsmith --gpu 2
#
# Skip Claude Code install:
#   bash /tmp/setup_cmd.sh --noclaude

set -euo pipefail

# Parse --user from args (need it before forwarding to setup.sh)
USER_NAME=""
for arg in "$@"; do
    if [ "${prev:-}" = "--user" ]; then
        USER_NAME="$arg"
    fi
    prev="$arg"
done
USER_NAME="${USER_NAME:-$(whoami)}"

WORK_DIR="/cs/student/project_msc/2025/csml/${USER_NAME}/SNLP/FT-NAMM"
REPO_DIR="${WORK_DIR}/evo-memory"
BRANCH="es-fine-tuning"
REPO_URL="https://github.com/mathisweil/evo-memory.git"

mkdir -p "${WORK_DIR}"

# Clone or update the repo so we have the latest setup.sh locally
if [ -d "${REPO_DIR}" ]; then
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    git clone -b "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

# Now run the real setup.sh from the local clone (never stale)
exec bash "${REPO_DIR}/scripts/setup.sh" "$@"
