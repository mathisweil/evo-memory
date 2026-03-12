#!/usr/bin/env bash
# Run this on the GPU VM to set everything up.
#
# The one-liner clones the repo first, then runs setup.sh from the local
# copy — so you always execute the latest version (no stale curl cache).
#
# Works from any shell (bash, zsh, csh, tcsh — doesn't matter).
#
# Step 1: Download (once)
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/setup/setup_cmd.sh -o /tmp/setup_cmd.sh
#
# Step 2: Run
#   bash /tmp/setup_cmd.sh                        # UCL VM, uses $(whoami)
#   bash /tmp/setup_cmd.sh --user jsmith --gpu 2  # UCL VM, explicit user + GPU
#   bash /tmp/setup_cmd.sh --dir ~/ft-namm        # any machine, custom dir
#   bash /tmp/setup_cmd.sh --branch my-feature     # use a different branch
#   bash /tmp/setup_cmd.sh --noclaude             # skip Claude Code install

set -euo pipefail

# Parse --user, --dir, --branch from args (need them before forwarding to setup.sh)
USER_NAME=""
CUSTOM_DIR=""
CUSTOM_BRANCH=""

for arg in "$@"; do
    if [ "${prev:-}" = "--user" ]; then
        USER_NAME="$arg"
    elif [ "${prev:-}" = "--dir" ]; then
        CUSTOM_DIR="$arg"
    elif [ "${prev:-}" = "--branch" ]; then
        CUSTOM_BRANCH="$arg"
    fi
    prev="$arg"
done

USER_NAME="${USER_NAME:-$(whoami)}"

BASE_DIR="/cs/student/project_msc/2025"

# Dynamically detect the degree folder
DEGREE_DIR=$(find "$BASE_DIR" -maxdepth 1 -type d -name "*${USER_NAME}*" -prune -o -type d -print | \
             xargs -I{} find {} -maxdepth 1 -type d -name "$USER_NAME" 2>/dev/null | \
             head -n 1 | awk -F'/' '{print $(NF-1)}')

if [ -z "$DEGREE_DIR" ]; then
    echo "Could not detect degree directory"
    exit 1
fi

if [ -n "${CUSTOM_DIR}" ]; then
    WORK_DIR="${CUSTOM_DIR}"
else
    WORK_DIR="${BASE_DIR}/${DEGREE_DIR}/${USER_NAME}/SNLP/FT-NAMM"
fi

REPO_DIR="${WORK_DIR}/evo-memory"
BRANCH="${CUSTOM_BRANCH:-es-fine-tuning}"
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

# Forward all args except --branch to setup.sh
SETUP_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch) shift 2 ;;
        --branch=*) shift ;;
        *) SETUP_ARGS+=("$1"); shift ;;
    esac
done
exec bash "${REPO_DIR}/setup/setup.sh" "${SETUP_ARGS[@]+"${SETUP_ARGS[@]}"}"
