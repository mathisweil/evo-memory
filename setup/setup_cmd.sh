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
#   bash /tmp/setup_cmd.sh --degree csml                     # UCL VM, uses $(whoami)
#   bash /tmp/setup_cmd.sh --user jsmith --gpu 2             # UCL VM, explicit user + GPU
#   bash /tmp/setup_cmd.sh --dir ~/ft-namm                   # any machine, custom dir
#   bash /tmp/setup_cmd.sh --branch my-feature               # use a different branch
#   bash /tmp/setup_cmd.sh --noclaude                        # skip Claude Code install

set -euo pipefail

# -----------------------------
# Defaults
# -----------------------------
USER_NAME=""
CUSTOM_DIR=""
CUSTOM_BRANCH=""
DEGREE="csml"

# Arguments that will be forwarded to setup.sh
SETUP_ARGS=()

# -----------------------------
# Parse CLI arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)
            USER_NAME="$2"
            SETUP_ARGS+=("$1" "$2")
            shift 2
            ;;
        --dir)
            CUSTOM_DIR="$2"
            SETUP_ARGS+=("$1" "$2")
            shift 2
            ;;
        --branch)
            CUSTOM_BRANCH="$2"
            shift 2
            ;;
        --degree)
            DEGREE="$2"
            SETUP_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            SETUP_ARGS+=("$1")
            shift
            ;;
    esac
done

# -----------------------------
# Resolve defaults
# -----------------------------
USER_NAME="${USER_NAME:-$(whoami)}"

if [[ -n "${CUSTOM_DIR}" ]]; then
    WORK_DIR="${CUSTOM_DIR}"
else
    WORK_DIR="/cs/student/project_msc/2025/${DEGREE}/${USER_NAME}/SNLP/FT-NAMM"
fi

REPO_DIR="${WORK_DIR}/evo-memory"
BRANCH="${CUSTOM_BRANCH:-es-fine-tuning}"
REPO_URL="https://github.com/mathisweil/evo-memory.git"

mkdir -p "${WORK_DIR}"

# -----------------------------
# Clone or update repository
# -----------------------------
if [[ -d "${REPO_DIR}" ]]; then
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    git clone -b "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

# -----------------------------
# Run the real setup script
# -----------------------------
exec bash "${REPO_DIR}/setup/setup.sh" "${SETUP_ARGS[@]}"
