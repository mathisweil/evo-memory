#!/usr/bin/env bash
# =============================================================================
# TPU VM Bootstrap: clone repo + run setup_tpu.sh
#
# Run this on your TPU VM (SSH or VSCode Remote) to get everything set up.
# Clones the repo, then hands off to setup_tpu.sh for venv + deps.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/setup/setup_tpu_cmd.sh -o /tmp/setup_tpu_cmd.sh
#   bash /tmp/setup_tpu_cmd.sh
#   bash /tmp/setup_tpu_cmd.sh --noclaude             # skip Claude Code
#   bash /tmp/setup_tpu_cmd.sh --dir ~/my-workspace   # custom directory
#   bash /tmp/setup_tpu_cmd.sh --branch my-feature    # different branch
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CUSTOM_DIR=""
CUSTOM_BRANCH=""
SETUP_ARGS=()

for arg in "$@"; do
    if [ "${prev:-}" = "--dir" ]; then
        CUSTOM_DIR="$arg"
    elif [ "${prev:-}" = "--branch" ]; then
        CUSTOM_BRANCH="$arg"
    fi
    prev="$arg"
done

WORK_DIR="${CUSTOM_DIR:-$HOME/FT-NAMM}"
BRANCH="${CUSTOM_BRANCH:-es-fine-tuning}"
REPO_URL="https://github.com/mathisweil/evo-memory.git"
REPO_DIR="${WORK_DIR}/evo-memory"

echo '============================================================'
echo ' TPU VM Bootstrap'
echo '============================================================'
echo ''
echo "Workspace: ${WORK_DIR}"
echo "Branch:    ${BRANCH}"
echo ''

# ---------------------------------------------------------------------------
# Clone or update repo
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}"

if [ -d "${REPO_DIR}" ]; then
    echo 'Repo already exists, pulling latest...'
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    echo 'Cloning repo...'
    git clone -b "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

echo ''

# ---------------------------------------------------------------------------
# Forward to setup_tpu.sh (strip --dir and --branch from args)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) shift 2 ;;
        --dir=*) shift ;;
        --branch) shift 2 ;;
        --branch=*) shift ;;
        *) SETUP_ARGS+=("$1"); shift ;;
    esac
done

exec bash "${REPO_DIR}/setup/setup_tpu.sh" "${SETUP_ARGS[@]+"${SETUP_ARGS[@]}"}"
