#!/usr/bin/env bash
# =============================================================================
# evo-memory — Bootstrap
#
# Clones the repo then runs setup.sh — use this for first-time machine setup.
#
# Download and run:
#   curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/main/setup/setup_cmd.sh \
#       -o /tmp/setup_cmd.sh
#
#   bash /tmp/setup_cmd.sh                    # auto-detect hardware
#   bash /tmp/setup_cmd.sh --tpu             # Google Cloud TPU VM
#   bash /tmp/setup_cmd.sh --gpu             # CUDA GPU VM
#   bash /tmp/setup_cmd.sh --local           # local / CPU-only
#
# Additional options:
#   --dir PATH     clone into PATH/evo-memory (default: ~/evo-memory-workspace)
#   --branch NAME  checkout this branch (default: main)
#   --noclaude     skip Claude Code install
#   --skip-gcs     skip Google Cloud Storage setup
#   --skip-wandb   skip Weights & Biases setup
#
# UCL GPU machines (csh shell — use backticks for command substitution):
#   bash /tmp/setup_cmd.sh --dir /cs/student/project_msc/2025/dsml/`whoami` --skip-gcs
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/mathisweil/evo-memory.git"
DEFAULT_BRANCH="main"

# ---------------------------------------------------------------------------
# Parse arguments — extract --dir and --branch; forward everything else
# ---------------------------------------------------------------------------
CUSTOM_DIR=""
BRANCH="${DEFAULT_BRANCH}"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)      CUSTOM_DIR="$2"; shift 2 ;;
        --dir=*)    CUSTOM_DIR="${1#*=}"; shift ;;
        --branch)   BRANCH="$2"; shift 2 ;;
        --branch=*) BRANCH="${1#*=}"; shift ;;
        *)          FORWARD_ARGS+=("$1"); shift ;;
    esac
done

WORK_DIR="${CUSTOM_DIR:-${HOME}/evo-memory-workspace}"
REPO_DIR="${WORK_DIR}/evo-memory"

echo '============================================================'
echo ' evo-memory — Bootstrap'
echo '============================================================'
echo "Workspace: ${WORK_DIR}"
echo "Branch:    ${BRANCH}"
echo ''

# ---------------------------------------------------------------------------
# Clone or update
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}"

if [ -d "${REPO_DIR}" ]; then
    echo 'Repo already exists — pulling latest...'
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
# Delegate to setup.sh (all remaining args forwarded)
# ---------------------------------------------------------------------------
exec bash "${REPO_DIR}/setup/setup.sh" "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"
