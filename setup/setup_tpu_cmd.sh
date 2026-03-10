#!/usr/bin/env bash
# =============================================================================
# TPU VM Bootstrap: clone repo + run setup_tpu.sh
#
# Run this on a freshly provisioned TPU VM to get everything set up.
#
# Step 1: Create your TPU VM (from your local machine):
#
#   # On-demand v4-32 (recommended for development — no preemption)
#   gcloud compute tpus tpu-vm create es-finetune \
#       --zone=us-central2-b \
#       --accelerator-type=v4-32 \
#       --version=tpu-ubuntu2204-base
#
#   # SSH into it
#   gcloud compute tpus tpu-vm ssh es-finetune --zone=us-central2-b
#
# Step 2: On the TPU VM, download and run this script:
#
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
