#!/usr/bin/env bash
# =============================================================================
# GPU VM Setup
#
# Run from the repo root (or setup/ dir) after cloning:
#   bash setup/setup.sh
#   bash setup/setup.sh --gpu 1          # pin to a specific GPU
#   bash setup/setup.sh --noclaude       # skip Claude Code install
#   bash setup/setup.sh --skip-gcs       # skip Google Cloud setup
#   bash setup/setup.sh --skip-wandb     # skip wandb setup
#
# For first-time bootstrap (clone + setup), use setup_cmd.sh instead.
#
# Prerequisites: python3.9+, CUDA 12.1+ GPU drivers, internet access.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
GPU_ID=""
INSTALL_CLAUDE=true
SETUP_GCS=true
SETUP_WANDB=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU_ID="$2"; shift 2 ;;
        --gpu=*)      GPU_ID="${1#*=}"; shift ;;
        --noclaude)   INSTALL_CLAUDE=false; shift ;;
        --skip-gcs)   SETUP_GCS=false; shift ;;
        --skip-wandb) SETUP_WANDB=false; shift ;;
        *)
            echo "Usage: bash setup/setup.sh [--gpu N] [--noclaude] [--skip-gcs] [--skip-wandb]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
# shellcheck source=setup/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------
if [ -n "${GPU_ID}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if command -v nvidia-smi &>/dev/null; then
        GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1 | tr -d ' ')
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    fi
fi

echo '============================================================'
echo ' evo-memory — GPU Setup'
echo '============================================================'
echo "Repo:  ${REPO_DIR}"
echo "GPU:   ${CUDA_VISIBLE_DEVICES:-not set}"
echo ''

# ---------------------------------------------------------------------------
# 1. Python venv + dependencies
# ---------------------------------------------------------------------------
echo '[1/4] Installing Python dependencies...'
cd "${REPO_DIR}"
source "${SCRIPT_DIR}/activate.sh"
python -c "from es_finetuning import ESTrainer, ESConfig; print('  imports OK')"
echo "  Python: $(python --version)"
echo ''

# ---------------------------------------------------------------------------
# 2. HuggingFace
# ---------------------------------------------------------------------------
echo '[2/4] HuggingFace...'
check_hf_login
echo ''

# ---------------------------------------------------------------------------
# 3. Weights & Biases
# ---------------------------------------------------------------------------
if [ "${SETUP_WANDB}" = true ]; then
    echo '[3/4] Weights & Biases...'
    setup_wandb
else
    echo '[3/4] Skipping wandb (--skip-wandb).'
fi
echo ''

# ---------------------------------------------------------------------------
# 4. Google Cloud Storage
# ---------------------------------------------------------------------------
if [ "${SETUP_GCS}" = true ]; then
    echo '[4/4] Google Cloud Storage...'
    setup_gcs
else
    echo '[4/4] Skipping GCS (--skip-gcs).'
fi
echo ''

# ---------------------------------------------------------------------------
# Claude Code (optional)
# ---------------------------------------------------------------------------
if [ "${INSTALL_CLAUDE}" = true ]; then
    echo '[+] Claude Code...'
    install_claude
    echo ''
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo '============================================================'
echo ' Setup complete!'
echo '============================================================'
echo ''
echo "Repo:  ${REPO_DIR}"
echo "venv:  ${REPO_DIR}/venv"
echo ''
echo 'Activate in new shells:'
echo '  source setup/activate.sh'
echo ''
echo 'Smoke test:'
echo '  python scripts/run_es.py --run_name smoke --num_iterations 2 \'
echo '    --population_size 2 --mini_batch_size 2 --no-gcs'
echo ''
