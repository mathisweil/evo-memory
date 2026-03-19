#!/usr/bin/env bash
# =============================================================================
# TPU VM Setup
#
# Run from the repo root (or setup/ dir) after cloning:
#   bash setup/setup_tpu.sh
#   bash setup/setup_tpu.sh --noclaude    # skip Claude Code install
#   bash setup/setup_tpu.sh --skip-gcs   # skip Google Cloud setup
#
# For first-time bootstrap (clone + setup), use setup_tpu_cmd.sh instead.
#
# Prerequisites:
#   TPU VM with runtime tpu-ubuntu2204-base (v4) or v2-alpha-tpuv5-lite (v5e)
#   or v2-alpha-tpuv6e (v6e). python3.10+, internet access.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${REPO_DIR}/venv"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
INSTALL_CLAUDE=true
SETUP_GCS=true
SETUP_WANDB=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --noclaude)   INSTALL_CLAUDE=false; shift ;;
        --skip-gcs)   SETUP_GCS=false; shift ;;
        --skip-wandb) SETUP_WANDB=false; shift ;;
        *)
            echo "Usage: bash setup/setup_tpu.sh [--noclaude] [--skip-gcs] [--skip-wandb]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
# shellcheck source=setup/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

echo '============================================================'
echo ' evo-memory — TPU Setup'
echo '============================================================'
echo "Repo:  ${REPO_DIR}"
echo ''

# ---------------------------------------------------------------------------
# 0. Verify TPU
# ---------------------------------------------------------------------------
echo '[0/4] Checking TPU...'
if [ -e /dev/accel0 ] || [ -d /dev/vfio ]; then
    echo '  TPU device detected.'
else
    echo '  WARNING: No TPU device found (/dev/accel0 or /dev/vfio).'
    echo '  Deps will install but training will fail without a TPU.'
fi
echo ''

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo '[1/4] System dependencies...'
sudo apt-get update -qq || echo '  (apt-get update had warnings — continuing)'
sudo apt-get install -y -qq libopenblas-dev python3.10-venv >/dev/null 2>&1
echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 2. Python venv + PyTorch XLA + project deps
# ---------------------------------------------------------------------------
echo '[2/4] Python environment...'

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    rm -rf "${VENV_DIR}"
    echo "  Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

# PyTorch + XLA (TPU build — CUDA build from requirements.txt is incompatible)
echo '  Installing PyTorch + torch_xla...'
pip install torch torch_xla[tpu] \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -q

# All other deps from requirements.txt (skip the torch/* lines and CUDA index)
echo '  Installing project dependencies from requirements.txt...'
grep -vE '^(torch|torchvision|torchaudio|--index-url|--extra-index-url|#|[[:space:]]*$)' \
    "${REPO_DIR}/requirements.txt" \
    | pip install -q -r /dev/stdin

echo "  Python: $(python --version)"
echo "  torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  XLA:    $(python -c 'import torch_xla; print(torch_xla.__version__)')"
cd "${REPO_DIR}"
python -c "from es_finetuning import ESTrainer, ESConfig; print('  imports OK')"
echo ''

# ---------------------------------------------------------------------------
# 3. HuggingFace
# ---------------------------------------------------------------------------
echo '[3/4] HuggingFace...'
check_hf_login
echo ''

# ---------------------------------------------------------------------------
# 4. Weights & Biases
# ---------------------------------------------------------------------------
if [ "${SETUP_WANDB}" = true ]; then
    echo '[4/4] Weights & Biases...'
    setup_wandb
else
    echo '[4/4] Skipping wandb (--skip-wandb).'
fi
echo ''

# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------
if [ "${SETUP_GCS}" = true ]; then
    echo '[+] Google Cloud Storage...'
    setup_gcs
    echo ''
fi

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
echo ' TPU Setup complete!'
echo '============================================================'
echo ''
echo "Repo:  ${REPO_DIR}"
echo "venv:  ${VENV_DIR}"
echo ''
echo 'Activate in new shells:'
echo '  source setup/activate_tpu.sh'
echo ''
echo 'TPU smoke test:'
echo '  export PJRT_DEVICE=TPU'
echo "  python -c \"import torch_xla.core.xla_model as xm; print('TPU:', xm.xla_device())\""
echo ''
echo 'Experiment smoke test:'
echo '  source setup/activate_tpu.sh'
echo '  python scripts/run_es.py --run_name tpu_smoke --num_iterations 2 \'
echo '    --population_size 2 --mini_batch_size 2 --no-gcs'
echo ''
