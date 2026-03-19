#!/usr/bin/env bash
# =============================================================================
# evo-memory — Unified Setup
#
# Auto-detects hardware, or specify with a flag:
#   bash setup/setup.sh              # auto-detect: TPU → GPU → local
#   bash setup/setup.sh --tpu       # Google Cloud TPU VM
#   bash setup/setup.sh --gpu       # CUDA GPU (auto-selects first GPU)
#   bash setup/setup.sh --gpu 2     # pin to a specific GPU index
#   bash setup/setup.sh --local     # local / CPU-only machine
#
# Optional flags (all modes):
#   --noclaude      skip Claude Code install
#   --skip-gcs      skip Google Cloud Storage setup
#   --skip-wandb    skip Weights & Biases setup
#
# For first-time bootstrap (clone + setup), use setup_cmd.sh instead.
# Prerequisites: python3.9+, internet access.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${REPO_DIR}/venv"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE=""         # tpu | gpu | local  (empty = auto-detect)
GPU_ID=""
INSTALL_CLAUDE=true
SETUP_GCS=true
SETUP_WANDB=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpu)        MODE=tpu; shift ;;
        --gpu)
            MODE=gpu
            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then GPU_ID="$2"; shift; fi
            shift ;;
        --gpu=*)      MODE=gpu; GPU_ID="${1#*=}"; shift ;;
        --local)      MODE=local; shift ;;
        --noclaude)   INSTALL_CLAUDE=false; shift ;;
        --skip-gcs)   SETUP_GCS=false; shift ;;
        --skip-wandb) SETUP_WANDB=false; shift ;;
        *)
            echo "Usage: bash setup/setup.sh [--tpu | --gpu [N] | --local]"
            echo "                           [--noclaude] [--skip-gcs] [--skip-wandb]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Auto-detect hardware
# ---------------------------------------------------------------------------
if [ -z "${MODE}" ]; then
    if [ -e /dev/accel0 ] || [ -d /dev/vfio ]; then
        MODE=tpu
    elif command -v nvidia-smi &>/dev/null; then
        MODE=gpu
    else
        MODE=local
    fi
    echo "Auto-detected mode: ${MODE}"
fi

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
# shellcheck source=setup/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

echo '============================================================'
echo " evo-memory — Setup (${MODE})"
echo '============================================================'
echo "Repo:  ${REPO_DIR}"
echo ''

# ===========================================================================
# TPU — system packages and PyTorch/XLA install
# ===========================================================================
if [ "${MODE}" = tpu ]; then

    echo '[0/4] Checking TPU device...'
    if [ -e /dev/accel0 ] || [ -d /dev/vfio ]; then
        echo '  TPU device detected.'
    else
        echo '  WARNING: No TPU device found (/dev/accel0 or /dev/vfio).'
        echo '  Deps will install but training will fail without a TPU.'
    fi
    echo ''

    echo '[1/4] System dependencies...'
    if sudo -n true 2>/dev/null; then
        sudo apt-get update -qq || echo '  (apt-get update had warnings — continuing)'
        sudo apt-get install -y -qq libopenblas-dev python3.10-venv >/dev/null 2>&1
        echo '  Done.'
    else
        echo '  No passwordless sudo — skipping apt-get.'
        echo '  Ensure libopenblas-dev and python3.10-venv are already installed.'
    fi
    echo ''

    echo '[2/4] Python environment (PyTorch + XLA)...'
    if [ ! -f "${VENV_DIR}/bin/activate" ]; then
        rm -rf "${VENV_DIR}"
        echo "  Creating venv at ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip -q
    echo '  Installing torch + torch_xla[tpu]...'
    pip install torch torch_xla[tpu] \
        -f https://storage.googleapis.com/libtpu-releases/index.html -q
    echo '  Installing project dependencies (requirements.txt, skip CUDA lines)...'
    grep -vE '^(torch|torchvision|torchaudio|--index-url|--extra-index-url|#|[[:space:]]*$)' \
        "${REPO_DIR}/requirements.txt" | pip install -q -r /dev/stdin
    echo "  Python: $(python --version)"
    echo "  torch:  $(python -c 'import torch; print(torch.__version__)')"
    echo "  XLA:    $(python -c 'import torch_xla; print(torch_xla.__version__)')"
    cd "${REPO_DIR}"
    python -c "from es_finetuning import ESTrainer, ESConfig; print('  imports OK')"
    echo ''

# ===========================================================================
# GPU / local — standard venv via activate.sh
# ===========================================================================
else

    if [ "${MODE}" = gpu ]; then
        if [ -n "${GPU_ID}" ]; then
            export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi &>/dev/null; then
            GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1 | tr -d ' ')
            export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        fi
        echo "GPU:   ${CUDA_VISIBLE_DEVICES:-not set}"
        echo ''
    fi

    echo '[1/3] Python environment...'
    cd "${REPO_DIR}"
    source "${SCRIPT_DIR}/activate.sh"
    python -c "from es_finetuning import ESTrainer, ESConfig; print('  imports OK')"
    echo "  Python: $(python --version)"
    echo ''

fi

# ===========================================================================
# Shared: HuggingFace, wandb, GCS, Claude Code
# ===========================================================================
if [ "${MODE}" = tpu ]; then
    _HF_STEP="[3/4]"; _WANDB_STEP="[4/4]"
else
    _HF_STEP="[2/3]"; _WANDB_STEP="[3/3]"
fi

echo "${_HF_STEP} HuggingFace..."
check_hf_login
echo ''

if [ "${SETUP_WANDB}" = true ]; then
    echo "${_WANDB_STEP} Weights & Biases..."
    setup_wandb
else
    echo "${_WANDB_STEP} Skipping wandb (--skip-wandb)."
fi
echo ''

if [ "${SETUP_GCS}" = true ]; then
    echo '[+] Google Cloud Storage...'
    setup_gcs
else
    echo '[+] Skipping GCS (--skip-gcs).'
fi
echo ''

if [ "${INSTALL_CLAUDE}" = true ]; then
    echo '[+] Claude Code...'
    install_claude
    echo ''
fi

# ===========================================================================
# Summary
# ===========================================================================
echo '============================================================'
echo ' Setup complete!'
echo '============================================================'
echo ''
echo "Repo:  ${REPO_DIR}"
echo "venv:  ${VENV_DIR}"
echo ''

if [ "${MODE}" = tpu ]; then
    echo 'Activate in new shells:'
    echo '  source setup/activate_tpu.sh'
    echo ''
    echo 'Smoke test:'
    echo '  export PJRT_DEVICE=TPU'
    echo "  python -c \"import torch_xla.core.xla_model as xm; print('TPU:', xm.xla_device())\""
    echo '  python scripts/run_es.py --run_name tpu_smoke --num_iterations 2 \'
    echo '    --population_size 2 --mini_batch_size 2 --no-gcs'
else
    echo 'Activate in new shells:'
    echo '  source setup/activate.sh     # bash/zsh'
    echo '  source setup/activate.csh    # csh/tcsh (UCL GPU machines)'
    echo ''
    echo 'Smoke test:'
    echo '  python scripts/run_es.py --run_name smoke --num_iterations 2 \'
    echo '    --population_size 2 --mini_batch_size 2 --no-gcs'
fi
echo ''
