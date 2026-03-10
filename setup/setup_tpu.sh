#!/usr/bin/env bash
# =============================================================================
# TPU VM Setup: ES Fine-Tuning with NAMM
#
# Creates a Python venv, installs PyTorch + torch_xla + all deps,
# configures HuggingFace, and optionally installs Claude Code.
#
# Usage (run ON the TPU VM after cloning the repo):
#   bash setup/setup_tpu.sh
#   bash setup/setup_tpu.sh --noclaude    # skip Claude Code install
#
# Or use setup_tpu_cmd.sh to bootstrap from scratch (clones repo first).
#
# Prerequisites:
#   - TPU VM created with runtime version tpu-ubuntu2204-base (v4)
#     or v2-alpha-tpuv5-lite (v5e) or v2-alpha-tpuv6e (v6e)
#   - python3 (3.10+) and pip
#   - Internet access (GitHub, PyPI, HuggingFace, npm)
#
# See docs/ for guides on ES fine-tuning and NAMM.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
INSTALL_CLAUDE=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --noclaude)
            INSTALL_CLAUDE=false
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash setup/setup_tpu.sh [--noclaude]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate repo (script must be run from repo root or setup/ dir)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${REPO_DIR}/venv"
HF_CACHE_DIR="${REPO_DIR}/.hf_cache"

echo '============================================================'
echo ' ES Fine-Tuning + NAMM — TPU VM Setup'
echo '============================================================'
echo ''
echo "Repo:     ${REPO_DIR}"
echo ''

# ---------------------------------------------------------------------------
# Verify TPU is available
# ---------------------------------------------------------------------------
echo '[0/5] Checking TPU availability...'
if [ -e /dev/accel0 ] || [ -d /dev/vfio ]; then
    echo '  TPU device detected.'
else
    echo '  WARNING: No TPU device found (/dev/accel0 or /dev/vfio).'
    echo '  Continuing anyway (deps will install, but training will fail).'
fi
echo ''

# ---------------------------------------------------------------------------
# 1. System deps
# ---------------------------------------------------------------------------
echo '[1/5] Installing system dependencies...'
sudo apt-get update -qq || echo '  (apt-get update had warnings — continuing)'
sudo apt-get install -y -qq libopenblas-dev python3.10-venv > /dev/null 2>&1
echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 2. Python venv + PyTorch/XLA + project deps
# ---------------------------------------------------------------------------
echo '[2/5] Setting up Python environment...'

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    rm -rf "${VENV_DIR}"
    echo "  Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

# PyTorch + XLA (TPU support)
echo '  Installing PyTorch + torch_xla...'
pip install torch torch_xla[tpu] \
    -f https://storage.googleapis.com/libtpu-releases/index.html \
    -q

# Project dependencies (skip GPU-only packages)
echo '  Installing project dependencies...'
pip install -q \
    "numpy<2" \
    "transformers==4.45.2" \
    accelerate \
    "datasets==2.20.0" \
    tiktoken \
    "wandb==0.16.6" \
    tqdm \
    "hydra-core==1.3.2" \
    "pandas==2.2.2" \
    "lm-eval==0.4.2" \
    "fugashi==1.3.2" \
    ftfy \
    "peft==0.11.1" \
    rouge \
    jieba \
    fuzzywuzzy \
    einops \
    "scipy==1.13.0" \
    sentencepiece \
    tensorboard \
    matplotlib \
    google-cloud-storage \
    "fsspec<=2024.5.0" \
    "protobuf<5,>=3.19.0"

# Deliberately NOT installing: bitsandbytes (GPU-only), torchvision, torchaudio

echo "  Python: $(python --version)"
echo "  torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "  XLA:    $(python -c 'import torch_xla; print(torch_xla.__version__)')"
cd "${REPO_DIR}"
python -c "from es_finetuning import ESTrainer, ESConfig; print('  es_finetuning imports OK')"
echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 3. HuggingFace setup
# ---------------------------------------------------------------------------
echo '[3/5] Setting up HuggingFace...'

mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
echo "  HF_HOME=${HF_HOME}"

if python -c "from huggingface_hub import HfFolder; assert HfFolder.get_token()" 2>/dev/null; then
    echo '  Already logged into HuggingFace.'
else
    echo '  Please log in to HuggingFace (needed for Llama 3.2 access):'
    echo '  Get your token from: https://huggingface.co/settings/tokens'
    huggingface-cli login
fi

echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 4. wandb setup
# ---------------------------------------------------------------------------
echo '[4/5] Setting up wandb...'

if python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
    echo '  Already logged into wandb.'
else
    echo '  Please log in to wandb:'
    echo '  Get your API key from: https://wandb.ai/authorize'
    wandb login
fi

echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 5. Install Claude Code
# ---------------------------------------------------------------------------
if [ "${INSTALL_CLAUDE}" = true ]; then
    echo '[5/5] Installing Claude Code...'

    if command -v claude &>/dev/null; then
        echo '  Claude Code already installed.'
        claude --version
    else
        if command -v npm &>/dev/null; then
            npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
            echo '  Claude Code installed.'
        elif command -v node &>/dev/null; then
            echo '  npm not found but node is available. Installing npm...'
            curl -fsSL https://npmjs.org/install.sh | sh
            npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
        else
            echo '  Node.js not found. Installing via nvm...'
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
            export NVM_DIR="$HOME/.nvm"
            [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
            nvm install --lts
            npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
        fi
    fi
else
    echo '[5/5] Skipping Claude Code install (--noclaude)'
fi

echo ''

# ---------------------------------------------------------------------------
# Done — print summary
# ---------------------------------------------------------------------------
echo '============================================================'
echo ' TPU Setup complete!'
echo '============================================================'
echo ''
echo "Repo:             ${REPO_DIR}"
echo "venv:             ${VENV_DIR}"
echo "HF cache:         ${HF_CACHE_DIR}"
echo ''
echo 'To activate the environment in a new shell:'
echo "  source ${REPO_DIR}/setup/activate_tpu.sh"
echo ''
echo 'Quick TPU smoke test:'
echo '  export PJRT_DEVICE=TPU'
echo "  python -c \"import torch_xla.core.xla_model as xm; print('TPU:', xm.xla_device())\""
echo ''
echo 'To run ES fine-tuning (smoke test):'
echo "  source ${REPO_DIR}/setup/activate_tpu.sh"
echo '  python scripts/run_es.py --run_name tpu_smoke --num_iterations 2 --population_size 2 --mini_batch_size 2'
echo ''
echo 'To start Claude Code:'
echo "  source ${REPO_DIR}/setup/activate_tpu.sh"
echo '  claude'
echo ''
