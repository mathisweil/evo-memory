#!/usr/bin/env bash
# =============================================================================
# GPU VM Setup: ES Fine-Tuning with NAMM
#
# Clones both repos (correct branches), creates a Python venv, installs all
# deps, installs Claude Code, and configures wandb + HuggingFace.
#
# Usage:
#   bash setup.sh                        # UCL VM, uses $(whoami)
#   bash setup.sh --user jsmith --gpu 0  # UCL VM, explicit username + GPU
#   bash setup.sh --dir ~/ft-namm        # any machine, custom workspace dir
#   bash setup.sh --noclaude             # skip Claude Code install
#
# Prerequisites:
#   - python3 (3.9+) and pip
#   - CUDA 12.1+ GPU drivers
#   - Internet access (GitHub, PyPI, HuggingFace, npm)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
GPU_ID=""
USER_NAME=""
CUSTOM_DIR=""
INSTALL_CLAUDE=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --gpu=*)
            GPU_ID="${1#*=}"
            shift
            ;;
        --user)
            USER_NAME="$2"
            shift 2
            ;;
        --user=*)
            USER_NAME="${1#*=}"
            shift
            ;;
        --dir)
            CUSTOM_DIR="$2"
            shift 2
            ;;
        --dir=*)
            CUSTOM_DIR="${1#*=}"
            shift
            ;;
        --noclaude)
            INSTALL_CLAUDE=false
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash setup.sh [--dir DIR | --user USERNAME] [--gpu GPU_ID] [--noclaude]"
            exit 1
            ;;
    esac
done

USER_NAME="${USER_NAME:-$(whoami)}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
if [ -n "${CUSTOM_DIR}" ]; then
    WORK_DIR="${CUSTOM_DIR}"
else
    WORK_DIR="/cs/student/project_msc/2025/csml/${USER_NAME}/SNLP/FT-NAMM"
fi
EVO_MEMORY_REPO="https://github.com/mathisweil/evo-memory.git"
EVO_MEMORY_BRANCH="es-fine-tuning"
ES_PAPER_REPO="https://github.com/shr1ram/es-fine-tuning-paper.git"
ES_PAPER_BRANCH="claude-inshallah"
VENV_DIR="${WORK_DIR}/venv"
HF_CACHE_DIR="${WORK_DIR}/.hf_cache"
REPO_DIR="${WORK_DIR}/evo-memory"
SCRIPT_DIR="${REPO_DIR}/scripts"

echo '============================================================'
echo ' ES Fine-Tuning + NAMM — GPU VM Setup'
echo '============================================================'
echo ''
echo "Workspace: ${WORK_DIR}"

# ---------------------------------------------------------------------------
# GPU selection — pin to a single GPU
# ---------------------------------------------------------------------------
if [ -n "${GPU_ID}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    echo "GPU:       ${GPU_ID} (set via --gpu)"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Auto-detect: pick the first available GPU
    if command -v nvidia-smi &>/dev/null; then
        GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1 | tr -d ' ')
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        echo "GPU:       ${GPU_ID} (auto-detected first GPU)"
    else
        echo "GPU:       nvidia-smi not found, skipping GPU pinning"
    fi
else
    GPU_ID="${CUDA_VISIBLE_DEVICES}"
    echo "GPU:       ${CUDA_VISIBLE_DEVICES} (from existing CUDA_VISIBLE_DEVICES)"
fi
echo ''

# ---------------------------------------------------------------------------
# 1. Create workspace
# ---------------------------------------------------------------------------
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# ---------------------------------------------------------------------------
# 2. Clone repos (or update if already cloned)
# ---------------------------------------------------------------------------
echo '[1/5] Cloning repositories...'

if [ -d "evo-memory" ]; then
    echo '  evo-memory already exists, pulling latest...'
    cd evo-memory
    git fetch origin
    git checkout "${EVO_MEMORY_BRANCH}"
    git pull origin "${EVO_MEMORY_BRANCH}"
    cd "${WORK_DIR}"
else
    git clone -b "${EVO_MEMORY_BRANCH}" "${EVO_MEMORY_REPO}"
fi

if [ -d "es-fine-tuning-paper" ]; then
    echo '  es-fine-tuning-paper already exists, pulling latest...'
    cd es-fine-tuning-paper
    git fetch origin
    git checkout "${ES_PAPER_BRANCH}"
    git pull origin "${ES_PAPER_BRANCH}"
    cd "${WORK_DIR}"
else
    git clone -b "${ES_PAPER_BRANCH}" "${ES_PAPER_REPO}"
fi

echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 3. Install Python dependencies via activate.sh
# ---------------------------------------------------------------------------
echo '[2/5] Installing Python dependencies...'

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
source "${SCRIPT_DIR}/activate.sh"

pip install --upgrade pip 2>&1 | tail -1
python -c "from es_finetuning import ESTrainer, ESConfig; print('  es_finetuning imports OK')"

echo "  Python: $(python --version)"
echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 4. HuggingFace setup
# ---------------------------------------------------------------------------
echo '[3/5] Setting up HuggingFace...'

mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
echo "  HF_HOME=${HF_HOME}"

# Check if already logged in
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
# 5. wandb setup
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
# 6. Install Claude Code
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
echo ' Setup complete!'
echo '============================================================'
echo ''
echo "Workspace:        ${WORK_DIR}"
echo "evo-memory:       ${WORK_DIR}/evo-memory (branch: ${EVO_MEMORY_BRANCH})"
echo "es-fine-tuning:   ${WORK_DIR}/es-fine-tuning-paper (branch: ${ES_PAPER_BRANCH})"
echo "venv:             ${VENV_DIR}"
echo "HF cache:         ${HF_CACHE_DIR}"
if [ -n "${GPU_ID}" ]; then
    echo "GPU:              ${GPU_ID}"
fi
echo ''
echo 'To activate the environment in a new shell:'
echo "  source ${SCRIPT_DIR}/activate.sh"
echo ''
echo 'To run ES fine-tuning:'
echo "  source ${SCRIPT_DIR}/activate.sh"
echo '  python run_es_finetuning.py --num_iterations 2 --population_size 2 --mini_batch_size 2'
echo ''
echo 'To start Claude Code:'
echo "  source ${SCRIPT_DIR}/activate.sh"
echo '  claude'
echo ''
