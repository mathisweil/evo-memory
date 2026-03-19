#!/usr/bin/env bash
# =============================================================================
# GPU VM Setup: ES Fine-Tuning with NAMM
#
# Clones the repo, creates a Python venv, installs all deps, installs
# Claude Code, and configures wandb + HuggingFace.
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
REPO_DIR="${WORK_DIR}/evo-memory"
VENV_DIR="${REPO_DIR}/venv"
HF_CACHE_DIR="${REPO_DIR}/.hf_cache"
SETUP_DIR="${REPO_DIR}/setup"

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

echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 3. Install Python dependencies via activate.sh
# ---------------------------------------------------------------------------
echo '[2/5] Installing Python dependencies...'

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
source "${SETUP_DIR}/activate.sh"

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
# 6. Google Cloud Storage setup
# ---------------------------------------------------------------------------
echo '[5/6] Setting up Google Cloud Storage...'

GCS_BUCKET="${GCS_BUCKET:-statistical-nlp}"
GCS_PROJECT="${GCS_PROJECT:-statistical-nlp}"

# Install gcloud CLI if not available
if ! command -v gcloud &>/dev/null; then
    echo '  gcloud CLI not found. Installing...'
    GCLOUD_DIR="${HOME}/.local/google-cloud-sdk"
    if [ ! -d "${GCLOUD_DIR}" ]; then
        curl -fsSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir="${HOME}/.local" 2>&1 | tail -5
    fi
    export PATH="${GCLOUD_DIR}/bin:${PATH}"
fi

# Authenticate if needed
if gcloud auth application-default print-access-token &>/dev/null 2>&1; then
    echo '  Already authenticated with GCS.'
else
    echo '  Please authenticate with Google Cloud (needed for experiment archival):'
    gcloud auth application-default login --project "${GCS_PROJECT}" --no-launch-browser
fi

# Verify bucket access
if gsutil ls "gs://${GCS_BUCKET}/" &>/dev/null 2>&1; then
    echo "  Bucket gs://${GCS_BUCKET}/ accessible."
else
    echo "  WARNING: Cannot access gs://${GCS_BUCKET}/. You may need to run: gcloud auth application-default login"
fi

echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 7. Install Claude Code
# ---------------------------------------------------------------------------
if [ "${INSTALL_CLAUDE}" = true ]; then
    echo '[6/6] Installing Claude Code...'

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
    echo '[6/6] Skipping Claude Code install (--noclaude)'
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
echo "venv:             ${VENV_DIR}"
echo "HF cache:         ${HF_CACHE_DIR}"
if [ -n "${GPU_ID}" ]; then
    echo "GPU:              ${GPU_ID}"
fi
echo ''
echo 'To activate the environment in a new shell:'
echo "  source ${SETUP_DIR}/activate.sh"
echo ''
echo 'To run ES fine-tuning:'
echo "  source ${SETUP_DIR}/activate.sh"
echo '  python scripts/run_es.py --run_name test --num_iterations 2 --population_size 2 --mini_batch_size 2'
echo ''
echo 'To start Claude Code:'
echo "  source ${SETUP_DIR}/activate.sh"
echo '  claude'
echo ''
