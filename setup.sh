#!/usr/bin/env bash
# =============================================================================
# GPU VM Setup: ES Fine-Tuning with NAMM
#
# Clones both repos (correct branches), creates conda env, installs all deps,
# installs Claude Code, and configures wandb + HuggingFace.
#
# Usage:
#   bash setup.sh              # auto-detect first available GPU
#   bash setup.sh --gpu 0      # pin to GPU 0
#   bash setup.sh --gpu 2      # pin to GPU 2 on a multi-GPU machine
#
# Prerequisites:
#   - conda or miniconda installed
#   - CUDA 12.1+ GPU drivers
#   - Internet access (GitHub, PyPI, HuggingFace, npm)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
GPU_ID=""
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash setup.sh [--gpu GPU_ID]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Config — edit these if your paths differ
# ---------------------------------------------------------------------------
WORK_DIR="${HOME}/es-finetuning-workspace"
EVO_MEMORY_REPO="https://github.com/mathisweil/evo-memory.git"
EVO_MEMORY_BRANCH="es-fine-tuning"
ES_PAPER_REPO="https://github.com/shr1ram/es-fine-tuning-paper.git"
ES_PAPER_BRANCH="claude-inshallah"
CONDA_ENV_NAME="th2"
HF_CACHE_DIR="${WORK_DIR}/.hf_cache"

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
echo '[1/7] Cloning repositories...'

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
# 3. Set up conda environment
# ---------------------------------------------------------------------------
echo '[2/7] Setting up conda environment...'

# Find conda
CONDA_SH=""
for candidate in \
    "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/cs/student/project_msc/2025/csml/gmaralla/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "${candidate}" ]; then
        CONDA_SH="${candidate}"
        break
    fi
done

if [ -z "${CONDA_SH}" ]; then
    echo 'ERROR: conda not found. Install miniconda first:'
    echo '  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
    echo '  bash Miniconda3-latest-Linux-x86_64.sh'
    exit 1
fi

source "${CONDA_SH}"
echo "  Found conda at: ${CONDA_SH}"

# Create or update the conda env
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Conda env '${CONDA_ENV_NAME}' already exists. Updating..."
    conda activate "${CONDA_ENV_NAME}"
else
    echo "  Creating conda env '${CONDA_ENV_NAME}' from env_namm.yaml..."
    conda env create -f "${WORK_DIR}/evo-memory/env_namm.yaml"
    conda activate "${CONDA_ENV_NAME}"
fi

echo "  Active env: ${CONDA_DEFAULT_ENV}"
echo "  Python: $(python --version)"
echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 4. Install es-finetuning package (editable)
# ---------------------------------------------------------------------------
echo '[3/7] Installing es-finetuning package...'

pip install -e "${WORK_DIR}/es-fine-tuning-paper/" 2>&1 | tail -3
python -c "from es_finetuning import ESTrainer, ESConfig; print('  es_finetuning imports OK')"

echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 5. Install tensorboard (not in env_namm.yaml but needed by es_finetuning)
# ---------------------------------------------------------------------------
echo '[4/7] Installing additional dependencies...'

pip install tensorboard 2>&1 | tail -3

echo '  Done.'
echo ''

# ---------------------------------------------------------------------------
# 6. HuggingFace setup
# ---------------------------------------------------------------------------
echo '[5/7] Setting up HuggingFace...'

mkdir -p "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
echo "  HF_HOME=${HF_HOME}"

# Persist HF_HOME and CUDA_VISIBLE_DEVICES in conda env activation
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh" << ENVEOF
export HF_HOME="${HF_CACHE_DIR}"
ENVEOF

if [ -n "${GPU_ID}" ]; then
    cat >> "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh" << GPUEOF
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
GPUEOF
    echo "  CUDA_VISIBLE_DEVICES=${GPU_ID} (persisted in conda activate)"
fi

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
# 7. wandb setup
# ---------------------------------------------------------------------------
echo '[6/7] Setting up wandb...'

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
# 8. Install Claude Code
# ---------------------------------------------------------------------------
echo '[7/7] Installing Claude Code...'

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
echo "Conda env:        ${CONDA_ENV_NAME}"
echo "HF cache:         ${HF_CACHE_DIR}"
if [ -n "${GPU_ID}" ]; then
    echo "GPU:              ${GPU_ID} (CUDA_VISIBLE_DEVICES persisted in conda activate)"
fi
echo ''
echo 'To run ES fine-tuning:'
echo "  conda activate ${CONDA_ENV_NAME}"
echo "  cd ${WORK_DIR}/evo-memory"
echo '  python run_es_finetuning.py --num_iterations 2 --population_size 2 --mini_batch_size 2'
echo ''
echo 'To start Claude Code:'
echo "  cd ${WORK_DIR}/evo-memory"
echo '  claude'
echo ''
