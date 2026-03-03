#!/usr/bin/env bash
# =============================================================================
# GPU VM Setup: ES Fine-Tuning with NAMM
#
# Clones both repos (correct branches), creates a Python venv, installs all
# deps, installs Claude Code, and configures wandb + HuggingFace.
#
# Usage:
#   bash setup.sh              # auto-detect first available GPU
#   bash setup.sh --gpu 0      # pin to GPU 0
#   bash setup.sh --gpu 2      # pin to GPU 2 on a multi-GPU machine
#
# Prerequisites:
#   - python3 (3.10+) and pip
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
VENV_DIR="${WORK_DIR}/venv"
HF_CACHE_DIR="${WORK_DIR}/.hf_cache"
ENV_SCRIPT="${WORK_DIR}/activate.sh"

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
# 3. Create Python venv and install dependencies
# ---------------------------------------------------------------------------
echo '[2/7] Setting up Python venv...'

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created venv at ${VENV_DIR}"
else
    echo "  venv already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo '  Installing pinned dependencies...'
pip install --upgrade pip 2>&1 | tail -1

# PyTorch with CUDA 12.1 (must be installed separately for --index-url)
pip install \
    "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1" \
    --index-url https://download.pytorch.org/whl/cu121 \
    2>&1 | tail -1

# All other deps (versions match env_namm.yaml)
pip install \
    "numpy<2" \
    "transformers==4.41.2" \
    "accelerate" \
    "datasets==2.20.0" \
    "tiktoken" \
    "wandb==0.16.6" \
    "tqdm" \
    "hydra-core==1.3.2" \
    "pandas==2.2.2" \
    "lm-eval==0.4.2" \
    "fugashi==1.3.2" \
    "ftfy" \
    "peft==0.11.1" \
    "bitsandbytes" \
    "rouge" \
    "jieba" \
    "fuzzywuzzy" \
    "einops" \
    "scipy==1.13.0" \
    "sentencepiece" \
    2>&1 | tail -3

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
# 9. Write activate.sh convenience script
# ---------------------------------------------------------------------------
cat > "${ENV_SCRIPT}" << ACTIVATEEOF
#!/usr/bin/env bash
# Source this to activate the environment:
#   source ${ENV_SCRIPT}
source "${VENV_DIR}/bin/activate"
export HF_HOME="${HF_CACHE_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cd "${WORK_DIR}/evo-memory"
ACTIVATEEOF
chmod +x "${ENV_SCRIPT}"

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
echo "  source ${ENV_SCRIPT}"
echo ''
echo 'To run ES fine-tuning:'
echo "  source ${ENV_SCRIPT}"
echo '  python run_es_finetuning.py --num_iterations 2 --population_size 2 --mini_batch_size 2'
echo ''
echo 'To start Claude Code:'
echo "  source ${ENV_SCRIPT}"
echo '  claude'
echo ''
