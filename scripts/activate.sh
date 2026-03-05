#!/usr/bin/env bash
# Source this to activate the environment and install deps:
#   source scripts/activate.sh
#
# First time: creates venv and installs everything (~2 min).
# Subsequent: activates venv and verifies deps are current (~5s).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
WORK_DIR="$(dirname "${REPO_DIR}")"
VENV_DIR="${WORK_DIR}/venv"

# Create venv if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Install/update deps from both repos
pip install -q -r "${REPO_DIR}/requirements.txt"
pip install -q -e "${WORK_DIR}/es-fine-tuning-paper/"

export HF_HOME="${WORK_DIR}/.hf_cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${REPO_DIR}"

echo "Activated: venv=${VENV_DIR}, GPU=${CUDA_VISIBLE_DEVICES}, cwd=$(pwd)"
