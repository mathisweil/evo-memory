#!/usr/bin/env bash
# bash/zsh ONLY — csh/tcsh users (UCL): source setup/activate.csh
#
# Source this to activate the environment and install deps:
#   source setup/activate.sh
#
# First time: creates venv and installs everything (~2 min).
# Subsequent: activates venv and verifies deps are current (~5s).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${REPO_DIR}/venv"

# Create venv if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Install/update deps
pip install -q -r "${REPO_DIR}/requirements.txt"

# Load .env if present (values already in environment take precedence)
if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_DIR}/.env"
    set +a
fi

export HF_HOME="${HF_CACHE_DIR:-${REPO_DIR}/.hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GCS_BUCKET="${GCS_BUCKET:-statistical-nlp}"
export GCS_PROJECT="${GCS_PROJECT:-statistical-nlp}"
cd "${REPO_DIR}"

echo "Activated: venv=${VENV_DIR}, GPU=${CUDA_VISIBLE_DEVICES}, GCS=${GCS_BUCKET}, cwd=$(pwd)"
