#!/bin/tcsh
# Activate the environment in csh/tcsh (UCL GPU machines).
#
# Usage — run from the repository root:
#   source setup/activate.csh
#
# First time: creates venv and installs deps (~2 min).
# Subsequent: activates venv and verifies deps are current (~5s).
#
# Environment variables are NOT auto-loaded from .env in csh.
# Set them in ~/.cshrc before sourcing this script (see .env.example):
#   setenv LLM_MODEL_PATH  meta-llama/Llama-3.2-1B-Instruct
#   setenv HF_CACHE_DIR    /path/to/hf/cache
#   setenv CUDA_VISIBLE_DEVICES 0

set REPO_DIR = "$cwd"
set VENV_DIR = "${REPO_DIR}/venv"

# Create venv if it doesn't exist
if (! -d "${VENV_DIR}") then
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
endif

# Activate venv (Python generates activate.csh automatically)
source "${VENV_DIR}/bin/activate.csh"

# Install/update deps
pip install -q -r "${REPO_DIR}/requirements.txt"

# Set env var defaults (skip if already set)
if (! $?HF_CACHE_DIR) then
    setenv HF_HOME "${REPO_DIR}/.hf_cache"
else
    setenv HF_HOME "$HF_CACHE_DIR"
endif

if (! $?CUDA_VISIBLE_DEVICES) then
    setenv CUDA_VISIBLE_DEVICES 0
endif

if (! $?GCS_BUCKET) then
    setenv GCS_BUCKET statistical-nlp
endif

if (! $?GCS_PROJECT) then
    setenv GCS_PROJECT statistical-nlp
endif

echo "Activated: venv=${VENV_DIR}, GPU=${CUDA_VISIBLE_DEVICES}, GCS=${GCS_BUCKET}, cwd=`pwd`"
