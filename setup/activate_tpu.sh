#!/usr/bin/env bash
# Source this to activate the TPU environment:
#   source setup/activate_tpu.sh
#
# First time after setup_tpu.sh: just activates venv (~instant).
# Sets PJRT_DEVICE=TPU so torch_xla uses the TPU backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${REPO_DIR}/venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: venv not found at ${VENV_DIR}"
    echo "Run setup/setup_tpu.sh first."
    return 1 2>/dev/null || exit 1
fi

source "${VENV_DIR}/bin/activate"

export HF_HOME="${REPO_DIR}/.hf_cache"
export PJRT_DEVICE=TPU
export GCS_BUCKET="statistical-nlp"
export GCS_PROJECT="statistical-nlp"
export VM_ID="${VM_ID:-$(hostname)}"

# Persist XLA compiled graphs so recompilation is skipped across VM teardowns.
export XLA_PERSISTENT_CACHE_PATH="gs://statistical-nlp/xla_cache"

cd "${REPO_DIR}"

echo "Activated: venv=${VENV_DIR}, PJRT_DEVICE=TPU, GCS=${GCS_BUCKET}, VM=${VM_ID}, cwd=$(pwd)"
