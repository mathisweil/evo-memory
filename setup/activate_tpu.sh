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

# Persist XLA compiled graphs so recompilation is skipped across reboots.
# Local path (XLA doesn't support gs:// natively). Sync to/from GCS with:
#   gsutil -m rsync -r .xla_cache gs://statistical-nlp/xla_cache   (upload)
#   gsutil -m rsync -r gs://statistical-nlp/xla_cache .xla_cache    (download)
export XLA_PERSISTENT_CACHE_PATH="${REPO_DIR}/.xla_cache"
mkdir -p "${XLA_PERSISTENT_CACHE_PATH}" 2>/dev/null

# Pull cached XLA graphs from GCS only when explicitly requested.
# Set XLA_CACHE_DOWNLOAD=1 before sourcing this script to enable.
XLA_CACHE_DOWNLOAD="${XLA_CACHE_DOWNLOAD:-0}"
if [ "${XLA_CACHE_DOWNLOAD}" = "1" ]; then
    if [ -z "$(ls -A "${XLA_PERSISTENT_CACHE_PATH}" 2>/dev/null)" ]; then
        echo "Downloading XLA cache from GCS (XLA_CACHE_DOWNLOAD=1)..."
        gsutil -m rsync -r "gs://${GCS_BUCKET}/xla_cache" "${XLA_PERSISTENT_CACHE_PATH}" 2>/dev/null || true
    else
        echo "Skipping XLA cache download (local cache is non-empty)."
    fi
else
    echo "Skipping XLA cache download (set XLA_CACHE_DOWNLOAD=1 to enable)."
fi

cd "${REPO_DIR}"

echo "Activated: venv=${VENV_DIR}, PJRT_DEVICE=TPU, GCS=${GCS_BUCKET}, VM=${VM_ID}, cwd=$(pwd)"
