#!/usr/bin/env bash
# bash/zsh ONLY — csh/tcsh users (UCL): source setup/activate.csh
#
# Source this to activate the evo-memory environment:
#   source setup/activate.sh            # auto-detect: TPU → GPU → local
#   source setup/activate.sh --tpu      # force TPU mode
#   source setup/activate.sh --gpu      # force GPU mode
#   source setup/activate.sh --local    # force local/CPU mode
#
# First run: creates venv and installs deps (~2 min).
# Subsequent: activates venv (~5 s).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${REPO_DIR}/venv"

# ── Detect mode ───────────────────────────────────────────────────────────────
_ACT_MODE=""
for _arg in "$@"; do
    case "$_arg" in
        --tpu)   _ACT_MODE=tpu   ;;
        --gpu)   _ACT_MODE=gpu   ;;
        --local) _ACT_MODE=local ;;
    esac
done

if [ -z "${_ACT_MODE}" ]; then
    if [ -e /dev/accel0 ] || [ -d /dev/vfio ]; then
        _ACT_MODE=tpu
    elif command -v nvidia-smi &>/dev/null; then
        _ACT_MODE=gpu
    else
        _ACT_MODE=local
    fi
fi

# ── Venv ──────────────────────────────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# On TPU, torch+XLA are installed separately by setup.sh — only sync the
# non-torch deps so we never overwrite the XLA-compatible torch build.
if [ "${_ACT_MODE}" = tpu ]; then
    grep -vE '^(torch|torchvision|torchaudio|--index-url|--extra-index-url|#|[[:space:]]*$)' \
        "${REPO_DIR}/requirements.txt" | pip install -q -r /dev/stdin 2>/dev/null
else
    pip install -q -r "${REPO_DIR}/requirements.txt"
fi

# ── .env ──────────────────────────────────────────────────────────────────────
if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_DIR}/.env"
    set +a
fi

# ── Shared exports ────────────────────────────────────────────────────────────
export HF_HOME="${HF_CACHE_DIR:-${REPO_DIR}/.hf_cache}"
export GCS_BUCKET="${GCS_BUCKET:-statistical-nlp}"
export GCS_PROJECT="${GCS_PROJECT:-statistical-nlp}"

# ── Mode-specific exports ────────────────────────────────────────────────────
if [ "${_ACT_MODE}" = tpu ]; then
    export PJRT_DEVICE=TPU
    export VM_ID="${VM_ID:-$(hostname)}"

    # Persist XLA compiled graphs so recompilation is skipped across reboots.
    export XLA_PERSISTENT_CACHE_PATH="${REPO_DIR}/.xla_cache"
    mkdir -p "${XLA_PERSISTENT_CACHE_PATH}" 2>/dev/null

    # Pull cached XLA graphs from GCS if local cache is empty.
    if [ -z "$(ls -A "${XLA_PERSISTENT_CACHE_PATH}" 2>/dev/null)" ]; then
        echo "Downloading XLA cache from GCS..."
        gsutil -m rsync -r "gs://${GCS_BUCKET}/xla_cache" "${XLA_PERSISTENT_CACHE_PATH}" 2>/dev/null || true
    fi
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

cd "${REPO_DIR}"

if [ "${_ACT_MODE}" = tpu ]; then
    echo "Activated: venv=${VENV_DIR}, PJRT_DEVICE=TPU, GCS=${GCS_BUCKET}, VM=${VM_ID}, cwd=$(pwd)"
else
    echo "Activated: venv=${VENV_DIR}, GPU=${CUDA_VISIBLE_DEVICES}, GCS=${GCS_BUCKET}, cwd=$(pwd)"
fi

unset _ACT_MODE _arg
