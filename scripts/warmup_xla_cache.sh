#!/usr/bin/env bash
set -euo pipefail

# Warm up TPU/XLA compilation paths by running one short ES iteration per config.
#
# Usage:
#   source setup/activate_tpu.sh
#   bash scripts/warmup_xla_cache.sh
#
# Optional overrides:
#   NAMM_CKPT=/abs/path/to/namm_pretrained_romain_v2.pt \
#   BATCH_SIZE=18 POP_SIZE=2 CACHE_SIZES="1024 2048 3072 4096 5120 6144" \
#   NUM_ITERATIONS=1 FILTER_BY_LENGTH=32768 bash scripts/warmup_xla_cache.sh
#
# If NAMM_CKPT is unset or set to "latest", the script resolves the newest
# cached or GCS-backed pretrained NAMM checkpoint automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

if [[ "${PJRT_DEVICE:-}" != "TPU" ]]; then
    echo "ERROR: PJRT_DEVICE=TPU is required. Run: source setup/activate_tpu.sh"
    exit 1
fi

resolve_namm_ckpt() {
    if [[ -n "${NAMM_CKPT:-}" && "${NAMM_CKPT}" != "latest" ]]; then
        if [[ ! -f "${NAMM_CKPT}" ]]; then
            echo "ERROR: NAMM checkpoint not found: ${NAMM_CKPT}"
            exit 1
        fi
        return
    fi

    echo "Resolving latest pretrained NAMM checkpoint..."
    local resolved=""
    if ! resolved="$(python3 scripts/upload_pretrained.py --latest-path)"; then
        echo "ERROR: Failed to resolve latest pretrained NAMM checkpoint."
        echo "Set NAMM_CKPT=/abs/path/to/checkpoint.pt or upload one with:"
        echo "  python3 scripts/upload_pretrained.py /abs/path/to/checkpoint.pt"
        exit 1
    fi
    if [[ -z "${resolved}" || ! -f "${resolved}" ]]; then
        echo "ERROR: Latest pretrained NAMM checkpoint did not resolve to a local file: ${resolved}"
        exit 1
    fi

    export NAMM_CKPT="${resolved}"
    echo "Using NAMM_CKPT=${NAMM_CKPT}"
}

resolve_namm_ckpt

BATCH_SIZE="${BATCH_SIZE:-18}"
POP_SIZE="${POP_SIZE:-2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
FILTER_BY_LENGTH="${FILTER_BY_LENGTH:-32768}"
CACHE_SIZES="${CACHE_SIZES:-1024 2048 3072 4096 5120 6144}"

COMMON_ARGS=(
    --num_iterations "${NUM_ITERATIONS}"
    --population_size "${POP_SIZE}"
    --mini_batch_size "${BATCH_SIZE}"
    --batch_size "${BATCH_SIZE}"
    --checkpoint_every 0
    --filter_by_length "${FILTER_BY_LENGTH}"
    --no-gcs
)

TS="$(date +%Y%m%d_%H%M%S)"

run_warmup() {
    local label="$1"
    shift
    echo "========================================================================"
    echo "Warmup: ${label}"
    echo "Command: python3 scripts/run_es.py $* ${COMMON_ARGS[*]}"
    echo "========================================================================"
    python3 scripts/run_es.py "$@" "${COMMON_ARGS[@]}"
}

echo "Starting TPU/XLA warmup runs..."
echo "BATCH_SIZE=${BATCH_SIZE}, POP_SIZE=${POP_SIZE}, NUM_ITERATIONS=${NUM_ITERATIONS}"
echo "CACHE_SIZES=${CACHE_SIZES}"
echo "NAMM_CKPT=${NAMM_CKPT}"
echo

# 1) Full-cache baseline (es_only)
run_warmup \
    "full_cache" \
    --run_name "warmup_full_${TS}" \
    --method es_only \
    --run_config full_cache_es_llama32_1b_tpu

# 2) Recency runs across cache sizes
for cache in ${CACHE_SIZES}; do
    run_warmup \
        "recency_c${cache}" \
        --run_name "warmup_recency_c${cache}_${TS}" \
        --method es_recency \
        --run_config recency_es_llama32_1b_tpu \
        --cache_size "${cache}"
done

# 3) NAMM runs across cache sizes
for cache in ${CACHE_SIZES}; do
    run_warmup \
        "namm_c${cache}" \
        --run_name "warmup_namm_c${cache}_${TS}" \
        --method es_namm \
        --run_config namm_bam_i1_llama32_1b_tpu \
        --namm_checkpoint "${NAMM_CKPT}" \
        --cache_size "${cache}"
done

echo
echo "Warmup complete."
