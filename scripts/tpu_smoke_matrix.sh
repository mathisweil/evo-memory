#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

if [[ "${PJRT_DEVICE:-}" != "TPU" ]]; then
    echo "ERROR: PJRT_DEVICE=TPU is required. Run: source setup/activate_tpu.sh"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required"
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

NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
POP_SIZE="${POP_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-18}"
CACHE_SIZE="${CACHE_SIZE:-1024}"
FILTER_BY_LENGTH="${FILTER_BY_LENGTH:-32768}"
RUN_EVAL="${RUN_EVAL:-1}"
GCS_MODE="${GCS_MODE:-local}"
RUN_PREFIX="${RUN_PREFIX:-smoke}"

if [[ "${RUN_EVAL}" != "0" && "${RUN_EVAL}" != "1" ]]; then
    echo "ERROR: RUN_EVAL must be 0 or 1"
    exit 1
fi

case "${GCS_MODE}" in
    local)
        GCS_ARGS=(--no-gcs)
        ;;
    gcs)
        GCS_ARGS=(--gcs)
        ;;
    *)
        echo "ERROR: GCS_MODE must be one of: local, gcs"
        exit 1
        ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
SMOKE_DIR="${SMOKE_DIR:-${REPO_DIR}/experiments/smoke_matrix/${TS}}"
mkdir -p "${SMOKE_DIR}"
SUMMARY_TSV="${SMOKE_DIR}/summary.tsv"
SUMMARY_JSON="${SMOKE_DIR}/summary.json"

printf 'method\tmethod_status\ttrain_status\teval_status\trun_name\trun_dir\tcheckpoint\terror\n' > "${SUMMARY_TSV}"

run_cmd_with_log() {
    local log_path="$1"
    shift

    echo "Command: $*"
    set +e
    "$@" 2>&1 | tee "${log_path}"
    local cmd_status=${PIPESTATUS[0]}
    set -e
    return "${cmd_status}"
}

append_summary() {
    local method="$1"
    local method_status="$2"
    local train_status="$3"
    local eval_status="$4"
    local run_name="$5"
    local run_dir="$6"
    local checkpoint="$7"
    local error_msg="$8"

    local clean_error="${error_msg//$'\t'/ }"
    clean_error="${clean_error//$'\n'/ ; }"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${method}" "${method_status}" "${train_status}" "${eval_status}" \
        "${run_name}" "${run_dir}" "${checkpoint}" "${clean_error}" \
        >> "${SUMMARY_TSV}"
}

run_smoke_method() {
    local method="$1"
    local run_config="$2"
    local needs_namm="$3"
    local add_cache_size="$4"

    local run_name="${RUN_PREFIX}_${method}_${TS}"
    local train_log="${SMOKE_DIR}/${method}_train.log"
    local eval_log="${SMOKE_DIR}/${method}_eval.log"

    local train_status="failed"
    local eval_status="not_run"
    local method_status="failed"
    local run_dir=""
    local checkpoint=""
    local error_msg=""

    local train_args=(
        python3 scripts/run_es.py
        --run_name "${run_name}"
        --method "${method}"
        --run_config "${run_config}"
        --num_iterations "${NUM_ITERATIONS}"
        --population_size "${POP_SIZE}"
        --mini_batch_size "${BATCH_SIZE}"
        --batch_size "${BATCH_SIZE}"
        --checkpoint_every 0
        --filter_by_length "${FILTER_BY_LENGTH}"
    )
    train_args+=("${GCS_ARGS[@]}")

    if [[ "${add_cache_size}" == "1" ]]; then
        train_args+=(--cache_size "${CACHE_SIZE}")
    fi

    if [[ "${needs_namm}" == "1" ]]; then
        if [[ -z "${NAMM_CKPT:-}" ]]; then
            error_msg="NAMM_CKPT is required for ${method}"
            append_summary "${method}" "failed" "failed" "not_run" \
                "${run_name}" "" "" "${error_msg}"
            return 1
        fi
        if [[ ! -f "${NAMM_CKPT}" ]]; then
            error_msg="NAMM checkpoint not found: ${NAMM_CKPT}"
            append_summary "${method}" "failed" "failed" "not_run" \
                "${run_name}" "" "" "${error_msg}"
            return 1
        fi
        train_args+=(--namm_checkpoint "${NAMM_CKPT}")
    fi

    echo "========================================================================"
    echo "[${method}] Training smoke run"
    echo "========================================================================"

    if run_cmd_with_log "${train_log}" "${train_args[@]}"; then
        train_status="passed"
    else
        error_msg="training command failed"
    fi

    checkpoint="$(find "${REPO_DIR}/experiments" -type f -path "*/${method}/${run_name}/checkpoints/es_checkpoint_final.pt" | sort | tail -n1 || true)"
    if [[ -n "${checkpoint}" ]]; then
        run_dir="${checkpoint%/checkpoints/es_checkpoint_final.pt}"
    elif [[ "${train_status}" == "passed" ]]; then
        train_status="failed"
        error_msg="training finished but no final checkpoint was found"
    fi

    if [[ "${train_status}" == "passed" && "${RUN_EVAL}" == "1" ]]; then
        local eval_args=(
            python3 scripts/run_eval.py
            --es_checkpoint "${checkpoint}"
            --run_config "${run_config}"
            --batch_size "${BATCH_SIZE}"
            --output_dir "${run_dir}"
        )
        if [[ "${add_cache_size}" == "1" ]]; then
            eval_args+=(--cache_size "${CACHE_SIZE}")
        fi
        if [[ "${needs_namm}" == "1" ]]; then
            eval_args+=(--namm_checkpoint "${NAMM_CKPT}")
        fi

        echo "========================================================================"
        echo "[${method}] Evaluation smoke run"
        echo "========================================================================"

        if run_cmd_with_log "${eval_log}" "${eval_args[@]}"; then
            eval_status="passed"
        else
            eval_status="failed"
            error_msg="evaluation command failed"
        fi
    elif [[ "${RUN_EVAL}" == "0" ]]; then
        eval_status="skipped"
    fi

    if [[ "${train_status}" == "passed" && ( "${eval_status}" == "passed" || "${eval_status}" == "skipped" ) ]]; then
        method_status="passed"
    fi

    append_summary "${method}" "${method_status}" "${train_status}" "${eval_status}" \
        "${run_name}" "${run_dir}" "${checkpoint}" "${error_msg}"

    if [[ "${method_status}" == "passed" ]]; then
        return 0
    fi
    return 1
}

failed_methods=0

echo "Smoke matrix output dir: ${SMOKE_DIR}"
echo "NUM_ITERATIONS=${NUM_ITERATIONS}, POP_SIZE=${POP_SIZE}, BATCH_SIZE=${BATCH_SIZE}"
echo "CACHE_SIZE=${CACHE_SIZE}, FILTER_BY_LENGTH=${FILTER_BY_LENGTH}, RUN_EVAL=${RUN_EVAL}"
echo "GCS_MODE=${GCS_MODE}"

resolve_namm_ckpt

if ! run_smoke_method "es_only" "full_cache_es_llama32_1b_tpu" 0 0; then
    failed_methods=$((failed_methods + 1))
fi
if ! run_smoke_method "es_recency" "recency_es_llama32_1b_tpu" 0 1; then
    failed_methods=$((failed_methods + 1))
fi
if ! run_smoke_method "es_namm" "namm_bam_i1_llama32_1b_tpu" 1 1; then
    failed_methods=$((failed_methods + 1))
fi

python3 - "${SUMMARY_TSV}" "${SUMMARY_JSON}" "${SMOKE_DIR}" "${failed_methods}" <<'PY'
import csv
import json
import sys
from datetime import datetime

summary_tsv, summary_json, smoke_dir, failed_methods = sys.argv[1:]

rows = []
with open(summary_tsv, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

payload = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "output_dir": smoke_dir,
    "failed_methods": int(failed_methods),
    "results": rows,
}

with open(summary_json, "w") as f:
    json.dump(payload, f, indent=2)
PY

echo "========================================================================"
echo "TPU smoke matrix summary"
echo "========================================================================"
cat "${SUMMARY_TSV}"
echo
echo "Summary JSON: ${SUMMARY_JSON}"

if [[ "${failed_methods}" -gt 0 ]]; then
    echo "Smoke matrix finished with ${failed_methods} failed method(s)."
    exit 1
fi

echo "Smoke matrix completed successfully."
