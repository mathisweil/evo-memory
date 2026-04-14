#!/usr/bin/env bash
# run_all_experiments.sh — Execute all remaining FAIR-01 experiments.
#
# Based on the audit in docs/experiment_recommendations.md, this script runs
# experiments in dependency order with smoke tests, error handling, auto-eval,
# and a summary table.
#
# Phases:
#   0: Smoke tests (abort early if pipeline is broken)
#   1: Download GCS checkpoints (needed for resume)
#   2: Must-fix (M1 resume, M3/cs1024 resume, M1_recency re-eval)
#   3: M4 joint LoRA + NAMM (paper's main contribution)
#   4: Should-run (M3 LR ablation, M1-r4, M1-r16)
#   5: A4 modularity ablation on M4 checkpoints
#   6: Summary
#
# Usage:
#   bash scripts/run_all_experiments.sh              # full run
#   bash scripts/run_all_experiments.sh --smoke-only # smoke tests then exit
#   bash scripts/run_all_experiments.sh --skip-smoke # skip smoke tests
#   bash scripts/run_all_experiments.sh --phase N    # start from phase N
#
# Prerequisites:
#   - GCS credentials: gcloud auth login
#   - WandB credentials: wandb login (or WANDB_API_KEY)
#   - GPU with >= 8GB VRAM
#   - Python venv at .venv/ or venv/
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACTS_DIR="experiment_artifacts/gcs"
EVAL_RESULTS_DIR="eval_results"

# GCS checkpoint local paths (populated by scripts/download_artifacts.py)
M1_CKPT="${ARTIFACTS_DIR}/M1/best_ckpt.pt"
M2_CS1024_CKPT="${ARTIFACTS_DIR}/M2_cs1024/latest.pt"
M3_CS1024_CKPT="${ARTIFACTS_DIR}/M3_cs1024/best_ckpt.pt"

# Detect Python binary
if [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    echo "ERROR: No venv found at .venv/ or venv/. Create one first." >&2
    exit 1
fi

# ── Argument parsing ───────────────────────────────────────────────────────
SMOKE_ONLY=false
SKIP_SMOKE=false
START_PHASE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-only) SMOKE_ONLY=true; shift ;;
        --skip-smoke) SKIP_SMOKE=true; shift ;;
        --phase) START_PHASE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Logging ────────────────────────────────────────────────────────────────
LOG_FILE="scripts/run_all_experiments.log"
SUMMARY_FILE="scripts/experiment_summary.txt"
: > "$LOG_FILE"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

phase_header() {
    log "============================================================"
    log "PHASE $1: $2"
    log "============================================================"
}

PASSED=()
FAILED=()
SKIPPED=()

run_step() {
    local name="$1"; shift
    log "START: $name"
    log "  CMD: $*"
    local t0
    t0=$(date +%s)
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        local elapsed=$(( $(date +%s) - t0 ))
        log "DONE:  $name (${elapsed}s)"
        PASSED+=("$name")
        return 0
    else
        local rc=$?
        local elapsed=$(( $(date +%s) - t0 ))
        log "FAIL:  $name (exit $rc after ${elapsed}s)"
        FAILED+=("$name")
        return $rc
    fi
}

skip_step() {
    log "SKIP: $1 — $2"
    SKIPPED+=("$1")
}

# ── Phase 0: Smoke tests ──────────────────────────────────────────────────
if [[ $START_PHASE -le 0 ]] && [[ "$SKIP_SMOKE" == "false" ]]; then
    phase_header 0 "Smoke tests"

    run_step "smoke_m1" \
        $PY scripts/run_lora.py \
            --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
            --run_name smoke_m1 \
            --num_epochs 1 \
            --eval_interval 999 \
            --no-gcs \
            --wandb_log false \
            --skip_baseline_eval || {
        log "ABORT: M1 smoke test failed."
        exit 1
    }

    run_step "smoke_m4_joint" \
        $PY scripts/run_joint.py \
            --config scripts/configs/joint_lora_m4_5t.yaml \
            --run_name smoke_m4_joint \
            --num_outer_loops 1 \
            --namm_iterations_per_stage 2 \
            --lora_epochs_per_stage 1 \
            --population_size 2 \
            --mini_batch_size 2 || {
        log "ABORT: M4 smoke test failed."
        exit 1
    }

    run_step "smoke_eval" \
        $PY scripts/run_eval.py \
            --run_config full_cache_baseline_llama32_1b \
            --num_samples 3 || {
        log "ABORT: eval smoke test failed."
        exit 1
    }

    log "All smoke tests passed."
    if [[ "$SMOKE_ONLY" == "true" ]]; then
        log "Smoke-only mode: exiting."
        exit 0
    fi
fi

# ── Phase 1: Download GCS checkpoints ─────────────────────────────────────
if [[ $START_PHASE -le 1 ]]; then
    phase_header 1 "Download GCS checkpoints"

    if [ -f "$M1_CKPT" ] && [ -f "$M2_CS1024_CKPT" ] && [ -f "$M3_CS1024_CKPT" ]; then
        log "All required checkpoints already present. Skipping download."
    else
        run_step "download_artifacts" \
            $PY scripts/download_artifacts.py
    fi

    for ckpt_path in "$M1_CKPT" "$M2_CS1024_CKPT" "$M3_CS1024_CKPT"; do
        if [ ! -f "$ckpt_path" ]; then
            log "ERROR: Required checkpoint not found: $ckpt_path"
            log "Run 'python scripts/download_artifacts.py' manually."
            exit 1
        fi
    done
    log "All required checkpoints verified."
fi

# ── Phase 2: Must-fix runs ────────────────────────────────────────────────
# Ref: docs/experiment_recommendations.md section 5a
if [[ $START_PHASE -le 2 ]]; then
    phase_header 2 "Must-fix runs"

    # 2a. Resume M1 training (epoch 28 -> 150)
    # Finding: M1 only reached 18.7% of training. Test F1=31.14 is unreliable.
    log "2a. Resume M1 LoRA (r=8) from epoch 28..."
    run_step "m1_r8_resume" \
        $PY scripts/run_lora.py \
            --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
            --run_name m1_r8_resume \
            --resume_checkpoint "$M1_CKPT" \
            --skip_baseline_eval || true

    # 2b. Resume M3/cs1024 training (epoch 25 -> 150)
    # Finding: M3/cs1024 only reached 16.7% of training.
    log "2b. Resume M3/cs1024 LoRA from epoch 25..."
    run_step "m3_cs1024_resume" \
        $PY scripts/run_lora.py \
            --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
            --run_name m3_lora_frozen_namm_resume \
            --namm_checkpoint "$M2_CS1024_CKPT" \
            --resume_checkpoint "$M3_CS1024_CKPT" \
            --skip_baseline_eval || true

    # 2c. Fix M1_recency eval
    # Finding: M1_recency used random NAMM init (scoring_initializer=0) instead
    # of classic recency eviction. Produced all-zero F1.
    log "2c. Re-run M1_recency with --use_classic_recency..."
    run_step "m1_recency_fix" \
        $PY scripts/eval_namm_splits.py \
            --run_config namm_bam_i1_llama32_1b_5t \
            --lora_checkpoint "$M1_CKPT" \
            --use_classic_recency \
            --cache_size 1024 \
            --batch_size 1 \
            --splits test extended_test \
            --run_label m1_recency_fixed \
            --output_dir "${EVAL_RESULTS_DIR}/lora_m1_recency_cs1024_5t_fixed" || true
fi

# ── Phase 3: M4 Joint LoRA + NAMM ─────────────────────────────────────────
# Ref: docs/experiment_recommendations.md section 5b Priority 1
# Config fixes applied: max_seq_len 3500->7000, lora_dropout 0.0->0.1
M4_OK=false

if [[ $START_PHASE -le 3 ]]; then
    phase_header 3 "M4 Joint LoRA + NAMM"

    if run_step "m4_joint_lora" \
        $PY scripts/run_joint.py \
            --config scripts/configs/joint_lora_m4_5t.yaml \
            --run_name m4_joint_lora; then
        M4_OK=true
    fi

    # Auto-eval M4 if training succeeded
    if $M4_OK; then
        # Stages are 0-indexed; with num_outer_loops=2, final adapter is stage_1
        M4_DIR=$(find experiments/ -maxdepth 3 -type d -name "m4_joint_lora" 2>/dev/null | head -1)
        if [ -n "$M4_DIR" ]; then
            M4_ADAPTER="${M4_DIR}/adapter/stage_1/best_ckpt.pt"
            M4_NAMM="${M4_DIR}/namm/latest.pt"

            if [ -f "$M4_ADAPTER" ] && [ -f "$M4_NAMM" ]; then
                log "Auto-eval M4 with NAMM on..."
                run_step "m4_eval" \
                    $PY scripts/eval_namm_splits.py \
                        --run_config namm_bam_i1_llama32_1b_5t \
                        --lora_checkpoint "$M4_ADAPTER" \
                        --namm_checkpoint "$M4_NAMM" \
                        --cache_size 1024 \
                        --batch_size 1 \
                        --splits test extended_test \
                        --run_label m4_namm_on \
                        --output_dir "${EVAL_RESULTS_DIR}/m4_eval" || true
            else
                log "WARNING: M4 checkpoints not at expected paths."
                log "  adapter: $M4_ADAPTER"
                log "  namm:    $M4_NAMM"
            fi
        fi
    fi
fi

# ── Phase 4: Should-run experiments ────────────────────────────────────────
if [[ $START_PHASE -le 4 ]]; then
    phase_header 4 "Should-run experiments"

    # 4a. M3 with M1-identical hyperparameters (confound ablation)
    # Ref: docs/experiment_recommendations.md section 5b Priority 2
    # Isolates NAMM effect from LR/dropout confound (M3 uses lr=1e-4, M1 uses 5e-5)
    log "4a. M3 LR ablation (lr=5e-5, dropout=0.1 matching M1)..."
    run_step "m3_lr_ablation" \
        $PY scripts/run_lora.py \
            --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
            --run_name m3_lr_ablation \
            --namm_checkpoint "$M2_CS1024_CKPT" \
            --learning_rate 5e-5 \
            --lora_dropout 0.1 \
            --skip_baseline_eval || true

    # Auto-eval M3 LR ablation
    M3_ABL_CKPT=$(find experiments/ -path "*/m3_lr_ablation/*/best_ckpt.pt" 2>/dev/null | head -1)
    if [ -n "$M3_ABL_CKPT" ]; then
        log "Auto-eval M3 LR ablation..."
        run_step "m3_lr_ablation_eval" \
            $PY scripts/eval_namm_splits.py \
                --run_config namm_bam_i1_llama32_1b_5t \
                --lora_checkpoint "$M3_ABL_CKPT" \
                --namm_checkpoint "$M2_CS1024_CKPT" \
                --cache_size 1024 \
                --batch_size 1 \
                --splits test extended_test \
                --run_label m3_lr_ablation \
                --output_dir "${EVAL_RESULTS_DIR}/m3_lr_ablation_eval" || true
    fi

    # 4b. M1-r4 (rank sweep for A1 ablation)
    # Ref: docs/experiment_recommendations.md section 5b Priority 3
    log "4b. M1-r4 (rank=4, alpha=8)..."
    run_step "m1_r4" \
        $PY scripts/run_lora.py \
            --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
            --run_name m1_r4 \
            --lora_rank 4 \
            --lora_alpha 8 \
            --skip_baseline_eval || true

    # 4c. M1-r16 (rank sweep for A1 ablation)
    log "4c. M1-r16 (rank=16, alpha=32)..."
    run_step "m1_r16" \
        $PY scripts/run_lora.py \
            --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
            --run_name m1_r16 \
            --lora_rank 16 \
            --lora_alpha 32 \
            --skip_baseline_eval || true
fi

# ── Phase 5: A4 modularity ablation on M4 checkpoints ─────────────────────
# Ref: docs/experiment_recommendations.md section 5b Priority 4
if [[ $START_PHASE -le 5 ]]; then
    phase_header 5 "A4 modularity ablation (M4 NAMM on/off)"

    M4_DIR=$(find experiments/ -maxdepth 3 -type d -name "m4_joint_lora" 2>/dev/null | head -1)
    if [ -z "$M4_DIR" ]; then
        skip_step "a4_m4_namm_off" "M4 experiment directory not found"
        skip_step "a4_m4_namm_on" "M4 experiment directory not found"
    else
        M4_ADAPTER="${M4_DIR}/adapter/stage_1/best_ckpt.pt"
        M4_NAMM="${M4_DIR}/namm/latest.pt"

        if [ ! -f "$M4_ADAPTER" ]; then
            skip_step "a4_m4_namm_off" "M4 adapter checkpoint not found"
            skip_step "a4_m4_namm_on" "M4 adapter checkpoint not found"
        else
            # A4: NAMM disabled — LoRA only, no eviction
            # No --namm_checkpoint, no --cache_size => full KV cache
            log "A4: M4 LoRA with NAMM disabled (full cache)..."
            run_step "a4_m4_namm_off" \
                $PY scripts/eval_namm_splits.py \
                    --run_config namm_bam_i1_llama32_1b_5t \
                    --lora_checkpoint "$M4_ADAPTER" \
                    --batch_size 1 \
                    --splits test extended_test \
                    --run_label a4_m4_namm_off \
                    --output_dir "${EVAL_RESULTS_DIR}/a4_m4_namm_off" || true

            # A4: NAMM enabled (paired comparison)
            if [ -f "$M4_NAMM" ]; then
                log "A4: M4 LoRA with NAMM enabled..."
                run_step "a4_m4_namm_on" \
                    $PY scripts/eval_namm_splits.py \
                        --run_config namm_bam_i1_llama32_1b_5t \
                        --lora_checkpoint "$M4_ADAPTER" \
                        --namm_checkpoint "$M4_NAMM" \
                        --cache_size 1024 \
                        --batch_size 1 \
                        --splits test extended_test \
                        --run_label a4_m4_namm_on \
                        --output_dir "${EVAL_RESULTS_DIR}/a4_m4_namm_on" || true
            else
                skip_step "a4_m4_namm_on" "M4 NAMM checkpoint not found"
            fi
        fi
    fi
fi

# ── Phase 6: Summary ──────────────────────────────────────────────────────
phase_header 6 "Summary"

{
    echo "Experiment Run Summary"
    echo "======================"
    echo "Date: $(date)"
    echo ""
    printf "%-25s %s\n" "STEP" "STATUS"
    printf "%-25s %s\n" "-------------------------" "------"
    for s in "${PASSED[@]+"${PASSED[@]}"}"; do
        printf "%-25s %s\n" "$s" "PASS"
    done
    for s in "${FAILED[@]+"${FAILED[@]}"}"; do
        printf "%-25s %s\n" "$s" "FAIL"
    done
    for s in "${SKIPPED[@]+"${SKIPPED[@]}"}"; do
        printf "%-25s %s\n" "$s" "SKIP"
    done
    echo ""
    echo "Passed:  ${#PASSED[@]}"
    echo "Failed:  ${#FAILED[@]}"
    echo "Skipped: ${#SKIPPED[@]}"
    echo ""
    echo "Log:     $LOG_FILE"
    echo "Results: $EVAL_RESULTS_DIR/"
} | tee "$SUMMARY_FILE"

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
