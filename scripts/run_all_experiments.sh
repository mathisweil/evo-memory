#!/usr/bin/env bash
# run_all_experiments.sh — Execute all remaining FAIR-01 experiments in
# dependency order, with smoke tests, per-step error handling, auto-eval,
# and a summary table at the end.
#
# Phases:
#   0: Smoke tests (abort on failure of any)
#   1: Download GCS checkpoints (M1, M2 cs1024) needed as dependencies
#   2: M1_recency fix (standalone — Option A from docs/m1_recency_investigation.md)
#   3: M3-matched cs1024 (priority 1; needs M2 cs1024 checkpoint)
#   4: M3-matched cs2048 (priority 2; needs M2 cs2048 checkpoint)
#   5: M1 rank sweep — r=4, r=16 (priorities 3-4; no dependencies)
#   6: M4 joint LoRA + NAMM (priority 5; no dependencies)
#   7: A4 modularity ablation on M4 (priorities 6-7; depends on phase 6)
#   8: Summary
#
# Usage:
#   bash scripts/run_all_experiments.sh                  # full run
#   bash scripts/run_all_experiments.sh --smoke-only     # smoke then exit
#   bash scripts/run_all_experiments.sh --skip-smoke     # skip phase 0
#   bash scripts/run_all_experiments.sh --phase N        # start from phase N
#
# Prerequisites:
#   - GCS credentials: gcloud auth login
#   - WandB credentials: wandb login (or WANDB_API_KEY)
#   - GPU with >= 8GB VRAM
#   - Python venv at .venv/ or venv/
#
# Per CLAUDE.md global rules: the script does NOT abort on per-step failures
# (phases 2-7). One failed run does not block the rest. Smoke-test failures
# in phase 0 still abort — they indicate a pipeline regression worth fixing
# before burning hours of GPU time.

set -uo pipefail  # note: NO -e — per-step failures are captured, not fatal

# ── Configuration ──────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACTS_DIR="experiment_artifacts/gcs"
EVAL_RESULTS_DIR="eval_results"

# GCS-hosted checkpoints (populated by scripts/download_artifacts.py).
# M1 is needed only for the M1_recency fix eval. M2 cs1024/cs2048 are
# needed as the frozen NAMM checkpoints for M3-matched reruns.
M1_CKPT="${ARTIFACTS_DIR}/M1/best_ckpt.pt"
M2_CS1024_CKPT="${ARTIFACTS_DIR}/M2_cs1024/latest.pt"
M2_CS2048_CKPT="${ARTIFACTS_DIR}/M2_cs2048/latest.pt"

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
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0 ;;
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
    # Run a named step and capture pass/fail into the summary arrays.
    # Returns the underlying command's exit code so callers can gate
    # dependent steps (e.g., auto-eval only if training succeeded).
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
    fi
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    log "FAIL:  $name (exit $rc after ${elapsed}s)"
    FAILED+=("$name")
    return "$rc"
}

skip_step() {
    log "SKIP: $1 — $2"
    SKIPPED+=("$1")
}

find_run_dir() {
    # Return the path of the most recently modified experiment run matching
    # the given run_name, or empty if none found.
    find experiments/ -maxdepth 4 -type d -name "$1" 2>/dev/null \
        | head -1
}

# ── Phase 0: Smoke tests ──────────────────────────────────────────────────
if [[ $START_PHASE -le 0 ]] && [[ "$SKIP_SMOKE" == "false" ]]; then
    phase_header 0 "Smoke tests"

    run_step "smoke_m1" \
        $PY scripts/run_lora.py \
            --config scripts/configs/m1_lora_5t.yaml \
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
            --config scripts/configs/m4_joint_lora_5t.yaml \
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
# M1_CKPT is required for the M1_recency fix eval (phase 2).
# M2 checkpoints are required for M3-matched training (phases 3-4).
# All three are nominally in GCS; download_artifacts.py resolves the canonical
# paths. Missing any of them is NOT fatal for this script — we just skip the
# dependent phases with a recorded SKIP.
if [[ $START_PHASE -le 1 ]]; then
    phase_header 1 "Download GCS checkpoints"

    if [ -f "$M1_CKPT" ] && [ -f "$M2_CS1024_CKPT" ] && [ -f "$M2_CS2048_CKPT" ]; then
        log "All required checkpoints already present. Skipping download."
    else
        run_step "download_artifacts" \
            $PY scripts/download_artifacts.py || true
    fi

    for ckpt in "$M1_CKPT" "$M2_CS1024_CKPT" "$M2_CS2048_CKPT"; do
        if [ ! -f "$ckpt" ]; then
            log "WARNING: Missing checkpoint: $ckpt — dependent phases will be skipped."
        fi
    done
fi

# ── Phase 2: M1_recency fix ───────────────────────────────────────────────
# Ref: docs/m1_recency_investigation.md. The prior M1_recency eval ran NAMM
# with untrained init params (scoring_initializer=0), not StreamingLLM
# recency. Option A in that doc is the correct re-run.
if [[ $START_PHASE -le 2 ]]; then
    phase_header 2 "M1_recency fix (LoRA M1 + classic recency eviction)"

    if [ ! -f "$M1_CKPT" ]; then
        skip_step "m1_recency_fix" "M1 checkpoint not found at $M1_CKPT"
    else
        for cs in 1024 2048; do
            run_step "m1_recency_cs${cs}_fixed" \
                $PY scripts/eval_namm_splits.py \
                    --lora_checkpoint "$M1_CKPT" \
                    --use_classic_recency \
                    --cache_size "$cs" \
                    --batch_size 1 \
                    --splits test extended_test \
                    --run_label "m1_recency_cs${cs}_fixed" \
                    --output_dir "${EVAL_RESULTS_DIR}/m1_recency_cs${cs}_fixed" || true
        done
    fi
fi

# ── Phase 3: M3-matched cs1024 ────────────────────────────────────────────
# Priority 1 per claude_code_remaining_work.md §5. The M3 config
# (m3_lora_frozen_namm_5t.yaml) is already corrected: lr=5e-5, dropout=0.1,
# early_stopping_patience=20. No CLI overrides required; just --run_name
# and the M2 checkpoint. See docs/m3_rerun_plan.md.
M3_CS1024_OK=false
if [[ $START_PHASE -le 3 ]]; then
    phase_header 3 "M3-matched cs1024 (LoRA + frozen M2 NAMM, matches M1 hyperparams)"

    if [ ! -f "$M2_CS1024_CKPT" ]; then
        skip_step "m3_cs1024_matched" "M2 cs1024 checkpoint not found at $M2_CS1024_CKPT"
    else
        if run_step "m3_cs1024_matched" \
            $PY scripts/run_lora.py \
                --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
                --run_name m3_cs1024_matched \
                --namm_checkpoint "$M2_CS1024_CKPT" \
                --wandb_group_name m3_matched \
                --skip_baseline_eval; then
            M3_CS1024_OK=true
        fi

        if $M3_CS1024_OK; then
            M3_DIR=$(find_run_dir "m3_cs1024_matched")
            if [ -n "$M3_DIR" ] && [ -f "$M3_DIR/best_ckpt.pt" ]; then
                run_step "m3_cs1024_matched_eval" \
                    $PY scripts/run_eval.py \
                        --config scripts/configs/eval_main_table.yaml \
                        --es_checkpoint "$M3_DIR/best_ckpt.pt" \
                        --namm_checkpoint "$M2_CS1024_CKPT" \
                        --cache_size 1024 \
                        --output_dir "$M3_DIR/eval" || true
            else
                skip_step "m3_cs1024_matched_eval" "best_ckpt.pt not found under $M3_DIR"
            fi
        fi
    fi
fi

# ── Phase 4: M3-matched cs2048 ────────────────────────────────────────────
# Priority 2. Cache-size sweep companion to phase 3.
if [[ $START_PHASE -le 4 ]]; then
    phase_header 4 "M3-matched cs2048"

    if [ ! -f "$M2_CS2048_CKPT" ]; then
        skip_step "m3_cs2048_matched" "M2 cs2048 checkpoint not found at $M2_CS2048_CKPT"
    else
        M3_CS2048_OK=false
        if run_step "m3_cs2048_matched" \
            $PY scripts/run_lora.py \
                --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
                --run_name m3_cs2048_matched \
                --namm_checkpoint "$M2_CS2048_CKPT" \
                --cache_size 2048 \
                --wandb_group_name m3_matched \
                --skip_baseline_eval; then
            M3_CS2048_OK=true
        fi

        if $M3_CS2048_OK; then
            M3_DIR=$(find_run_dir "m3_cs2048_matched")
            if [ -n "$M3_DIR" ] && [ -f "$M3_DIR/best_ckpt.pt" ]; then
                run_step "m3_cs2048_matched_eval" \
                    $PY scripts/run_eval.py \
                        --config scripts/configs/eval_main_table.yaml \
                        --es_checkpoint "$M3_DIR/best_ckpt.pt" \
                        --namm_checkpoint "$M2_CS2048_CKPT" \
                        --cache_size 2048 \
                        --output_dir "$M3_DIR/eval" || true
            else
                skip_step "m3_cs2048_matched_eval" "best_ckpt.pt not found under $M3_DIR"
            fi
        fi
    fi
fi

# ── Phase 5: M1 rank sweep (r=4, r=16) ────────────────────────────────────
# Priorities 3-4 — the A1 rank-sweep ablation. alpha=2*rank per FAIR-01.
if [[ $START_PHASE -le 5 ]]; then
    phase_header 5 "M1 rank sweep (r=4, r=16)"

    for pair in "4 8" "16 32"; do
        read -r rank alpha <<< "$pair"
        run_name="m1_r${rank}"
        ok=false
        if run_step "$run_name" \
            $PY scripts/run_lora.py \
                --config scripts/configs/m1_lora_5t.yaml \
                --run_name "$run_name" \
                --lora_rank "$rank" \
                --lora_alpha "$alpha" \
                --wandb_group_name m1_rank_sweep \
                --skip_baseline_eval; then
            ok=true
        fi

        if $ok; then
            rd=$(find_run_dir "$run_name")
            if [ -n "$rd" ] && [ -f "$rd/best_ckpt.pt" ]; then
                # M1 eval is full-cache (no NAMM) at cache_size=1024 per FAIR-01.
                run_step "${run_name}_eval" \
                    $PY scripts/run_eval.py \
                        --config scripts/configs/eval_main_table.yaml \
                        --es_checkpoint "$rd/best_ckpt.pt" \
                        --cache_size 1024 \
                        --output_dir "$rd/eval" || true
            else
                skip_step "${run_name}_eval" "best_ckpt.pt not found under $rd"
            fi
        fi
    done
fi

# ── Phase 6: M4 joint LoRA + NAMM ─────────────────────────────────────────
# Priority 5. All M4 hyperparameters are pinned in m4_joint_lora_5t.yaml
# (num_outer_loops=3, 67 NAMM + 50 LoRA per stage, early_stopping_patience=20,
# lora_eval_interval=14, LR and dropout matched to M1). No CLI overrides.
M4_OK=false
M4_DIR=""
if [[ $START_PHASE -le 6 ]]; then
    phase_header 6 "M4 joint LoRA + NAMM"

    if run_step "m4_joint_lora" \
        $PY scripts/run_joint.py \
            --config scripts/configs/m4_joint_lora_5t.yaml \
            --run_name m4_joint_lora; then
        M4_OK=true
    fi

    if $M4_OK; then
        M4_DIR=$(find_run_dir "m4_joint_lora")
        # With num_outer_loops=3 the final adapter stage is stage_2 (0-indexed).
        M4_ADAPTER="${M4_DIR}/adapter/stage_2/best_ckpt.pt"
        M4_NAMM="${M4_DIR}/namm/latest.pt"

        if [ -f "$M4_ADAPTER" ] && [ -f "$M4_NAMM" ]; then
            log "Auto-eval M4 (NAMM on, cs=1024)..."
            run_step "m4_eval_namm_on" \
                $PY scripts/run_eval.py \
                    --config scripts/configs/eval_main_table.yaml \
                    --es_checkpoint "$M4_ADAPTER" \
                    --namm_checkpoint "$M4_NAMM" \
                    --cache_size 1024 \
                    --output_dir "${M4_DIR}/eval" || true
        else
            log "WARNING: M4 final-stage checkpoints not at expected paths:"
            log "  adapter: $M4_ADAPTER"
            log "  namm:    $M4_NAMM"
        fi
    fi
fi

# ── Phase 7: A4 modularity ablation (M4 NAMM on vs off) ───────────────────
# Priorities 6-7. Paired comparison on the same M4 checkpoint.
if [[ $START_PHASE -le 7 ]]; then
    phase_header 7 "A4 modularity ablation (M4 NAMM on/off)"

    if [ -z "$M4_DIR" ]; then
        M4_DIR=$(find_run_dir "m4_joint_lora")
    fi

    if [ -z "$M4_DIR" ]; then
        skip_step "a4_m4_namm_off" "M4 experiment directory not found"
        skip_step "a4_m4_namm_on"  "M4 experiment directory not found"
    else
        M4_ADAPTER="${M4_DIR}/adapter/stage_2/best_ckpt.pt"
        M4_NAMM="${M4_DIR}/namm/latest.pt"

        if [ ! -f "$M4_ADAPTER" ]; then
            skip_step "a4_m4_namm_off" "M4 adapter checkpoint not found"
            skip_step "a4_m4_namm_on"  "M4 adapter checkpoint not found"
        else
            # A4-off: M4 LoRA adapter alone, full KV cache (no eviction).
            # MUST omit --namm_checkpoint entirely per .claude/rules/eval.md.
            run_step "a4_m4_namm_off" \
                $PY scripts/run_eval.py \
                    --config scripts/configs/eval_main_table.yaml \
                    --run_config full_cache_baseline_llama32_1b \
                    --es_checkpoint "$M4_ADAPTER" \
                    --output_dir "experiments/ablations/a4_modularity/m4_namm_off" || true

            # A4-on: M4 LoRA + M4 NAMM, cs=1024 (the paired comparison arm).
            if [ -f "$M4_NAMM" ]; then
                run_step "a4_m4_namm_on" \
                    $PY scripts/run_eval.py \
                        --config scripts/configs/eval_main_table.yaml \
                        --es_checkpoint "$M4_ADAPTER" \
                        --namm_checkpoint "$M4_NAMM" \
                        --cache_size 1024 \
                        --output_dir "experiments/ablations/a4_modularity/m4_namm_on" || true
            else
                skip_step "a4_m4_namm_on" "M4 NAMM checkpoint not found"
            fi
        fi
    fi
fi

# ── Phase 8: Summary ──────────────────────────────────────────────────────
phase_header 8 "Summary"

{
    echo "Experiment Run Summary"
    echo "======================"
    echo "Date: $(date)"
    echo ""
    printf "%-32s %s\n" "STEP" "STATUS"
    printf "%-32s %s\n" "--------------------------------" "------"
    for s in "${PASSED[@]+"${PASSED[@]}"}"; do
        printf "%-32s %s\n" "$s" "PASS"
    done
    for s in "${FAILED[@]+"${FAILED[@]}"}"; do
        printf "%-32s %s\n" "$s" "FAIL"
    done
    for s in "${SKIPPED[@]+"${SKIPPED[@]}"}"; do
        printf "%-32s %s\n" "$s" "SKIP"
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
