#!/usr/bin/env bash
# Master script: runs all experiments in dependency order (FAIR-01).
# Usage: bash scripts/run_all_experiments.sh experiments/experiment_5
set -euo pipefail

# ── Args ─────────────────────────────────────────────────────────────────
EXPERIMENT_DIR="${1:?Usage: $0 <experiment_dir>}"
mkdir -p "$EXPERIMENT_DIR"
LOG="$EXPERIMENT_DIR/run_all.log"
exec > >(tee -a "$LOG") 2>&1

PASSED=()
FAILED=()
SKIPPED=()

run_step() {
    local name="$1"; shift
    echo ""
    echo "================================================================"
    echo "  [$name] START  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    if "$@"; then
        echo "  [$name] PASSED"
        PASSED+=("$name")
    else
        echo "  [$name] FAILED (exit $?)"
        FAILED+=("$name")
        return 1
    fi
}

skip_step() {
    echo "  [$1] SKIPPED — dependency failed"
    SKIPPED+=("$1")
}

# ── Smoke tests ──────────────────────────────────────────────────────────
echo "Running smoke tests..."

run_step "smoke_eval" python scripts/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --num_samples 3 \
    --output_dir "$EXPERIMENT_DIR/smoke_eval" || {
    echo "ABORT: smoke test failed. Fix before running full experiments."
    exit 1
}

# ── Tier 1: Baselines (no training, parallel-safe) ───────────────────────
echo ""
echo "======== TIER 1: BASELINES ========"

run_step "B0" python scripts/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --output_dir "$EXPERIMENT_DIR/baseline" || true

run_step "B1" python scripts/run_eval.py \
    --run_config recency_baseline_llama32_1b \
    --cache_size 1024 \
    --output_dir "$EXPERIMENT_DIR/es_recency/b1_recency" || true

# ── Tier 2: Main conditions ─────────────────────────────────────────────
echo ""
echo "======== TIER 2: MAIN CONDITIONS ========"

# M1 — LoRA rank sweep (r=4, r=8, r=16)
M1_OK=true

run_step "M1-r4" python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r4 \
    --lora_rank 4 --lora_alpha 8 || M1_OK=false

run_step "M1-r8" python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r8 || M1_OK=false

run_step "M1-r16" python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r16 \
    --lora_rank 16 --lora_alpha 32 || M1_OK=false

# M2 — Standalone NAMM (CMA-ES)
M2_OK=true
M2_CKPT=""

run_step "M2" python scripts/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b_5t' \
    wandb_run_name=m2_namm_standalone \
    wandb_group_name=main_conditions \
    seed=1337 || M2_OK=false

# Locate M2 checkpoint (latest Hydra output)
if $M2_OK; then
    M2_CKPT=$(find outputs/ -name "ckpt.pt" -path "*/1337/*" -newer "$LOG" 2>/dev/null | sort | tail -1)
    if [ -z "$M2_CKPT" ]; then
        echo "WARNING: M2 completed but could not find checkpoint. M3 will be skipped."
        M2_OK=false
    else
        echo "  M2 checkpoint: $M2_CKPT"
    fi
fi

# M3 — LoRA + frozen NAMM (requires M2 checkpoint)
if $M2_OK; then
    run_step "M3" python scripts/run_lora.py \
        --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
        --run_name m3_lora_frozen_namm \
        --namm_checkpoint "$M2_CKPT" || true
else
    skip_step "M3"
fi

# M4 — Joint LoRA + NAMM (no dependency on M2)
M4_OK=true

run_step "M4" python scripts/run_joint.py \
    --config scripts/configs/joint_lora_m4_5t.yaml \
    --run_name m4_joint_lora \
    --adapter_type lora || M4_OK=false

# ── Tier 3: Ablations ───────────────────────────────────────────────────
echo ""
echo "======== TIER 3: ABLATIONS ========"

# A4 — NAMM disabled at eval on M4 checkpoint
if $M4_OK; then
    M4_ADAPTER="$EXPERIMENT_DIR/joint_lora/m4_joint_lora/adapter/stage_1/"
    M4_NAMM="$EXPERIMENT_DIR/joint_lora/m4_joint_lora/namm/latest.pt"

    run_step "A4-on" python scripts/run_eval.py \
        --es_checkpoint "$M4_ADAPTER" \
        --namm_checkpoint "$M4_NAMM" \
        --cache_size 1024 \
        --output_dir "$EXPERIMENT_DIR/../ablations/a4_modularity/m4_namm_on" || true

    run_step "A4-off" python scripts/run_eval.py \
        --es_checkpoint "$M4_ADAPTER" \
        --output_dir "$EXPERIMENT_DIR/../ablations/a4_modularity/m4_namm_off" || true
else
    skip_step "A4-on"
    skip_step "A4-off"
fi

# ── Report generation ────────────────────────────────────────────────────
echo ""
echo "======== REPORT ========"

run_step "report" python scripts/generate_report.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --output "$EXPERIMENT_DIR/paper_results.csv" || true

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo "  PASSED  (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "  FAILED  (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "  SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo ""
echo "  Log: $LOG"
echo "================================================================"

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
