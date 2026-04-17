#!/usr/bin/env bash
# Section C smoke test. Runs a tiny dump for each of the three conditions
# (1 prompt per QA task → ~5 prompts × 3 conditions), then runs the
# analysis script. The goal is to catch shape / dtype / key bugs in the
# instrumentation + analyser pipeline end-to-end, not to produce
# publishable numbers.
#
# Requires a machine that can fit the NAMM + LoRA + LLM in GPU memory
# at cache_size=1024. On a laptop with no CUDA, the cs=1024 attention
# capture is too slow to be useful — run this on the cluster.
#
# Defaults point to the laptop-local checkpoint layout; override via env
# vars to run on the cluster (see scripts/run_all_experiments.sh for the
# cluster variant).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO_ROOT/venv/bin/python}"

NAMM_CKPT="${NAMM_CKPT:-$REPO_ROOT/local_folder/final_cs1024/namm_cs1024_maskfix/ckpt.pt}"
M1_LORA="${M1_LORA:-$REPO_ROOT/local_folder/final_cs1024/lora_m1_lr1e4_matched/best_ckpt.pt}"
M4_LORA="${M4_LORA:-$REPO_ROOT/local_folder/final_cs1024/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt}"

RUN_CONFIG="${RUN_CONFIG:-namm_bam_i1_llama32_1b_5t}"
CACHE_SIZE="${CACHE_SIZE:-1024}"
DUMP_ROOT="${DUMP_ROOT:-$REPO_ROOT/eval_results/section_c_smoke}"
PROMPTS_PER_TASK="${PROMPTS_PER_TASK:-1}"

rm -rf "$DUMP_ROOT"
mkdir -p "$DUMP_ROOT/B0" "$DUMP_ROOT/M1" "$DUMP_ROOT/M4"

echo "[smoke] B0 (base + NAMM)"
"$PY" "$REPO_ROOT/scripts/eval_namm_splits.py" \
    --namm_checkpoint "$NAMM_CKPT" \
    --cache_size "$CACHE_SIZE" \
    --run_config "$RUN_CONFIG" \
    --splits test \
    --dump_namm_state "$DUMP_ROOT/B0" \
    --dump_condition_label B0 \
    --dump_max_prompts_per_task "$PROMPTS_PER_TASK"

echo "[smoke] M1 (base + M1-LoRA + NAMM)"
"$PY" "$REPO_ROOT/scripts/eval_namm_splits.py" \
    --namm_checkpoint "$NAMM_CKPT" \
    --lora_checkpoint "$M1_LORA" \
    --cache_size "$CACHE_SIZE" \
    --run_config "$RUN_CONFIG" \
    --splits test \
    --dump_namm_state "$DUMP_ROOT/M1" \
    --dump_condition_label M1 \
    --dump_max_prompts_per_task "$PROMPTS_PER_TASK"

echo "[smoke] M4 (base + M4-LoRA + NAMM)"
"$PY" "$REPO_ROOT/scripts/eval_namm_splits.py" \
    --namm_checkpoint "$NAMM_CKPT" \
    --lora_checkpoint "$M4_LORA" \
    --cache_size "$CACHE_SIZE" \
    --run_config "$RUN_CONFIG" \
    --splits test \
    --dump_namm_state "$DUMP_ROOT/M4" \
    --dump_condition_label M4 \
    --dump_max_prompts_per_task "$PROMPTS_PER_TASK"

echo "[smoke] analyze"
"$PY" "$REPO_ROOT/scripts/analyze_mask_drift.py" \
    --dumps_root "$DUMP_ROOT" \
    --metrics_out "$DUMP_ROOT/section_c_metrics.json" \
    --figures_out "$DUMP_ROOT/figures"

echo "[smoke] OK → $DUMP_ROOT/section_c_metrics.json"
