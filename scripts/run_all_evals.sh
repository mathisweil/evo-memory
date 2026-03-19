#!/bin/bash
# Run all 8 chat-templated evals, then launch NAMM instruct training.
# Usage: tmux → bash run_all_evals_then_namm.sh
set -e

CKPT="results/m1_sft/1337/ckpt.pt"
REC_CFG="recency_baseline_llama32_1b_instruct"
GROUP="Llama-3.2-1B-Instruct/chat-template-evals"

echo "=== Phase 1: Base instruct evals ==="
python run_eval.py --method base --seed 1337 --wandb-group "$GROUP"
python run_eval.py --method base --seed 1337 --run-config $REC_CFG --cache-size 4096 --wandb-group "$GROUP"
python run_eval.py --method base --seed 1337 --run-config $REC_CFG --cache-size 2048 --wandb-group "$GROUP"
python run_eval.py --method base --seed 1337 --run-config $REC_CFG --cache-size 1024 --wandb-group "$GROUP"

echo "=== Phase 2: LoRA SFT evals ==="
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --wandb-group "$GROUP"
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --run-config $REC_CFG --cache-size 4096 --wandb-group "$GROUP"
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --run-config $REC_CFG --cache-size 2048 --wandb-group "$GROUP"
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --run-config $REC_CFG --cache-size 1024 --wandb-group "$GROUP"

echo "=== Phase 3: NAMM instruct training (50 iters, 3 tasks) ==="
python main.py 'run@_global_=namm_bam_i1_llama32_1b_instruct' seed=1337

echo "=== All done ==="
