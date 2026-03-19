#!/bin/bash
# Evaluate the NAMM-Instruct checkpoint at cache_size=1024.
# Usage: NAMM_CKPT=/path/to/ckpt.pt bash run_namm_instruct_eval.sh
set -e

CKPT="${NAMM_CKPT:?Error: NAMM_CKPT env var must be set to the checkpoint path}"

echo "=== NAMM Instruct eval (cache_size=1024) ==="
python main.py 'run@_global_=namm_bam_eval_llama32_1b_instruct' init_from="$CKPT" seed=1337

echo "=== Done ==="
