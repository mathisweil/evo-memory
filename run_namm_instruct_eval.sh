#!/bin/bash
# Evaluate the NAMM-Instruct checkpoint (50 iters, 3 tasks) at cache_size=1024.
set -e

CKPT="/cs/student/project_msc/2025/csml/gmaralla/NAMM_implementation/exp_local/memory_evolution_hf/Llama-3.2-1B-Instruct/NAMM/attn-spec-norm/bam/binary-1024cs/lb3subset-eval-cma-es-p8-rMeanTrue-shared-8pop-16qs-256fixDel-llama32-1b-instruct-stage1/1337/ckpt.pt"

echo "=== NAMM Instruct eval (cache_size=1024) ==="
python main.py 'run@_global_=namm_bam_eval_llama32_1b_instruct' init_from="$CKPT" seed=1337

echo "=== Done ==="
