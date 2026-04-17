#!/bin/bash
# Run a group of val eval variants sequentially on whichever GPU is exposed
# via CUDA_VISIBLE_DEVICES. Writes one results.json per variant under
# eval_results/val_extra/<variant>/.
set -euo pipefail

VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    echo "usage: $0 <variant1> [variant2 ...]" >&2
    echo "  variants: m1_full m1_namm m1_trunc m4_namm a4 b0 m2" >&2
    exit 2
fi

REPO=/cs/student/project_msc/2025/csml/sruppage/evo-memory
cd "$REPO"

M1_LORA="experiment_artifacts/gcs/final_cs1024/m1_lora_matched.pt"
M4_LORA="experiment_artifacts/gcs/final_cs1024/m4_lora_namm.pt"
NAMM_MATCHED="experiment_artifacts/gcs/final_cs1024/namm_cs1024_maskfix.pt"
NAMM_M2="experiment_artifacts/gcs/M2_cs1024_maskfix/ckpt.pt"
OUT=eval_results/val_extra
PY=.venv/bin/python

for V in "${VARIANTS[@]}"; do
    echo "========== $V =========="
    case "$V" in
        m1_full)
            # cache_size=8192 matches stored m1_matched_full_cache.json: a
            # "full cache" eval for 4096-6500 token prompts (never evicts).
            $PY scripts/eval_namm_splits.py \
                --lora_checkpoint "$M1_LORA" \
                --cache_size 8192 --batch_size 1 --splits val \
                --run_label "$V" --output_dir "$OUT/$V"
            ;;
        m1_namm)
            $PY scripts/eval_namm_splits.py \
                --lora_checkpoint "$M1_LORA" \
                --namm_checkpoint "$NAMM_MATCHED" \
                --cache_size 1024 --batch_size 1 --splits val \
                --run_label "$V" --output_dir "$OUT/$V"
            ;;
        m1_trunc)
            # No --cache_size override: matches stored m1_matched_trunc1024.json
            # (cache=null; inputs are already truncated to 1024 tokens).
            $PY scripts/eval_namm_splits.py \
                --lora_checkpoint "$M1_LORA" \
                --truncate_input_to 1024 \
                --batch_size 1 --splits val \
                --run_label "$V" --output_dir "$OUT/$V"
            ;;
        m4_namm)
            $PY scripts/eval_namm_splits.py \
                --lora_checkpoint "$M4_LORA" \
                --namm_checkpoint "$NAMM_MATCHED" \
                --cache_size 1024 --batch_size 1 --splits val \
                --run_label "$V" --output_dir "$OUT/$V"
            ;;
        a4)
            # cache_size=8192 matches stored a4_m4_lora_no_namm.json
            $PY scripts/eval_namm_splits.py \
                --lora_checkpoint "$M4_LORA" \
                --cache_size 8192 --batch_size 1 --splits val \
                --run_label "$V" --output_dir "$OUT/$V"
            ;;
        b0)
            $PY scripts/eval_namm_splits.py \
                --plain \
                --batch_size 1 --splits val \
                --run_label "$V" --output_dir "$OUT/$V"
            ;;
        m2)
            $PY scripts/eval_namm_splits.py \
                --namm_checkpoint "$NAMM_M2" \
                --cache_size 1024 --batch_size 1 --splits val \
                --run_label "$V" --output_dir "$OUT/$V"
            ;;
        *)
            echo "Unknown variant: $V" >&2
            exit 2
            ;;
    esac
    echo "Done: $V"
done

echo "GROUP COMPLETE: ${VARIANTS[*]}"
