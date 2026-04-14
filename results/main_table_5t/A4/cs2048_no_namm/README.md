# A4 — M4 (cs2048) LoRA, NAMM disabled (full cache)

Same ablation as A4 cs1024 but for the cs=2048 LoRA.

**Source run:** `eval_results/lora_m4_cs2048_5t_ablation/ext_newbest_20260413_135800/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 29.90 | 26.38 |
| lb/2wikimqa | 18.78 | 28.88 |
| lb/qasper_e | 32.79 | 26.03 |
| lb/hotpotqa_e | 35.78 | 36.52 |
| lb/2wikimqa_e | 31.75 | 27.45 |
| **mean F1** | **29.80** | **29.05** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext_no_namm \
    --output_dir eval_results/lora_m4_cs2048_5t_ablation
```

