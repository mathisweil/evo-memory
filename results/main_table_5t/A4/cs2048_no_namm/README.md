# A4 — M4 (cs2048) LoRA, NAMM disabled (full cache)

Same ablation as A4 cs1024 but for the cs=2048 LoRA.

**Source run:** `eval_results/lora_m4_cs2048_5t_ablation/ext_no_namm_20260412_141739/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 43.56 | 24.54 |
| lb/2wikimqa | 38.89 | 31.15 |
| lb/qasper_e | 34.63 | 22.88 |
| lb/hotpotqa_e | 35.80 | 29.92 |
| lb/2wikimqa_e | 17.46 | 21.31 |
| **mean F1** | **34.07** | **25.96** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext_no_namm \
    --output_dir eval_results/lora_m4_cs2048_5t_ablation
```

