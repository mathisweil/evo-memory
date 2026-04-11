# A4 — M4 (cs2048) LoRA, NAMM disabled (full cache)

Same ablation as A4 cs1024 but for the cs=2048 LoRA.

**Source run:** `eval_results/lora_m4_cs2048_5t_ablation/ext_no_namm_20260411_151122/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 17.32 | 15.47 |
| lb/2wikimqa | 24.60 | 30.28 |
| lb/qasper_e | 15.08 | 12.40 |
| lb/hotpotqa_e | 32.94 | 36.08 |
| lb/2wikimqa_e | 23.02 | 27.15 |
| **mean F1** | **22.59** | **24.28** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext_no_namm \
    --output_dir eval_results/lora_m4_cs2048_5t_ablation
```

