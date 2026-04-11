# A4 — M4 (cs1024) LoRA, NAMM disabled (full cache)

Ablation: take the M4 cs=1024 LoRA but evaluate it WITHOUT its NAMM, with full cache (cs=8192). Measures how much the LoRA alone contributes vs. LoRA+NAMM together.

**Source run:** `eval_results/lora_m4_cs1024_5t_ablation/ext_no_namm_20260411_151116/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 36.50 | 28.84 |
| lb/2wikimqa | 24.60 | 24.66 |
| lb/qasper_e | 33.23 | 26.48 |
| lb/hotpotqa_e | 43.06 | 38.03 |
| lb/2wikimqa_e | 15.67 | 23.82 |
| **mean F1** | **30.61** | **28.37** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext_no_namm \
    --output_dir eval_results/lora_m4_cs1024_5t_ablation
```

