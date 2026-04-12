# A4 — M4 (cs1024) LoRA, NAMM disabled (full cache)

Ablation: take the M4 cs=1024 LoRA but evaluate it WITHOUT its NAMM, with full cache (cs=8192). Measures how much the LoRA alone contributes vs. LoRA+NAMM together.

**Source run:** `eval_results/lora_m4_cs1024_5t_ablation/ext_no_namm_20260412_141756/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 46.19 | 30.41 |
| lb/2wikimqa | 25.00 | 23.55 |
| lb/qasper_e | 28.12 | 21.41 |
| lb/hotpotqa_e | 26.67 | 31.57 |
| lb/2wikimqa_e | 17.46 | 25.07 |
| **mean F1** | **28.69** | **26.40** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext_no_namm \
    --output_dir eval_results/lora_m4_cs1024_5t_ablation
```

