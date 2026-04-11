# M1 — LoRA only (no NAMM)

LoRA SFT on the 5-task QA subset, full KV cache during training and eval (cache_size=8192). No eviction.

**Source run:** `eval_results/lora_m1_5t/ext_20260411_145757/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 45.03 | 35.92 |
| lb/2wikimqa | 10.00 | 23.59 |
| lb/qasper_e | 35.62 | 27.81 |
| lb/hotpotqa_e | 30.51 | 47.83 |
| lb/2wikimqa_e | 30.16 | 31.93 |
| **mean F1** | **30.26** | **33.42** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_5t
```

