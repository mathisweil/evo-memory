# M1 LoRA + recency eviction, cache_size=1024

M1 LoRA (trained with full cache) evaluated under a fixed recency policy at cs=1024. Tests how much of M1's gain survives aggressive cache compression with a naive eviction heuristic (no learned policy).

**Source run:** `eval_results/lora_m1_recency_cs1024_5t/ext_20260411_173400/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 0.00 | 0.00 |
| lb/2wikimqa | 0.00 | 0.00 |
| lb/qasper_e | 0.00 | 0.00 |
| lb/hotpotqa_e | 0.00 | 0.00 |
| lb/2wikimqa_e | 0.00 | 0.00 |
| **mean F1** | **0.00** | **0.00** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_recency_cs1024_5t
```

