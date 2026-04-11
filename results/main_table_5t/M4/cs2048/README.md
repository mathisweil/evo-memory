# M4 — LoRA on frozen NAMM, cache_size=2048

Same as M4 cs1024 but with the cs=2048 NAMM and a LoRA trained against it.

**Source run:** `eval_results/lora_m4_cs2048_5t/ext_20260411_150302/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 13.88 | 11.22 |
| lb/2wikimqa | 24.21 | 28.31 |
| lb/qasper_e | 13.85 | 8.77 |
| lb/hotpotqa_e | 26.79 | 38.86 |
| lb/2wikimqa_e | 23.47 | 20.49 |
| **mean F1** | **20.44** | **21.53** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m4_cs2048_5t
```

