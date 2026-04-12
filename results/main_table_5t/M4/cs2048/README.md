# M4 — LoRA on frozen NAMM, cache_size=2048

Same as M4 cs1024 but with the cs=2048 NAMM and a LoRA trained against it.

**Source run:** `eval_results/lora_m4_cs2048_5t/ext_20260412_141752/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 39.68 | 23.41 |
| lb/2wikimqa | 25.00 | 22.00 |
| lb/qasper_e | 30.47 | 14.59 |
| lb/hotpotqa_e | 35.51 | 29.55 |
| lb/2wikimqa_e | 24.60 | 27.24 |
| **mean F1** | **31.05** | **23.36** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m4_cs2048_5t
```

