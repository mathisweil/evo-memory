# M4 — LoRA on frozen NAMM, cache_size=2048

Same as M4 cs1024 but with the cs=2048 NAMM and a LoRA trained against it.

**Source run:** `eval_results/lora_m4_cs2048_5t/ext_newbest_20260413_135804/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 35.74 | 21.42 |
| lb/2wikimqa | 25.00 | 22.67 |
| lb/qasper_e | 29.86 | 22.52 |
| lb/hotpotqa_e | 35.12 | 26.99 |
| lb/2wikimqa_e | 31.41 | 23.93 |
| **mean F1** | **31.43** | **23.51** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m4_cs2048_5t
```

