# M1 LoRA (no NAMM training) + NAMM eviction cs2048

Same as M1_under_NAMM/cs1024 but at cache_size=2048.

**Source run:** `eval_results/lora_m1_namm_cs2048_5t/ext_20260412_205551/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 35.34 | 23.74 |
| lb/2wikimqa | 27.56 | 20.10 |
| lb/qasper_e | 27.20 | 21.60 |
| lb/hotpotqa_e | 35.19 | 35.75 |
| lb/2wikimqa_e | 32.65 | 25.28 |
| **mean F1** | **31.59** | **25.29** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_namm_cs2048_5t
```

