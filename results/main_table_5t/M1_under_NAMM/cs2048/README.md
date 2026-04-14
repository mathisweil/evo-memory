# M1 LoRA (no NAMM training) + NAMM eviction cs2048

Same as M1_under_NAMM/cs1024 but at cache_size=2048.

**Source run:** `eval_results/lora_m1_namm_cs2048_5t/ext_20260413_090611/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 35.34 | 24.26 |
| lb/2wikimqa | 27.56 | 20.73 |
| lb/qasper_e | 28.58 | 21.52 |
| lb/hotpotqa_e | 35.19 | 34.32 |
| lb/2wikimqa_e | 32.65 | 26.92 |
| **mean F1** | **31.86** | **25.55** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_namm_cs2048_5t
```

