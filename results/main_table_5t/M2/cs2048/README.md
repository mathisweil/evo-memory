# M2 — standalone NAMM, cache_size=2048 (friend's checkpoint)

Same as M2 cs1024 but using a NAMM checkpoint trained at cache_size=2048 by a collaborator.

**Source run:** `eval_results/namm_cs2048_5t/ext_20260413_135808/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 19.81 | 20.67 |
| lb/2wikimqa | 25.00 | 21.46 |
| lb/qasper_e | 7.30 | 10.62 |
| lb/hotpotqa_e | 12.28 | 24.71 |
| lb/2wikimqa_e | 15.18 | 20.06 |
| **mean F1** | **15.91** | **19.50** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/namm_cs2048_5t
```

