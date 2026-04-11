# M2 — standalone NAMM, cache_size=2048 (friend's checkpoint)

Same as M2 cs1024 but using a NAMM checkpoint trained at cache_size=2048 by a collaborator.

**Source run:** `eval_results/namm_cs2048_5t/ext_20260411_150302/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 9.65 | 11.18 |
| lb/2wikimqa | 25.00 | 14.96 |
| lb/qasper_e | 6.61 | 8.65 |
| lb/hotpotqa_e | 25.79 | 32.78 |
| lb/2wikimqa_e | 13.61 | 19.92 |
| **mean F1** | **16.13** | **17.50** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/namm_cs2048_5t
```

