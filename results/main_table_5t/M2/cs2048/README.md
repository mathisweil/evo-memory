# M2 — standalone NAMM, cache_size=2048 (friend's checkpoint)

Same as M2 cs1024 but using a NAMM checkpoint trained at cache_size=2048 by a collaborator.

**Source run:** `eval_results/namm_cs2048_5t/ext_20260412_141757/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 26.79 | 20.20 |
| lb/2wikimqa | 25.00 | 20.90 |
| lb/qasper_e | 6.06 | 10.14 |
| lb/hotpotqa_e | 18.45 | 24.71 |
| lb/2wikimqa_e | 15.18 | 20.49 |
| **mean F1** | **18.30** | **19.29** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/namm_cs2048_5t
```

