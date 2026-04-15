# M2 maskfix — standalone NAMM cs1024 (correct attention)

NAMM eviction with the attention mask fix. The policy was trained with correct prefill attention.

**Status:** ⏳ pending — no completed `ext_*` run found in `eval_results/namm_cs1024_maskfix_5t_bs1/`.

## Command

```bash
eval_namm_splits.py --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt --cache_size 1024
```
