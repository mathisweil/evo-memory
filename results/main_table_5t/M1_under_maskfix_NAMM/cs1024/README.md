# M1 LoRA + maskfix NAMM cs1024 (distribution shift)

M1 LoRA evaluated under the maskfix NAMM. Shows distribution shift — M1 was not trained with this eviction regime.

**Status:** ⏳ pending — no completed `ext_*` run found in `eval_results/lora_m1_namm_cs1024_maskfix_5t_bs1/`.

## Command

```bash
eval_namm_splits.py --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt --lora_checkpoint M1
```
