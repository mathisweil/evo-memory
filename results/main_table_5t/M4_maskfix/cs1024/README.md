# M4 maskfix — LoRA + maskfix NAMM cs1024 (correct attention)

LoRA trained with the maskfix NAMM (correct prefill attention). Both NAMM and LoRA operate with the attention fix.

**Status:** ⏳ pending — no completed `ext_*` run found in `eval_results/lora_m4_cs1024_maskfix_5t_bs1/`.

## Command

```bash
eval_namm_splits.py --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt --lora_checkpoint M4_maskfix
```
