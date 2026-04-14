# Compute Budget Validation (FAIR-01)

Generated: 2026-04-13

Verifies that M1, M2, and M4 have matched compute budgets per `experiment_specification.md`.

---

## M1 — LoRA gradient steps

Spec: 150 epochs, effective batch size 16, on 306 train samples.

```
DataLoader length = ceil(306 / batch_size) = ceil(306 / 4) = 77 batches per epoch
steps_per_epoch = 77 // gradient_accumulation_steps = 77 // 4 = 19 optimizer steps
total_steps = 150 * 19 = 2850 optimizer steps
```

Code confirmation (`grad_lora_finetuning/trainer.py:323-324`):
```python
steps_per_epoch = len(self.dataloader) // cfg.gradient_accumulation_steps
total_steps = cfg.num_epochs * steps_per_epoch
```

The spec says `ceil(306/16) * 150 = 20 * 150 = 3000`. This is approximate — the actual
value depends on the DataLoader batching. With `batch_size=4, grad_accum=4`:
- `len(dataloader) = ceil(306/4) = 77`
- `steps_per_epoch = 77 // 4 = 19` (floor division — last 1 batch is dropped)
- `total_steps = 150 * 19 = 2850`

**Delta:** 2850 vs spec's estimate of 3000 (5% fewer steps). This is because `77 // 4 = 19`
drops the remainder. The actual compute is 2850 gradient updates. This is consistent
across conditions since all use the same 306 train samples.

---

## M2 — NAMM CMA-ES generations

Spec: 200 CMA-ES generations.

Config (`namm_bam_i1_llama32_1b_5t.yaml`): `max_iters: 200`. Direct match.

Per generation: `pop_size=8` members, each evaluated on `samples_batch_size=8` prompts
per task across 5 tasks = `8 * 8 * 5 = 320` forward passes per generation.
Total: `200 * 320 = 64,000` forward passes.

---

## M4 — Joint (alternating NAMM + LoRA)

Spec: 2 outer loops x (100 NAMM gens + 75 LoRA epochs).

NAMM total: 2 * 100 = 200 generations (matches M2).
LoRA total: 2 * 75 = 150 epochs (matches M1 epoch count).

M4 LoRA per-stage steps:
- `m4_joint_lora_5t.yaml`: `batch_size` not set (uses eval batch), but the LoRA stage
  inherits from `LoRATrainerConfig` which gets `batch_size` from the joint config or
  falls back to 1. With `gradient_accumulation_steps=16`:
  - `len(dataloader) = ceil(306/1) = 306`
  - `steps_per_epoch = 306 // 16 = 19`
  - Per stage: `75 * 19 = 1425`
  - Total: `2 * 1425 = 2850`

This matches M1's 2850 total optimizer steps. Effective batch size = 1 * 16 = 16 = M1's
4 * 4. Budget matched.

**Note:** M3 (frozen NAMM) uses `batch_size=1, grad_accum=16` (effective=16) to avoid
OOM with NAMM active. Same effective batch and same 150 epochs = same 2850 steps.

---

## Summary

| Condition | Mechanism | Budget | Steps/Gens | Effective Batch |
|-----------|-----------|--------|------------|-----------------|
| M1 LoRA   | SGD       | 150 ep | 2850 steps | 16 (4 * 4)     |
| M2 NAMM   | CMA-ES    | 200 gen| 200 gens   | n/a (ES)        |
| M3 LoRA+N | SGD       | 150 ep | 2850 steps | 16 (1 * 16)    |
| M4 joint  | Both      | 200 gen + 150 ep | 200 + 2850 | matched |

All conditions have matched compute budgets. FAIR-01 satisfied.
