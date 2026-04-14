# M4 Joint Training — Consolidated Analysis

**Date:** 2026-04-14

Consolidates the readiness review (`docs/m4_readiness_review.md`), the
3-loop schedule decision, and the early-stopping plumbing change into a
single reference for running M4 and interpreting its logs.

---

## 1. Schedule decision: 3 × (67 NAMM + 50 LoRA)

M4 runs **3 outer loops**, each containing a 67-iteration CMA-ES NAMM stage
followed by a 50-epoch LoRA stage. Totals:

- 3 × 67 = **201 NAMM generations** (matches M2's 200)
- 3 × 50 = **150 LoRA epochs** (matches M1's 150)

Allocated compute is therefore identical to M1 + M2 — the FAIR-01
requirement. Effective LoRA compute is smaller (see §4).

### Why 3 loops, not 2

The earlier design was 2 × (100 NAMM + 75 LoRA). Switching to 3 × (67 + 50)
at the same total budget buys an extra co-adaptation cycle:

- **2 loops → 1 co-adaptation cycle.** Loop 0 NAMM trains on the
  base model, loop 1 NAMM trains on the adapted model. 50% of LoRA epochs
  happen under a base-model-aimed NAMM.
- **3 loops → 2 co-adaptation cycles.** Loop 0 NAMM trains on the base
  model, loops 1 and 2 train on the increasingly adapted model. Only 33%
  of LoRA epochs happen under the base-model-aimed NAMM.

### Per-stage sizing rationale

- **67 NAMM iters per stage.** M2 reaches its best val F1 at iteration ~105
  out of 200. Loop 0 from scratch at 67 iters will not fully converge, but
  it does not need to — it only needs to produce a reasonable starting
  policy for loop 1. Loops 1 and 2 warm-start from the previous
  checkpoint (see §2) and only have to adapt, for which 67 iters is
  plenty.
- **50 LoRA epochs per stage.** M1 / M3 early-stop around epoch 30
  (best val F1 at ~epoch 17, patience-20 exhausted by ~epoch 30). 50
  allocated epochs provides 20 epochs of headroom for the mechanism to
  trigger cleanly on each stage. Each stage trains under a progressively
  better NAMM.

## 2. Warm-start and continuity behaviour

From the readiness review (`run_joint.py:567-570` and `_run_lora_stage`):

- **NAMM warm-starts across loops.** CMA-ES mean, covariance, step sizes,
  and elite archive persist into the next outer loop. Iteration numbering
  is continuous: with `namm_iterations_per_stage=67`, loop 0 covers iters
  0–66, loop 1 covers 67–133, loop 2 covers 134–200. `latest.pt` is
  overwritten at every NAMM iteration and always reflects the most recent
  generation (not the best-so-far).
- **LoRA weights continue.** `_run_lora_stage` receives the same
  `memory_model` object each outer loop, so the B / A tensors carry
  forward — loop k LoRA starts from the loop k-1 solution.
- **LoRA optimizer and LR schedule restart.** `init_from=None` at the
  LoRA-stage entry means a fresh AdamW optimizer (no carried moments) and
  a fresh warmup+cosine schedule are built every stage. With
  `warmup_ratio=0.03` over 50 epochs per stage, M4 runs three
  ~1.5-epoch warmups and three cosine decays rather than M1's single
  ~4.5-epoch warmup over 150 epochs. This is documented as an
  asymmetry, not as a bug (see §5).

## 3. Early stopping status

**Now plumbed through.** `scripts/run_joint.py` accepts
`--lora_early_stopping_patience` (added 2026-04-14) and forwards it to
`LoRATrainerConfig.early_stopping_patience` inside `_run_lora_stage`
(run_joint.py:773). The value is logged into `config_dict` so each
joint run's wandb/experiment record shows whether it was active.

`scripts/configs/m4_joint_lora_5t.yaml` sets
`lora_early_stopping_patience: 20`, matching M1 / M3. Patience applies
**per stage**: a single LoRA stage stops early after 20 evals without
val F1 improvement; the outer loop then advances to the next NAMM stage
as normal. Early-stopping out of one stage does NOT terminate the joint
run.

This closes the prior asymmetry with M1 / M3 that was flagged in
`docs/m4_readiness_review.md` Q6.

## 4. Compute budget — allocated vs effective

| Quantity         | M1 / M3 | M2  | M4 (allocated)    | M4 (effective, expected)                  |
|------------------|---------|-----|--------------------|-------------------------------------------|
| LoRA epochs      | 150     | —   | 150 (3 × 50)       | ~90 (3 × ~30 before early-stop)           |
| NAMM iters       | —       | 200 | 201 (3 × 67)       | 201 (no early-stop on CMA-ES side)        |

M4's effective LoRA compute (~90 epochs) is larger than a single M1 run
that early-stops at ~30, but each of M4's three stages still early-stops.
The extra work is real training time under a *changing* NAMM, not wasted
epochs — the evolving eviction policy prevents the overfitting that ends
M1 early.

When the headline M4 number is reported, cite "3 × 50 allocated LoRA
epochs, effective ≈ 90 after per-stage early stopping (patience 20)"
rather than "150 epochs" — the latter implies a single uninterrupted
150-epoch schedule, which M4 does not run.

## 5. Known asymmetries with M1 / M3

All three are intrinsic to the joint-alternating design, not configuration
bugs.

1. **Stage 0 NAMM trains on the un-adapted model.** At outer-loop entry,
   LoRA B is zero-init, so the attention patterns Stage 0 NAMM evolves
   against are those of the base Llama-3.2-1B-Instruct. This is the same
   state M2 trains under, so Stage 0 NAMM is equivalent to a partial
   (67-iter) M2 run. The "useful to the adapted model" concern is handled
   by loops 1 and 2, where NAMM evolves against the adapted forward pass.
2. **Three LR warmups instead of one.** See §2. Report M4's LR schedule
   as "three 50-epoch stages, each with a fresh warmup+cosine", not "a
   150-epoch cosine with warmup".
3. **Post-stage eval uses FINAL, not BEST, LoRA weights.**
   `eval_after_each_loop=true` evaluates the live `memory_model` at the
   end of the stage (run_joint.py:652-653), not the best-checkpoint
   weights saved mid-stage. Best-checkpoint eval is available after the
   fact via `adapter/stage_K/best_ckpt.pt` — add it as a separate
   `run_eval.py` invocation in the final M4 evaluation pipeline if the
   final-epoch number is suspicious.

## 6. What to watch in WandB

During the run, a monotonic-improvement pattern across loops is the
success criterion. Red flags:

- **Loop 0 NAMM val F1 at iter 67 much worse than M2 at iter 67.**
  Either the 67-iter budget per stage is too tight for from-scratch
  convergence, or the CMA-ES state is diverging. Compare against the M2
  `experiments/experiment_*/m2*/results.json` curves.
- **Per-loop LoRA val F1 does not improve across loops.** Loop 2 LoRA
  should beat loop 1 LoRA should beat loop 0 LoRA. If loop 1 < loop 0,
  NAMM in loop 1 is likely hurting the adapted model rather than helping
  it — inspect the NAMM iter-67 to iter-133 trajectory.
- **Per-loop `eval_after_each_loop` F1 does not improve monotonically.**
  This is the headline metric; non-monotonicity here is the load-bearing
  concern. One-off dips are acceptable; a downward trend is not.
- **Eviction utilisation not saturating at `cache_size=1024`.** Check
  `eviction_stats.budget_utilization_pct` in `results.json` — NAMM should
  converge to keeping close to 1024 tokens. Large budget under-utilisation
  means the scoring head has drifted.

## 7. Run command

```bash
venv/bin/python scripts/run_joint.py \
    --config scripts/configs/m4_joint_lora_5t.yaml \
    --run_name m4_joint_lora_matched
```

All M4 hyperparameters live in `m4_joint_lora_5t.yaml`; no CLI
overrides are required. The config sets `num_outer_loops=3`,
`namm_iterations_per_stage=67`, `lora_epochs_per_stage=50`,
`lora_early_stopping_patience=20`, `lora_eval_interval=14`, and M1-matched
LoRA hyperparameters (`learning_rate=5e-5`, `lora_dropout=0.1`,
`lora_batch_size=1`, `gradient_accumulation_steps=16`).

Final checkpoint paths (0-indexed stages):

- Adapter: `experiments/experiment_N/joint_lora/m4_joint_lora_matched/adapter/stage_2/`
- NAMM: `experiments/experiment_N/joint_lora/m4_joint_lora_matched/namm/latest.pt`

Expected runtime on an RTX 3090 Ti: NAMM stages ~2 h each + LoRA stages
~4 h each + post-stage eval ~15 min, × 3 outer loops ≈ 18–20 h. Early
stopping in the LoRA stages may shave another 2–4 h off the total.
