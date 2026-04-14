# M4 Joint Training Readiness Review (D2)

**Date:** 2026-04-14

Review of `scripts/run_joint.py` to answer whether the M4 joint config is
ready to run after the Part C3 fix (`lora_eval_interval: 14`).

---

## Q1: Does Stage 0 NAMM train on the base (un-adapted) model?

**Yes — and this is the intended design.** At the start of the outer loop,
LoRA adapters are attached to `memory_model` but their B matrix is zero-init,
so they contribute zero delta to the forward pass. Stage 0 NAMM therefore
evolves against effectively the base Llama-3.2-1B-Instruct weights plus the
5-task QA distribution.

This is the same model state under which M2 trains. Stage 0 NAMM is therefore
equivalent to a partial (100-iter) M2 run. The "useful to the adapted model"
concern is real but is handled by Stage 1: after the Stage 0 LoRA finishes,
the model has changed and Stage 1 NAMM trains against the adapted forward
pass.

## Q2: Does NAMM in loop k>0 resume from loop k-1 state or start fresh?

**Resumes from loop k-1 state.** See `scripts/run_joint.py:567-570`:

```python
namm_trainer.max_iters = namm_end_iter
if k > 0:
    namm_trainer.start_iter = namm_start_iter
    namm_trainer.force_initial_re_eval = False
```

The CMA-ES state (mean, covariance, step sizes, elite archive) persists across
outer loops. Iteration numbering is continuous: loop 0 covers iters 0–99, loop
1 covers iters 100–199, for a total of 200 iterations matching the M2 budget.

## Q3: Does LoRA in loop k>0 resume or restart?

**Weights persist, optimizer resets.** `_run_lora_stage` is called with the
same `memory_model` object across loops, so LoRA B/A tensors carry forward —
Stage 1 LoRA trains starting from the Stage 0 LoRA solution. But
`init_from=None` (line 760) means a fresh optimizer (no Adam moments) and a
fresh LR schedule are constructed each stage.

This matters: a 75-epoch cosine schedule re-runs its warmup each stage. The
effective M4 LR trajectory is two 75-epoch warm-up-and-decay curves rather
than a single 150-epoch curve. This is a real difference from M1/M3 — flag it
in the paper when reporting M4 rather than claiming "150 epochs total" as if
it were equivalent. "2 × 75-epoch LoRA stages, each with a fresh
warmup+cosine" is the correct description.

## Q4: Does the LR schedule reset per loop?

**Yes, see Q3.** Each Stage B builds a fresh `LoRATrainerConfig` → fresh
Transformer-style scheduler with `warmup_ratio=0.03` over
`lora_epochs_per_stage=75` epochs. Loop 0 and loop 1 both warm up from near-0
LR to `5e-5` then cosine-decay back.

## Q5: What batch size does the LoRA stage use?

`_run_lora_stage` at `run_joint.py:750-751`:

```python
batch_size=args.lora_batch_size,
gradient_accumulation_steps=args.gradient_accumulation_steps,
```

After the joint config fixes these are `1` and `16` respectively → effective
batch = 16, matching M1/M3. Correct.

## Q6: Does `early_stopping_patience` apply to M4 LoRA stages?

**No — it is not plumbed through.** `_run_lora_stage` constructs its
`LoRATrainerConfig` without setting `early_stopping_patience`, so the trainer
default (0, meaning off) applies. Each LoRA stage runs its full 75 epochs.

This is consistent with the outer-loop budget accounting (M4 compute is
pinned to M1+M2's total compute), but asymmetric with M1/M3 which now use
`early_stopping_patience: 20`. The asymmetry is documented in the config
comparison matrix.

## Q7: Does the post-stage eval use the FINAL or BEST LoRA weights?

**FINAL.** `eval_after_each_loop=true` triggers a single eval using whatever
weights are in `memory_model` at the end of Stage B. With `lora_eval_interval`
pre-C3 at 999999, there was no mid-stage best-checkpointing either — the final
model and the best model were always the same (no selection happened).

After C3 (`lora_eval_interval=14`), mid-stage eval happens every 14 steps
inside the LoRA stage, and `always_save_checkpoint=true` (implicit from
`_run_lora_stage:759`) writes a best checkpoint. But the post-stage eval at
`run_joint.py:652-653` uses the live `memory_model`, not the best-checkpoint
weights. So the stage-end eval still sees the final weights.

**This is a limitation to document, not a bug.** The best-checkpoint selection
is available in the stage directory for analysis, but the headline M4 number
reported by `eval_after_each_loop` is the final-epoch number. A follow-up
eval against `adapter/stage_1/best_ckpt.pt` would recover the best-of-stage
selection if desired — add this as a separate eval invocation in the final
M4 evaluation pipeline.

---

## Ready to run?

**Yes, after the Part C3 fix.** The remaining asymmetries (Q3 warmup restart,
Q6 no early stopping, Q7 final-weights eval) are intrinsic to the
joint-alternating design, not config bugs. Document them in the paper's M4
section rather than "fix" them.

Command:

```bash
venv/bin/python scripts/run_joint.py \
    --config scripts/configs/m4_joint_lora_5t.yaml \
    --run_name m4_joint_lora_matched
```

Expected runtime: NAMM stages ~3h each + LoRA stages ~6h each + post-stage
eval ~15 min, ×2 outer loops = ~18–20 hours on an RTX 3090 Ti.
