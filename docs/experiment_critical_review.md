# Experiment Critical Review

**Scope:** Deep analysis of completed experiments for ACL paper submission readiness.
**Date:** 2026-04-13

---

## 1a. Training Completeness — Did Runs Finish?

### M1 (r=8): 28 of 150 epochs (18.7%)

The M1-r8 run crashed 3 times across WandB segments (`kz6vqo2o` -> `x9a4smmf` -> `qfoxxi2m`). The final segment was killed at epoch 28, step 684. Best val avg F1 = 45.48 at step 336.

**Was the learning curve still rising?** Yes. The progression across segments (39.15 -> 40.60 -> 45.48) shows continuous improvement. Step 336 is only ~11% of the planned ~3000 total steps (150 epochs x ceil(306/16) = ~2869 steps). Given that validation F1 was still climbing, the final-epoch model would likely outperform the step-336 checkpoint.

**Resume capability:** `run_lora.py` supports `--resume_checkpoint` (line 103, fed to `LoRATrainerConfig.init_from`). The trainer (`grad_lora_finetuning/trainer.py`) loads optimizer state, scheduler state, and resumes from the exact global step. **Recommendation: resume from the GCS checkpoint rather than restart.**

**Impact on results:** The test F1 of 31.14 is from an undertrained model. A fully-trained M1 (150 epochs) would likely exceed this, which would strengthen or weaken the M3>M1 comparison depending on the magnitude of improvement. This is a **paper-blocking issue**: we cannot claim M3 outperforms M1 if M1 was only 18.7% trained.

### M3/cs1024: 25 of 150 epochs (16.7%)

Crashed at step 609, best val F1 = 45.59 at step 340. Similar trajectory to M1 - still improving. Test F1 = 32.28.

**Key comparison problem:** M1 (28 epochs) and M3 (25 epochs) are both severely undertrained, but their test F1s are close (31.14 vs 32.28). If both were fully trained, the gap could widen in either direction. The current 1.14-point advantage for M3 is within the noise floor for 70 test samples.

### M3/cs2048: 15 of 150 epochs (10.0%)

Crashed at step 371, best val F1 = 44.86 at step 244. Only 10% of planned compute. Test F1 = 31.06.

### M3/cs3072: "Finished" at epoch 4 (2.7%)

Only 4 epochs (117 steps). Val F1 = 33.37. This run either had a configuration error or hit early stopping after only 4 epochs. Given `early_stopping_patience=5` and `eval_interval=2`, the run would have stopped if 5 consecutive eval checks (every 2 steps) showed no improvement. But 4 epochs x ~19 steps/epoch = ~76 steps, and with eval every 2 steps that's ~38 evals. The low val F1 (33.37 vs 45.59 for cs1024) suggests the larger cache size may have caused an issue — possibly OOM-reduced batch sizes or the larger NAMM processing burden.

### Verdict

**No main-table condition completed its full training budget.** This is the single most critical issue. All F1 values in the results table are from partially-trained models, making any cross-condition comparison unreliable.

---

## 1b. Val-Test Gap — Why Is It So Large?

| Condition | Best Val F1 | Test Micro F1 | Gap |
|-----------|-----------|-------------|-----|
| M1 (r=8) | 45.48 | 31.14 | **-14.3** |
| M3/cs1024 | 45.59 | 32.28 | **-13.3** |
| M2/cs1024 | 27.90 | 20.30 | **-7.6** |

### Investigation: Are val and test F1 computed identically?

**Val F1 (during training):** Computed in `grad_lora_finetuning/trainer.py:756-787` via `_evaluate_f1(split='val')`. This calls `task_sampler.evaluate()` which runs `evaluate_lb()` in `namm/tasks.py`, which calls `longbench.get_score()` which uses `qa_f1_score()`. The result is a weighted mean across tasks (`lb/avg_f1`, line 782-785), weighted by per-task sample count.

**Test F1 (standalone eval):** Computed in `scripts/eval_namm_splits.py` via the same `task_sampler.evaluate()` -> `evaluate_lb()` -> `longbench.get_score()` -> `qa_f1_score()` path. The micro F1 in `all_results.json` is computed by `scripts/patch_micro_mean_f1.py` as a sample-count-weighted mean.

**Key finding: The metric computation is identical.** Both use `qa_f1_score` from `namm/evaluation/metrics.py`, which normalizes (lowercase, strip articles/punctuation, whitespace collapse) then computes token-level F1.

### Root cause analysis

1. **Sample count asymmetry:** Val has 64 samples, test has 70. This alone doesn't explain a 14-point gap.

2. **Checkpoint selection on val:** The best checkpoint is chosen by val F1. If the model overfits the val split at step 336 (small val set of 64 samples), the test performance will be lower. With only 64 val samples, the estimate has high variance — the model could be "lucky" on val without genuinely generalizing.

3. **Chat template mismatch:** During training, `apply_chat_template_to_prompts()` wraps eval prompts in chat format (line 268 of `run_lora.py`). Standalone eval via `eval_namm_splits.py` does NOT apply chat template to prompts (the evaluator's `evaluate_lb` uses raw prompts). This is a **potential metric computation difference** — if the model was fine-tuned on chat-formatted inputs (SFT mode), evaluating on raw inputs would degrade performance.

    **Code evidence:** In `run_lora.py:267-268`, when `sft_mode=true`, the script calls `task_sampler.apply_chat_template_to_prompts(memory_evaluator.tokenizer)`. But in `eval_namm_splits.py`, there is no such call. The evaluator receives raw LongBench prompts.

    **However:** The val eval during training also uses the task_sampler with chat templates applied (same task_sampler instance). So val and test during the run_lora.py script use the same format. The gap appears when comparing val F1 from run_lora.py with test F1 from eval_namm_splits.py.

4. **True val-test generalization gap:** With an early-stopped model (step 336 out of ~2869), the model is still in an early training regime. The val-selected checkpoint could be a local optimum on the small val set that doesn't transfer.

### Verdict

The 14-point gap is concerning but likely explained by (a) checkpoint selection on a small val set creating optimistic val estimates, and (b) possible chat template differences between training-time eval and standalone eval. **Investigate the chat template issue in eval_namm_splits.py.**

---

## 1c. Hyperparameter Fairness — M1 vs M3 Confounds

### Actual config comparison

| Parameter | M1 (m1_lora_5t.yaml) | M3 (m3_lora_frozen_namm_5t.yaml) | Match? |
|-----------|-----------------------------------|-----------------------------------|--------|
| `learning_rate` | 5e-5 | 1e-4 | **NO** (2x higher for M3) |
| `lora_dropout` | 0.1 | 0.05 | **NO** (half for M3) |
| `batch_size` | 1* | 1 | YES |
| `gradient_accumulation_steps` | 16* | 16 | YES |
| `warmup_ratio` | 0.03 | 0.03 | YES |
| `weight_decay` | 0.01 | 0.01 | YES |
| `max_grad_norm` | 1.0 | 1.0 | YES |
| `lora_rank` | 8 | 8 | YES |
| `lora_alpha` | 16 | 16 | YES |
| `lora_target_modules` | [q_proj, v_proj] | [q_proj, v_proj] | YES |
| `eval_interval` | 24 | 2 | Different (M1 recently changed to 24 for OOM) |

*Note: M1's spec says batch_size=4, grad_accum=4 (eff=16), but the current config has been modified to batch_size=1, grad_accum=16 for 8GB VRAM. The completed M1 run used the original batch_size=4 setting.*

### Analysis

**Learning rate confound (2x):** M3 uses `learning_rate=1e-4` vs M1's `5e-5`. This is the most serious confound. If M3 outperforms M1, we cannot cleanly attribute it to NAMM presence — the higher LR could be responsible. The M3 config comment says "lower lr for NAMM stability" (in deprecated/m3_lora_frozen_namm.yaml:14), but 1e-4 is actually HIGHER than M1's 5e-5.

**Possible justification:** With NAMM active, gradients may be noisier (evicted tokens change the loss landscape each step), requiring a higher LR to escape local optima. However, this argument is post-hoc and untested. A reviewer could equally argue the higher LR causes M3 to overfit faster.

**Dropout confound (half):** M3 uses `lora_dropout=0.05` vs M1's `0.1`. Lower dropout is less regularization. Combined with the higher LR, M3 has a fundamentally different optimization regime.

**Impact on main claim:** The paper's central argument is that "training under NAMM eviction helps LoRA adapt to compressed context." But if M3 simply benefits from better hyperparameters, the eviction is irrelevant. **This is paper-weakening but not paper-blocking**, because the LR and dropout differences are documented and can be framed as "condition-specific tuning." However, a reviewer will likely request an ablation (M3 with M1's LR) to isolate the NAMM effect.

### Recommendation

Run M3 with M1-identical hyperparameters (`lr=5e-5`, `dropout=0.1`) as an additional condition. If M3 still matches/beats M1, the NAMM effect is real regardless of LR. If M3 drops significantly, the confound is real.

---

## 1d. FAIR-01 Memory Equivalence — M1 Eval Cache

**The problem:** M1 is evaluated with full KV cache (~6500 tokens), while M3 is evaluated at `cache_size=1024`. The spec acknowledges M1 as an "upper bound," but this means the main table compares conditions under different memory budgets.

**M1_recency all zeros:** The intended fix was to also evaluate M1's LoRA checkpoint under recency eviction at `cache_size=1024`. This would show whether M1's LoRA simply doesn't work under compression, or if the eviction is incompatible with M1's learned weights.

The M1_recency run produced all-zero F1 because **the model generated `!!!!!!!!...` for every prompt.** This is degenerate generation, not zero-F1 from wrong answers.

### Root cause of M1_recency all-zeros

From the M1_recency command:
```bash
python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8
```

The command does NOT pass `--namm_checkpoint` or `--use_classic_recency`. Looking at the `eval_namm_splits.py` flow:
1. Line 277-284: Without `--namm_checkpoint` and without `--use_classic_recency`, it uses init params (scoring_initializer=0, all scores ~ 0).
2. With scoring_initializer=0, all token scores are near zero. The default `threshold_only=false` means it uses top-k selection, which keeps the `cache_size` highest-scoring tokens. With all scores equal, this is effectively random — not recency.
3. The LoRA was trained on full-context patterns. Random eviction destroys the attention patterns the LoRA depends on.
4. With 75-85% of tokens randomly evicted, the model's attention collapses, producing degenerate `!` repetitions.

**The fix:** The command should have used `--use_classic_recency` to get proper recency eviction (keep last N tokens, drop oldest). The actual command ran the LoRA under a randomly-initialized NAMM scoring policy, which is neither recency nor learned eviction — it's random eviction.

```bash
# Corrected command:
python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --use_classic_recency --cache_size 1024 --batch_size 1 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_recency_cs1024_5t
```

---

## 1e. M2 Cache Size Scaling — Does More Cache Help?

| Cache | M2 Test F1 | M3 Test F1 |
|-------|-----------|-----------|
| 1024 | 20.30 | 32.28 |
| 2048 | 17.40 | 31.06 |

### Why M2 gets worse with larger cache

**M2/cs1024 best iter = 105. M2/cs2048 best iter = 40.**

The cs2048 NAMM converged much earlier (iter 40 vs 105). Two possible explanations:

1. **Easier task = shallower objective:** At cs2048, only ~50-70% of tokens are evicted (vs 75-85% at cs1024). With less aggressive eviction needed, the CMA-ES objective landscape is flatter — many policies perform similarly, and CMA-ES finds a reasonable policy quickly but cannot refine it. The "best at iter 40" may be noise rather than a converged optimum.

2. **Higher-dimensional search with smaller gradient signal:** At cs2048, the model retains more tokens, so the marginal benefit of evicting any particular token is smaller. CMA-ES has the same parameter count but smaller fitness differences between population members, making the optimization signal weaker.

**Why M3/cs2048 < M3/cs1024:** With a weaker M2/cs2048 NAMM, the frozen policy during M3 training is less discriminating. The LoRA has less to adapt to (eviction is milder), so it learns a smaller adaptation. The result (31.06 vs 32.28) is only 1.22 F1 points — within noise for 70 samples.

---

## 1f. M4 Joint Training Design — Critical Assessment

M4 has not been run. Before running it:

### 2 outer loops: sufficient?

**Stage 0 (NAMM, 100 iters):** NAMM trains on a random-weight base model (no LoRA yet). M2 standalone found its best policy at iter 105. With only 100 iterations in Stage 0, the NAMM may not converge. However, in M4, the NAMM only needs to be "good enough" — it doesn't need to be optimal because it will be refined in Stage 2 after the LoRA adapts.

**Stage 1 (LoRA, 75 epochs):** LoRA trains with Stage 0 NAMM active. 75 epochs is half the M1 budget. M1 only reached epoch 28 before crashing, but had best val F1 still rising. 75 epochs should be sufficient for meaningful adaptation.

**Stage 2 (NAMM, 100 iters):** NAMM re-trains on the LoRA-adapted model. The LoRA now produces different attention patterns, so the NAMM must re-learn. Starting from Stage 0's NAMM (not from scratch) via CMA-ES state continuation should help.

**Stage 3 (LoRA, 75 epochs):** Final LoRA refinement. The LR schedule resets (new LoRAGradTrainer created at line 760 in `_run_lora_stage`), so the model gets a fresh warmup.

### Code analysis of `run_joint.py`

1. **LoRA weights persist across stages:** LoRA params are frozen/unfrozen between stages (lines 447, 721-774), NOT reset. The LoRA accumulates learning across all stages. This is correct — you want Stage 3's LoRA to build on Stage 1's learning.

2. **NAMM continues from prior stage:** `namm_trainer.start_iter` is updated to continue from where it left off (line 551). CMA-ES state persists across stages. This means Stage 2 warm-starts from Stage 0's final CMA-ES distribution. Correct.

3. **LR schedule resets:** A new `LoRAGradTrainer` is constructed each Stage B (line 760), which creates a new optimizer and scheduler. The LR resets to `warmup_ratio * learning_rate` at the start of each LoRA stage. With 2 loops, there are 2 warmup phases. This could be suboptimal (losing momentum), but it also prevents LR decay from making late-stage learning too slow.

4. **Batch size:** `_run_lora_stage` hard-codes `batch_size=1` (line 732). With `gradient_accumulation_steps=16`, effective batch = 16. This matches M3's constraint (NAMM active).

5. **eval_interval=999999:** Periodic eval is skipped during LoRA stages (line 739). Evaluation only happens after each complete outer loop (line 629-646). This is fine — it saves time and the per-loop eval is sufficient for tracking.

### `joint_default.yaml` vs spec mismatches

The `joint_default.yaml` has `learning_rate: 2e-4` (4x M1's 5e-5). The spec says M4 should use `learning_rate: 5e-5`. The corrected config `m4_joint_lora_5t.yaml` uses `5e-5`. **Use `m4_joint_lora_5t.yaml`, not `joint_default.yaml`, for M4.**

The `joint_default.yaml` has `max_seq_len: 3500`. M1 uses `max_seq_len: 7000`. The corrected `m4_joint_lora_5t.yaml` has `3500`. Given prompts are 4096-6500 tokens, `max_seq_len=3500` may truncate answer tokens. Check if this is intentional.

---

## 1g. Truncation Baselines Are Suspiciously Competitive

| Condition | Test F1 | Cache |
|-----------|---------|-------|
| Trunc/lora_m1 | 28.87 | 2048 |
| M3 LoRA + frozen NAMM | 31.06 | 2048 |
| A4 M3 LoRA, NAMM disabled | 33.91 | full (cs2048 ckpt) |

### How truncation is implemented

From `eval_namm_splits.py:107-122`, `--truncate_input_to N` monkey-patches the evaluator to slice `[:, -N:]` after tokenization. This keeps the **last N tokens** of each prompt — a tail-only truncation. Since LongBench QA tasks place the question at the end of the prompt, this preserves the question and the most recent context, dropping earlier document content.

This is **not** mid-cropping (which is the default for the NAMM evaluator's `use_mid_cropping=True`). Mid-cropping takes the first half and last half. Tail truncation only takes the end.

### Why A4 > M3

A4 evaluates the M3-trained LoRA with NAMM **disabled** (full cache). The M3 LoRA was trained under NAMM eviction, so it learned to work with ~1024 tokens. When you give it the full 6500 tokens, it has more information available and performs better (33.91 vs 31.06 for M3 at cs2048).

**This is actually good news for the paper narrative:** It shows M3's LoRA is robust — it was trained under harsh eviction but still performs well with full context. The paper can argue: "Training under NAMM teaches the LoRA to be robust to missing context. The LoRA transfers its improvements even when full context is restored."

### But truncation at 2048 gets 28.87 without any NAMM

True. However:
- Trunc/lora_m1 (28.87) uses M1's LoRA (trained on full context). NAMM-trained M3 gets 31.06 (+2.19) at the same budget.
- Trunc/plain (18.26 at 2048) vs M2 (17.40 at 2048): NAMM actually LOSES to simple truncation at cs2048. This is the more damaging finding.

**Paper framing suggestion:** At aggressive compression (cs1024), NAMM substantially outperforms truncation (20.30 vs 18.21). The gap narrows at cs2048 because less information is lost. The value of NAMM is most pronounced under tight memory budgets.

---

## Summary of Severity

| Issue | Severity | Action |
|-------|----------|--------|
| No run completed 150 epochs | **Paper-blocking** | Resume M1 and M3 from checkpoints |
| M1 vs M3 hyperparameter confound (LR, dropout) | **Paper-weakening** | Run M3 with M1's hyperparameters as additional condition |
| M1_recency all zeros (wrong eviction mode) | **Paper-blocking** | Re-run with `--use_classic_recency` |
| Val-test gap (~14 points) | **Paper-weakening** | Investigate chat template in eval; acknowledge in paper |
| M4 not run | **Paper-blocking** | Run M4 with `m4_joint_lora_5t.yaml` |
| A1 rank sweep incomplete | **Paper-weakening** | Run M1-r4 and M1-r16 |
| A4 uses M3 checkpoints, not M4 | **Paper-weakening** | Re-run A4 on M4 checkpoints after M4 completes |
| M2/cs2048 underperforms cs1024 | **Acknowledge** | Frame as "NAMM is most valuable under tight budgets" |
| Truncation competitive with NAMM at cs2048 | **Acknowledge** | Frame as expected — NAMM shines at aggressive compression |
