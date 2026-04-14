# Claude Code Task: Config Review, Repair & Completion

## Context

This is the `evo-memory` project — an ACL paper studying what happens when you train LoRA under NAMM KV-cache eviction. The core comparison is M1 (LoRA, no eviction) vs M3 (LoRA, frozen NAMM evicting during training). M4 (joint LoRA + NAMM) is planned but not yet run.

Read these files first:
- `README.md` — project structure, CLI args, config system
- `experiment_specification.md` — full experiment plan with results
- `analysis_specification.md` — analysis plan
- All scripts: `scripts/run_lora.py`, `scripts/run_joint.py`, `scripts/run_eval.py`, `scripts/run_namm.py`, `scripts/experiment_utils.py`
- All existing configs: `scripts/configs/*.yaml`
- Full Hydra config tree: `config/config.yaml`, `config/run/`, `config/task/`, `config/model/`, `config/policy/`, `config/evolution/`, `config/trainer/`
- Evaluation code: `namm/evaluation/`, `grad_lora_finetuning/trainer.py`, `grad_lora_finetuning/datasets.py`

**Read the actual Python source for every script before touching any config.** The YAML keys must match what the scripts actually parse. Dead keys that no script reads are worse than missing keys — they give false confidence.

---

## Part A: Issues I've Identified (Verify and Fix)

I've done a side-by-side comparison of all the configs. Below are the problems I found, ranked by severity. **For each one, verify in the source code whether my assessment is correct, then fix or document as indicated.**

---

### ISSUE 1 — CRITICAL: M3 hyperparameters break the M1-vs-M3 comparison

The entire paper hinges on comparing M1 (LoRA, full context) against M3 (LoRA, NAMM-evicted context). The ONLY difference should be whether NAMM is active during training. But the configs differ in two other ways:

| Parameter | M1 config | M3 config | Problem |
|-----------|-----------|-----------|---------|
| `learning_rate` | `5e-5` | `1e-4` | **M3 uses 2× the LR** |
| `lora_dropout` | `0.1` | `0.05` | **M3 uses half the dropout** |

If M3 outperforms M1, a reviewer will say: "That's because you gave M3 a higher learning rate, not because of NAMM." This is a fatal confound.

**The existing M3 runs used these mismatched hyperparameters.** Those results (cs1024: 32.28, cs2048: 31.06 test F1) cannot be used as the primary M1-vs-M3 comparison. They can be kept as a secondary data point (e.g. "M3 with tuned hyperparameters"), but the main comparison MUST use identical hyperparameters.

**Action:** Change `lora_rh_m4_instruct_5t.yaml` to match M1 exactly:
- `learning_rate: 5e-5` (was 1e-4)
- `lora_dropout: 0.1` (was 0.05)

This means M3 needs to be re-run with the corrected config. The old M3 runs become "M3-tuned" in the results table.

But before making this change, **verify in `run_lora.py`**: are there any code paths where NAMM being active changes the effective learning rate or gradient computation? For example, does the loss get divided by a different number of tokens when NAMM is evicting? If so, a different base LR might actually be necessary to get the same effective learning rate — but then the correction factor should be documented and principled, not a 2× hand-tuned bump.

---

### ISSUE 2 — CRITICAL: Missing data filtering params in LoRA configs

The FAIR-01 constraint requires `min_conditioning_length=4096` and `max_conditioning_length=6500` across all conditions. The M4 joint config (`joint_lora_m4_5t.yaml`) has these. But the M1 and M3 LoRA configs (`lora_rh_m1_instruct_5t.yaml`, `lora_rh_m4_instruct_5t.yaml`) **do not**.

These configs reference `run_config: namm_bam_i1_llama32_1b_5t` which is a Hydra preset that presumably contains the length filters. But `run_lora.py` is NOT a Hydra script — it uses argparse. 

**Investigate:** Read `run_lora.py` and trace how data is loaded. Specifically:
1. Does `run_lora.py` read the `run_config` key and load the corresponding Hydra YAML to get `min_conditioning_length` and `max_conditioning_length`?
2. Or does it expect these as explicit CLI args / config keys?
3. If `run_lora.py` doesn't use these params, how does it filter the dataset? Does the dataset module (`grad_lora_finetuning/datasets.py`) have its own filtering logic?
4. Check the actual completed M1 and M3 WandB runs — how many training samples did they use? If it's 306 (matching the expected 5-task split), the filtering is working. If it's a different number, the data may be wrong.

**Action:** If `run_lora.py` does NOT read these from the Hydra run_config, add `min_conditioning_length: 4096` and `max_conditioning_length: 6500` explicitly to both LoRA configs and ensure the script reads them.

---

### ISSUE 3 — HIGH: `early_stopping_patience: 5` is dangerously aggressive

Both M1 and M3 configs have `early_stopping_patience: 5` with `eval_interval: 14`.

That means: if val F1 doesn't improve for 5 consecutive evaluations (5 × 14 = 70 steps ≈ 3.5 epochs), training stops. Out of a planned 150-epoch schedule, this could halt training at epoch 7 if the model hits an early plateau before a second learning phase.

**But the completed runs tell a different story:** M1 ran to step 684 (epoch 28) with best at step 336 (epoch ~17). That's 348 steps = ~25 evaluations past the best. If patience=5 were active, it would have stopped at step ~406 (epoch ~20). The run went well past that.

**Investigate:** 
1. Read `run_lora.py` — is `early_stopping_patience` actually implemented? If yes, why didn't M1 stop at step ~406?
2. Possibilities: (a) early stopping isn't implemented despite the config key existing, (b) the counter uses a different metric (exact match instead of F1?), (c) small fluctuations kept resetting the patience counter.
3. If early stopping IS implemented and functional, `patience: 5` is too aggressive. With 20 steps per epoch and eval every 14 steps, 5 evals = 70 steps = 3.5 epochs. For a research experiment targeting 150 epochs, this is wrong. 

**Action:** 
- If early stopping is implemented: increase to `early_stopping_patience: 20` (= 280 steps ≈ 14 epochs). This gives the model enough room to recover from plateaus while still preventing runaway overfitting.
- If early stopping is NOT implemented: remove the key from all configs (dead keys are misleading) and add a comment saying it's not supported.
- Whatever the answer, it MUST be identical across M1, M3, and M4.

---

### ISSUE 4 — HIGH: M4 has no mid-stage checkpointing

M1 and M3 use `eval_interval: 14` — they evaluate on val every 14 training steps and save the best checkpoint. This means the saved model is the best seen during training.

M4 (`joint_lora_m4_5t.yaml`) has `lora_eval_interval: 999999` — effectively **no evaluation during LoRA stages**. Each 75-epoch LoRA stage runs blind. Only `eval_after_each_loop` saves results (once after all 75 epochs).

This creates an asymmetry: M1 and M3 get best-of-N checkpointing. M4 gets the last checkpoint of each stage, which might be overfit.

**Investigate:** Read `run_joint.py`:
1. During the LoRA stages, does it call the trainer with an eval loop? Or does it just train?
2. Does `eval_after_each_loop` use the FINAL model from the LoRA stage, or does it pick the best?
3. Is there a mechanism to do best-checkpoint selection within a LoRA stage inside joint training?

**Action:** If `run_joint.py` supports mid-stage eval (i.e., it passes `eval_interval` to the LoRA trainer), set `lora_eval_interval: 14` to match M1/M3. If it doesn't, document this as a limitation.

---

### ISSUE 5 — HIGH: `batch_size` discrepancy between spec and config (M1)

The experiment spec says M1 should use `batch_size: 4, gradient_accumulation_steps: 4`. The actual config has `batch_size: 1, gradient_accumulation_steps: 16`. Both give effective batch = 16.

Mathematically the gradients are identical (sum of 16 per-sample gradients). But:
- `batch_size=4` is faster (GPU parallelism), reasonable since M1 has no NAMM and plenty of memory
- `batch_size=1` is consistent with M3 and M4 (which need batch_size=1 for memory with NAMM active)

**The config choice (batch_size=1 everywhere) is actually BETTER than the spec for fairness**, because now M1, M3, and M4 have identical batch processing. The spec is wrong.

**Action:** Keep `batch_size: 1, gradient_accumulation_steps: 16` in all LoRA configs. Update `experiment_specification.md` to reflect the actual config values, not the planned ones. Add a comment in M1's config explaining why batch_size=1 is used despite having available memory: "Matches M3/M4 for controlled comparison."

---

### ISSUE 6 — MEDIUM: `eval_interval: 14` vs spec's `2`

The experiment spec says `eval_interval: 2` for M1 and M3. The actual configs have `eval_interval: 14`. 

With ~20 steps per epoch, eval_interval=2 means evaluating every ~0.1 epochs = 1500 total evals over 150 epochs. Each eval does inference on 64 val samples. That's extremely frequent and would dominate wall-clock time.

`eval_interval: 14` means eval every ~0.7 epochs = ~214 evals total. This is practical and still catches the best checkpoint with reasonable granularity.

**Action:** Keep `eval_interval: 14` in all configs. This is a sensible practical choice. But ensure it's **identical across M1, M3, and M4**. Currently M1=14, M3=14, M4=999999. M4 should also be 14 (see Issue 4).

Update `experiment_specification.md` to say `eval_interval: 14` instead of 2.

---

### ISSUE 7 — MEDIUM: No `seed` in M1 or M3 configs

The experiment spec says `seed: 42` for LoRA conditions. Neither `lora_rh_m1_instruct_5t.yaml` nor `lora_rh_m4_instruct_5t.yaml` specifies a seed.

**Investigate:** Read `run_lora.py` — does it have a `--seed` argument? What's the default? If the default is 42, this is fine but should be made explicit. If it's different or random, this is a reproducibility problem.

**Action:** Add `seed: 42` to both LoRA configs. If `run_lora.py` doesn't support a seed arg, add one.

---

### ISSUE 8 — MEDIUM: `batch_size_eval` inconsistency

M1 has `batch_size_eval: 1`, M3 has `batch_size_eval: 2`. This shouldn't affect results (inference is deterministic with temperature=0.0), but it's inconsistent and could indicate that someone tuned M3 differently.

Counter-intuitively, M3 (with NAMM active, so smaller KV cache per sample) can fit a larger eval batch. So `batch_size_eval: 2` for M3 is technically justified by memory.

**Action:** Set `batch_size_eval: 2` for both M1 and M3 (M1 has even more memory available with no NAMM). If M1 can't fit 2 (unlikely with no NAMM), keep at 1 for both. The key is consistency.

**Investigate:** Does `run_eval.py` (standalone eval) also use this parameter? Or does it have its own batch_size? The eval config (`eval_main_table.yaml`) has `batch_size: null`. Check what null resolves to.

---

### ISSUE 9 — MEDIUM: `wandb_project` mismatch

M1 and M3 configs say `wandb_project: Experiments`. But existing completed runs are in project `memory_evolution_hf` (per experiment spec §6). If the project name changed, new runs will go to a different project and won't be next to the old runs in WandB.

**Investigate:** Check what project the existing runs (`kz6vqo2o`, `ovosogkj`) are actually in. Set the config to match.

**Action:** Set `wandb_project` to whatever the existing runs use. Consistency with past runs matters more than a new name.

---

### ISSUE 10 — LOW: `joint_default.yaml` has dangerous defaults

`joint_default.yaml` has `learning_rate: 2e-4`, `lora_dropout: 0.0`, `max_seq_len: 3500`, `cache_size: null`. If someone accidentally uses this instead of `joint_lora_m4_5t.yaml`:
- LR is 4× too high
- No dropout regularization  
- `max_seq_len: 3500` would truncate every context (they're 4096–6500 tokens)
- `cache_size: null` means no NAMM eviction, defeating the purpose

The M4-specific config (`joint_lora_m4_5t.yaml`) fixes all of these. But the default is a foot-gun.

**Action:** Update `joint_default.yaml` to have sensible defaults that at least won't silently produce garbage. Set `max_seq_len: 7000`, `learning_rate: 5e-5`, `lora_dropout: 0.1`, `cache_size: 1024`. Add a prominent comment: "For M4-LoRA runs, use joint_lora_m4_5t.yaml instead."

---

## Part B: Additional Verification Tasks

Beyond the issues above, do the following:

### B1. Trace every config key through the code

For EACH config file (`lora_rh_m1_instruct_5t.yaml`, `lora_rh_m4_instruct_5t.yaml`, `joint_lora_m4_5t.yaml`, `eval_main_table.yaml`, `eval_default.yaml`):

1. List every YAML key
2. Find where in the Python source it's read (file, function, line)
3. Flag any key that is NOT read by any script (dead key)
4. Flag any script parameter that has NO corresponding config key (missing key)
5. Check if the YAML value type matches what the script expects (e.g., int vs string, list vs string)

Write the results to `docs/config_key_trace.md`.

### B2. Check the Hydra run preset

Read `config/run/namm_bam_i1_llama32_1b_5t.yaml` (if it exists). This is referenced by every config. Document what it contains — especially:
- `min_conditioning_length`
- `max_conditioning_length`  
- `max_memory_length` / `cache_size`
- `max_answer_tokens`
- `batch_size`
- `max_new_tokens` (for generation)
- The 5 tasks it specifies

Also check: `config/run/full_cache_baseline_llama32_1b.yaml` and `config/run/recency_baseline_llama32_1b.yaml` — do these exist for B0/B1?

### B3. Understand the val-test metric gap

The completed results show huge val-test gaps (val F1 45 → test F1 31). Before we re-run anything:

1. Read how val F1 is computed during training (`grad_lora_finetuning/trainer.py` or wherever the training loop calls eval)
2. Read how test F1 is computed in standalone eval (`scripts/run_eval.py`)
3. Are they using the same function? Same metric (micro vs macro)? Same tokenization for F1?
4. What is `val_tasks_aggregate` (used in NAMM) vs `avg F1` (used in LoRA val)? Are these the same metric?

Write findings to `docs/metric_investigation.md`. If the val and test metrics are computed differently, that's a finding that needs to be in the paper.

### B4. Check if temperature is enforced during val eval

The FAIR-01 constraint says `temperature=0.0` for all evaluations. The eval configs set this. But what about val eval DURING training? Read `run_lora.py` and the trainer — does it use greedy decoding for val eval? Or does it use a different temperature?

If val eval uses a different temperature, that would explain part of the val-test gap (val with sampling has higher variance → lucky high scores).

---

## Part C: Config Fixes and Completion

Based on Parts A and B, update every config file. The goal is: **every config should be complete, self-contained, and correct.** No parameter should rely on implicit defaults in the script — if the script has a default, the config should explicitly set the value anyway, so the config is the single source of truth.

### C1. Fix `lora_rh_m4_instruct_5t.yaml` (M3)

Critical changes:
```yaml
learning_rate: 5e-5      # was 1e-4 — MUST match M1 for fair comparison
lora_dropout: 0.1        # was 0.05 — MUST match M1 for fair comparison
```

Add missing params (copy from M1 where identical):
```yaml
seed: 42
min_conditioning_length: 4096    # FAIR-01 — add if run_lora.py supports it
max_conditioning_length: 6500    # FAIR-01 — add if run_lora.py supports it
```

Change `batch_size_eval` to match M1 (or set both to the safe maximum).

If `early_stopping_patience` is functional, increase to 20. If it's a dead key, remove it.

### C2. Fix `lora_rh_m1_instruct_5t.yaml` (M1)

Add missing params:
```yaml
seed: 42
min_conditioning_length: 4096    # FAIR-01 — add if run_lora.py supports it
max_conditioning_length: 6500    # FAIR-01 — add if run_lora.py supports it
```

Match `batch_size_eval` with M3.

If `early_stopping_patience` is functional, increase to 20. If dead key, remove.

### C3. Fix `joint_lora_m4_5t.yaml` (M4)

Changes:
```yaml
lora_eval_interval: 14   # was 999999 — match M1/M3 for consistent checkpointing
```

Add missing params:
```yaml
seed: 42                 # add if run_joint.py supports it
```

Verify that `lora_batch_size: 1` and `gradient_accumulation_steps: 16` give effective_batch=16 in the joint training loop (it might compute this differently).

### C4. Fix `joint_default.yaml`

Update dangerous defaults:
```yaml
max_seq_len: 7000        # was 3500 — 3500 truncates all 4096-6500 token contexts
learning_rate: 5e-5      # was 2e-4
lora_dropout: 0.1        # was 0.0
cache_size: 1024         # was null
```

### C5. Fix `eval_default.yaml` and `eval_main_table.yaml`

These look mostly correct. Verify that every key is read by `run_eval.py`. Add any missing keys that `run_eval.py` supports.

### C6. After all fixes, produce a comparison matrix

Create `docs/config_comparison_matrix.md` — a single table showing every parameter across ALL conditions. Every value should come from the actual config file (not the spec):

| Parameter | M1 | M3 | M4 (joint) | M4 eval | Match? |
|-----------|----|----|------------|---------|--------|
| learning_rate | 5e-5 | 5e-5 | 5e-5 | — | ✅ |
| lora_dropout | 0.1 | 0.1 | 0.1 | — | ✅ |
| batch_size | 1 | 1 | 1 | — | ✅ |
| ... | | | | | |

Flag any remaining differences with justification (e.g., `namm_active: true` for M3 is intentional, not a confound).

---

## Part D: Investigate and Document

### D1. The M3-rerun question

The existing M3 results used lr=1e-4 and dropout=0.05 (i.e., the current broken config). After fixing the config (Issue 1), M3 needs to be re-run.

**But**: we should keep the old M3 results too. They tell a useful story: "M3 with separately tuned hyperparameters achieves X, while M3 with M1-matched hyperparameters achieves Y." If Y > M1, the NAMM effect is clean. If Y < M1 but X > M1, the improvement came from hyperparameter tuning, not NAMM.

Create a plan in `docs/m3_rerun_plan.md`:
1. The corrected M3 config (matching M1 exactly except NAMM)
2. The exact command to run it
3. How to name it in WandB to distinguish from the old runs
4. What to do with the old M3 results (keep as "M3-tuned" ablation)

### D2. M4 joint training readiness review

Review `run_joint.py` source code and answer:
1. Does the NAMM stage in loop 0 train on the base (unfinetuned) model? If yes, is that NAMM policy useful at all when the LoRA stage changes the model?
2. Does the NAMM in loop 1 start from the loop 0 checkpoint, or from scratch?
3. Does the LoRA in loop 1 start from the loop 0 LoRA checkpoint, or from scratch?
4. Does the LR schedule (warmup + decay) reset per loop?
5. What batch_size does the LoRA stage use? Does it match the config's `lora_batch_size`?

Write findings to `docs/m4_readiness_review.md` with a recommendation: is the M4 config ready to run, or does it need changes?

### D3. Update experiment_specification.md

After all config changes, update `experiment_specification.md` to reflect reality:
1. Fix the batch_size values (spec says 4 for M1, actual config uses 1)
2. Fix the eval_interval (spec says 2, actual config uses 14)
3. Note that M3 will be re-run with matched hyperparameters
4. Add the corrected hyperparameter table for M3
5. Mark old M3 results as "M3-tuned (lr=1e-4, dropout=0.05)" in the results table

---

## Output Summary

```
docs/
├── config_key_trace.md             # B1: every key traced through code
├── metric_investigation.md         # B3: val vs test F1 computation
├── config_comparison_matrix.md     # C6: cross-condition parameter table
├── m3_rerun_plan.md                # D1: plan for corrected M3
├── m4_readiness_review.md          # D2: joint training code review

scripts/configs/
├── lora_rh_m1_instruct_5t.yaml    # C2: completed and fixed
├── lora_rh_m4_instruct_5t.yaml    # C1: critical fixes (lr, dropout)
├── joint_lora_m4_5t.yaml          # C3: lora_eval_interval fix
├── joint_default.yaml             # C4: safe defaults
├── eval_default.yaml              # C5: verified
├── eval_main_table.yaml           # C5: verified

experiment_specification.md         # D3: updated to match actual configs
```

---

## Principles

1. **The independent variable between M1 and M3 is NAMM on/off. Nothing else.** Every other hyperparameter must be identical. Any difference is a confound.

2. **Read the code before changing anything.** A config key only matters if the script reads it. A missing config key only matters if the script has a bad default.

3. **Completed runs are not invalidated, they're relabelled.** The old M3 runs (lr=1e-4) become "M3-tuned." The new M3 runs (lr=5e-5) become the primary comparison.

4. **Every config should be the single source of truth.** No parameter should rely on "the script default is probably right." If a script defaults to seed=42, the config should explicitly say `seed: 42`.

5. **When in doubt, match M1.** M1 is the anchor condition. M3 matches M1 in everything except NAMM. M4 matches M1 in LoRA hyperparameters. Deviations from M1 must be justified by NAMM memory constraints (e.g., batch_size=1), not by tuning.
