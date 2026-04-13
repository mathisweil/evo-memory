# Claude Code Task: Critical Experiment Review & Config Audit

## Your Role

You are reviewing a research project's experiments for an ACL paper submission. Your job is NOT just to create missing files — it's to **critically analyse whether the experiments we've already run are sound, whether the configs are correct, and whether the results are trustworthy**. Think like a reviewer who will reject the paper if the experimental setup is flawed.

---

## Step 0: Read Everything First

Before doing anything else, read all of these files thoroughly. You need the full picture before you can spot problems.

**Project docs** (read first):
- `README.md` — project structure, scripts, CLI args, config system
- `experiment_specification.md` — every experiment, exact hyperparams, fairness constraints, completion status with results
- `analysis_specification.md` — the analysis plan and what's been done

**All entry-point scripts** (read the actual Python source):
- `scripts/run_lora.py` — how does it parse configs? What keys does it expect? How does it handle NAMM during training?
- `scripts/run_namm.py` — Hydra-based, how does it compose configs?
- `scripts/run_joint.py` — how does alternation work? Does the LoRA stage inherit batch_size from M1 or use its own?
- `scripts/run_eval.py` — how does it handle cache_size=null vs cache_size=1024? How does it compute F1?
- `scripts/experiment_utils.py` — shared eval functions, metric computation

**All existing configs** (every single one):
- `scripts/configs/*.yaml` — all of them
- `config/config.yaml` and the entire `config/` Hydra tree (`config/run/`, `config/task/`, `config/model/`, `config/policy/`, `config/evolution/`, `config/trainer/`)

**Evaluation code**:
- `namm/evaluation/` — how is F1 computed? Is it micro-averaged? Token-level?
- `grad_lora_finetuning/trainer.py` — how does the training loop handle eval?
- `grad_lora_finetuning/datasets.py` — how are the SFT datasets constructed?

**Results** (if present locally):
- `results/main_table_5t/` — all subdirectories and their `results.json` files

---

## Step 1: Critical Analysis of Completed Experiments

This is the most important step. For each completed experiment, analyse whether the results are trustworthy and the setup is sound. Write your findings to `docs/experiment_critical_review.md`.

### 1a. Training Completeness — Did Anything Actually Finish?

The experiment spec targets 150 epochs for all LoRA conditions, but look at what actually happened:

- **M1 (r=8)**: Crashed 3 times, only reached **epoch 28** out of 150. Best val F1 = 45.48 at step 336. Was the learning curve still rising? Should this be trained to completion?
- **M3/cs1024**: Crashed at **epoch 25** out of 150. Best val F1 = 45.59 at step 340.
- **M3/cs2048**: Crashed at **epoch 15** out of 150. Best val F1 = 44.86 at step 244.
- **M3/cs3072**: "Finished" at epoch 4 (??). Val F1 = 33.37.

**Questions to answer:**
- Read the training code: at step 336 (M1) and step 340 (M3/cs1024), where in the learning curve are we? Did validation F1 plateau or was it still climbing?
- The spec says 150 epochs × ceil(306/16) = ~3000 steps. These runs stopped at ~340-684 steps. That's 10-22% of the planned training. Are the "best checkpoint" results from early training actually representative of what the fully-trained model would achieve?
- Is there code to resume from checkpoint? If so, should the recommendation be to resume rather than restart?

### 1b. Val-Test Gap — Why Is It So Large?

Look at these numbers:

| Condition | Best Val F1 | Test Micro F1 | Gap |
|-----------|-----------|-------------|-----|
| M1 (r=8) | 45.48 | 31.14 | **-14.3** |
| M3/cs1024 | 45.59 | 32.28 | **-13.3** |
| M2/cs1024 | 27.90 | 20.30 | **-7.6** |

A 14-point gap between val and test is enormous. This could indicate:
- **Different metrics**: Is val F1 computed differently from test F1? (e.g. macro vs micro, different tokenisation)
- **Different sample counts**: Val has 64 samples, test has 69/70 (the spec is inconsistent — check which is correct)
- **Overfitting to val set**: Best checkpoint is selected on val F1 — if the model overfits the val split, test performance drops
- **Evaluation code bug**: Read `run_eval.py` and the val-eval code in `run_lora.py` — are they using the same scoring function?

**Investigate**: Read the actual evaluation code path for (a) validation during training and (b) standalone test eval via `run_eval.py`. Are they identical? Report any differences.

### 1c. Hyperparameter Fairness — M1 vs M3 Are NOT Comparable

The FAIR-01 constraints require controlled comparisons, but M1 and M3 use **different hyperparameters**:

| Parameter | M1 | M3 | Problem? |
|-----------|----|----|----------|
| `learning_rate` | 5e-5 | 1e-4 | **2× higher for M3** — confounds NAMM effect with LR effect |
| `lora_dropout` | 0.1 | 0.05 | **Half for M3** — confounds NAMM effect with regularisation |
| `batch_size` | 4 | 1 | Different (grad_accum compensates to eff=16), but per-step gradient noise differs |
| `warmup_ratio` | 0.03 | not specified in spec | Is M3 using warmup? |
| `weight_decay` | 0.01 | not specified in spec | Is M3 using weight decay? |
| `max_grad_norm` | 1.0 | not specified in spec | Is M3 using gradient clipping? |

**This is a serious experimental confound.** If M3 outperforms M1, we can't cleanly attribute it to NAMM — it might just be the higher learning rate or lower dropout.

**Investigate**:
1. Read the M3 config file (`lora_rh_m4_instruct.yaml` or the `_5t` variant if it exists) — does it specify warmup, weight_decay, max_grad_norm? If not, what defaults does `run_lora.py` use?
2. Look at the actual WandB configs for the M3 runs (`ovosogkj`, `m4knrhmr`) if accessible — what hyperparameters were actually used?
3. **Assess**: Can we justify the LR difference (e.g. "M3 needs higher LR because the NAMM introduces noise and the model needs to adapt faster") or is this an oversight that invalidates the comparison?
4. **Recommend**: Should M3 be re-run with M1-identical hyperparameters? Or is the current setup defensible with caveats?

### 1d. FAIR-01 Memory Equivalence — M1 Eval Cache

The spec says: *"all conditions evaluated with cache_size=1024 at test time"*

But M1 is evaluated with **full cache** (no eviction). The spec acknowledges this, calling M1 "an upper bound." But this means the main results table is comparing:
- M3 with 1024-token KV cache vs M1 with ~6500-token KV cache

If M3 (32.28) beats M1 (31.14), that's impressive — but it would be even more informative to **also** eval M1 with cache_size=1024 using recency eviction. That would show: does NAMM-eviction + LoRA beat recency-eviction + LoRA? The M1_recency result would answer this, but it's broken (all zeros).

**Investigate**:
1. Why does M1_recency produce all zeros? Read the eval code path: when you load a LoRA checkpoint and apply recency eviction at eval time, what happens? Is there a code bug where the LoRA weights aren't loaded when a recency policy is active?
2. Should we add an M1 eval at cache_size=1024 with NAMM (not recency) eviction? i.e. load M1's LoRA checkpoint + M2's NAMM checkpoint at eval time, without having trained them together. This would be a new condition: "M1+M2 at eval only."

### 1e. M2 Cache Size Scaling — Does More Cache Help?

| Cache | M2 Test F1 | M3 Test F1 |
|-------|-----------|-----------|
| 1024 | 20.30 | 32.28 |
| 2048 | 17.40 | 31.06 |

M2 gets **worse** with a larger cache (20.30 → 17.40). M3 also gets slightly worse (32.28 → 31.06). This is counter-intuitive — more memory should help.

**Investigate**:
1. Is the cs2048 NAMM checkpoint actually good? M2/cs2048 best iter was 40 (vs 105 for cs1024). Check: did cs2048 training converge, or did it stop too early?
2. Read the NAMM training code: with cs2048, the compression ratio drops from 4× to ~3×. Does the CMA-ES objective change? Does the scoring function normalise by cache size?
3. Is this a dataset artefact? With contexts of 4096-6500 tokens and cs2048, only ~50-70% of tokens are evicted (vs 75-85% at cs1024). Maybe the NAMM is less useful when eviction is less aggressive.

### 1f. M4 Joint Training Design — Will It Work?

M4 hasn't been run yet. Before running it, critically assess the design:

1. **2 outer loops enough?** Stage 0: NAMM trains 100 iters from scratch on a random-LoRA model. Stage 1: LoRA trains 75 epochs with the Stage 0 NAMM active. Stage 2: NAMM re-trains 100 iters on the Stage 1 LoRA model. Stage 3: LoRA trains 75 more epochs.
   - In Stage 0, the NAMM is learning to evict tokens for a model that **hasn't been fine-tuned yet** (the base LLaMA). That NAMM policy may be completely wrong for the fine-tuned model.
   - Does `run_joint.py` reset the LoRA weights between stages, or does it continue from the previous stage's checkpoint? Read the code.
   - Does the NAMM in Stage 2 start from the Stage 0 checkpoint or from scratch? Read the code.

2. **Learning rate resets**: Does the LoRA LR schedule (warmup + decay) reset at each outer loop? Or does it continue? If it resets, the model gets 2× warmup phases. If it doesn't, Stage 3's LoRA training starts with a decayed LR.

3. **Effective batch size**: M1 uses batch_size=4, grad_accum=4 (effective=16). M4 runs LoRA with NAMM active, which means batch_size probably needs to drop to 1 (like M3) for memory. Does `run_joint.py` handle this? Does it use grad_accum=16 to compensate?

4. **Cold-start risk**: M4 cold-starts the NAMM (no pretrained checkpoint). M2 showed that NAMM needs ~105 iterations to find its best policy. With only 100 iterations per stage, Stage 0 may barely converge. **Consider**: should M4 use 3 outer loops instead of 2, or increase `namm_iterations_per_stage`?

### 1g. Truncation Baselines Are Suspiciously Competitive

| Condition | Test F1 | Cache |
|-----------|---------|-------|
| Trunc/lora_m1 | 28.87 | 2048 |
| M3 LoRA + frozen NAMM | 31.06 | 2048 |
| A4 M3 LoRA, NAMM disabled | 33.91 | full (cs2048 ckpt) |

Truncation at 2048 tokens gives 28.87. NAMM at 2048 gives 31.06. The gap is only ~2 F1 points. If you turn NAMM off entirely (A4), you get 33.91 — better than both.

This is a problem for the paper narrative. A reviewer will ask: "Why use a learned eviction policy when simple truncation gets you 90% of the way there, and removing the eviction policy entirely does even better?"

**Investigate**:
1. How is truncation implemented? Beginning, end, or middle? Read the code.
2. The A4 result (33.91 with NAMM disabled) is higher than M3 with NAMM enabled (31.06). This means M3's LoRA learned so well that it doesn't need NAMM at eval time. This actually *supports* the paper's argument — but only if we frame it correctly.
3. What story do these numbers tell? Perhaps: "Training under eviction teaches the LoRA to be robust to missing context, and this robustness transfers even when full context is available."

---

## Step 2: Config-vs-Code Audit

For every config file in `scripts/configs/` and every Hydra run preset in `config/run/`, verify:

1. **Key existence**: Does every key in the YAML actually correspond to a parameter that the script reads? Flag dead keys that are ignored by the script.
2. **Default inheritance**: When a CLI arg isn't specified and a config key is missing, what default does the script use? Are there cases where the config says one thing but the script default overrides it?
3. **Type correctness**: Are there any string/int/float mismatches? (e.g. `"1024"` vs `1024`)
4. **FAIR-01 compliance**: For each config, check if it enforces: `split_seed=42`, `train_frac=0.7`, `val_frac=0.15`, `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64`. If any of these are missing from the config, does the script default match?
5. **Config referenced in experiment spec but missing**: The spec references `lora_rh_m1_instruct_5t.yaml` and `lora_rh_m4_instruct_5t.yaml`. Do these `_5t` variants exist, or only the non-`_5t` versions? If only the non-`_5t` versions exist, how do they differ from what the spec requires?

Write findings to `docs/config_code_audit.md` with a table per config file.

---

## Step 3: Config Comparison Matrix

Create `docs/config_comparison_matrix.md` — a single table showing every hyperparameter across ALL conditions (B0, B1, M1, M2, M3, M4, A1, A4). This makes it immediately visible where conditions differ and whether those differences are intentional or accidental.

Format:

| Parameter | M1 (spec) | M1 (actual config) | M3 (spec) | M3 (actual config) | M4 (spec) | M4 (actual/default) | Match? | Notes |
|-----------|-----------|-------------------|-----------|-------------------|-----------|-------------------|--------|-------|
| learning_rate | 5e-5 | ? | 1e-4 | ? | 5e-5 | ? | ⚠️ M3 differs | Confound |
| lora_dropout | 0.1 | ? | 0.05 | ? | ? | ? | ⚠️ M3 differs | Confound |
| batch_size | 4 | ? | 1 | ? | ? | ? | ⚠️ | Memory constraint |
| warmup_ratio | 0.03 | ? | ? | ? | ? | ? | ❓ Unknown for M3 | |
| weight_decay | 0.01 | ? | ? | ? | ? | ? | ❓ Unknown for M3 | |
| ... | | | | | | | | |

The "actual config" column comes from reading the YAML files. The "Match?" column flags any discrepancy between spec and actual, or any inter-condition difference that isn't explained by design.

---

## Step 4: Investigate Specific Bugs and Anomalies

### 4a. M1_recency All Zeros
Read the eval code path for LoRA + recency eviction. Trace the exact sequence: model loading → LoRA weight application → recency policy attachment → forward pass → F1 scoring. The most likely causes:
- LoRA checkpoint not loaded when recency policy is active
- Recency eviction removes tokens that the LoRA-adapted attention patterns depend on, producing empty/degenerate outputs
- The recency policy is applied at a different stage than expected (before vs after LoRA forward pass)
- Generation produces empty strings → F1=0

Report what's happening with code references.

### 4b. Test Sample Count: 69 or 70?
The experiment spec says `306 train / 64 val / 69 test` in some places and `70 test` in others (the results table header says 70, the results.json schema says num_samples: 70). Read the data splitting code to determine the true count.

### 4c. Truncation Implementation
Read the eval code: how is truncation implemented? Does it truncate from the beginning, end, or middle? This matters because LongBench tasks often have the question at the end, so truncating from the end would destroy the query.

### 4d. M2 cs2048 Convergence
The M2/cs2048 best checkpoint is at iteration 40 out of 200. Check the WandB logs or training curves in the codebase: did performance plateau at iter 40 and stay flat, or did it oscillate? If CMA-ES kept exploring without improving, the best-at-40 checkpoint might just be noise.

---

## Step 5: Recommendations Document

Based on all the above analysis, create `docs/experiment_recommendations.md` with sections:

### 5a. Must-Fix Before Paper Submission
Issues that would cause a reviewer to reject. For each issue:
- What the problem is
- How severe it is (paper-blocking vs weakening)
- The recommended fix
- Estimated effort (hours/days)

### 5b. Should-Run Experiments (Prioritised)
For each recommended run:
- What it is and why it matters
- Dependencies (what must complete first)
- Exact command to run (with correct config)
- Estimated compute time
- What question it answers

### 5c. Config Fixes
For each config file, specify exactly what should change and why. Distinguish:
- **Fixes** — config is wrong/inconsistent with spec or code
- **Improvements** — config works but hyperparameters are questionable
- **New** — config doesn't exist yet and needs to be created

### 5d. Defensibility Arguments
For things we can't easily fix (like already-completed runs with suboptimal hyperparameters), draft the argument we'd make in the paper to defend the experimental design. Be honest about weaknesses a reviewer would spot.

### 5e. Paper Narrative Implications
Based on the results, what's the strongest story we can tell? The original hypothesis was "training under NAMM eviction helps the LoRA adapt to compressed context." But M3 > M1 even though M1 uses full cache. And A4 (NAMM off) > M3 (NAMM on). What does this mean for the narrative? Suggest how to frame these results positively and honestly.

---

## Step 6: Create/Fix Configs

**Only after completing Steps 1-5**, create or modify config files as needed. For each config you create or change:
1. State which analysis finding from Steps 1-5 motivates the change
2. Show what the old config had (if it existed) vs what the new one has
3. Confirm every key is actually read by the target script (cite the line in the Python source)

**Do NOT modify any config that was used for a completed run** — create new variants (e.g. `_5t_v2`) instead, to preserve reproducibility.

---

## Step 7: Master Run Script

Create `scripts/run_all_experiments.sh` that executes all remaining experiments in dependency order. It should incorporate any design changes from Step 5 (e.g. if you recommend 3 outer loops for M4, the script should use 3). Include smoke tests first, error handling, auto-eval after each training run, and a summary table at the end.

---

## Output Files

```
docs/
├── experiment_critical_review.md     # Step 1: deep analysis (MOST IMPORTANT)
├── config_code_audit.md              # Step 2: config-vs-code verification
├── config_comparison_matrix.md       # Step 3: cross-condition parameter table
├── experiment_recommendations.md     # Step 5: prioritised action items + narrative

scripts/
├── configs/                          # Step 6: only after analysis is done
│   └── (new or fixed configs)
├── run_all_experiments.sh            # Step 7: master runner
```

**The analysis docs (Steps 1-5) are the primary deliverable. Config creation (Steps 6-7) is secondary and must be justified by the analysis.**

---

## Key Principles

1. **Read code, don't guess.** Every claim about what the code does must reference the actual source file and ideally the function/line. Don't assume a config key works — verify the script reads it.

2. **Be a sceptical reviewer.** If a result looks too good or too bad, investigate why. If two conditions differ in ways that aren't controlled, flag it loudly.

3. **The experiment spec is the intent, the code and configs are the truth.** When they disagree, document the disagreement and say which should change.

4. **Completed runs are sacred.** Don't recommend re-running things unless the results are clearly invalid. Instead, recommend additional runs that fill the gaps or provide the missing controls.

5. **Think about the paper narrative.** The paper claims that training LoRA under NAMM-evicted context helps the model adapt to cache compression. Every experiment should support or challenge this claim in a controlled way. If the controls are broken, the claim is unsupported — flag this.

6. **Distinguish severity.** Not every issue is paper-blocking. Some are "acknowledge as limitation," some are "fix before submission," some are "nice to have." Be clear about which is which.
