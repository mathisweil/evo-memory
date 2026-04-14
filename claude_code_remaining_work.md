# Claude Code Task: Remaining Work — M4 Schedule, Bug Investigations & Run Script

## Context

The config review has been completed. The key changes already made:
- M3 config fixed: lr→5e-5, dropout→0.1 (matches M1)
- M1/M3 early_stopping_patience→20
- M4 joint: lora_eval_interval→14
- joint_default.yaml: safe defaults
- Docs produced: config_key_trace, metric_investigation, config_comparison_matrix, m3_rerun_plan, m4_readiness_review
- experiment_specification.md updated

The M4 readiness review confirmed:
- NAMM **warm-starts** between loops (uses previous loop's checkpoint)
- LoRA **continues** from previous loop's checkpoint
- LR schedule / warmup **restarts** each LoRA stage
- Early stopping is **NOT plumbed** through the joint training loop

This prompt covers everything that remains.

---

## 1. Update M4 Joint Config to 3 Outer Loops

Since the readiness review confirmed NAMM warm-starts between loops, **change `joint_lora_m4_5t.yaml` to use 3 outer loops**.

### Rationale

Empirical facts:
- M2 standalone NAMM: best at iteration ~105 out of 200. NAMM from scratch needs ~100 iterations.
- M1/M3 LoRA: early-stops at ~30 epochs (best val at ~epoch 17, patience exhausted by ~30).

With 3 loops: `3 × (67 NAMM + 50 LoRA) = 201 NAMM + 150 LoRA`

- **67 NAMM per stage:** Loop 0 from scratch is tight but doesn't need to fully converge — it just needs a reasonable starting policy. Loops 1-2 warm-start from the previous checkpoint and only need to adapt, so 67 iterations is plenty.
- **50 LoRA per stage:** LoRA early-stops at ~30 regardless. 50 gives 20 epochs headroom for the mechanism to trigger cleanly. Each loop trains under a progressively better NAMM.
- **2 co-adaptation cycles** instead of 1. With 2 loops, 50% of effective LoRA training is under a NAMM learned for the base model. With 3 loops, only 33%.

The total allocated budget (201 NAMM + 150 LoRA) matches M1+M2. Effective LoRA compute will be ~90 epochs (3 × ~30 before early-stop) vs M1's ~30. This is a feature: the changing NAMM prevents the overfitting that kills M1 early.

### Changes to `scripts/configs/joint_lora_m4_5t.yaml`

```yaml
num_outer_loops: 3               # was 2 — more co-adaptation cycles
namm_iterations_per_stage: 67    # was 100 — 3 × 67 = 201 ≈ 200
lora_epochs_per_stage: 50        # was 75 — 3 × 50 = 150
```

Also update `scripts/configs/joint_default.yaml` to match:
```yaml
num_outer_loops: 3
namm_iterations_per_stage: 67
lora_epochs_per_stage: 50
```

### Update experiment_specification.md

In the M4 section, update:
- The command to reflect `num_outer_loops: 3`
- The parameter table: `num_outer_loops: 3`, `namm_iterations_per_stage: 67` (3 × 67 = 201), `lora_epochs_per_stage: 50` (3 × 50 = 150)
- Add a note explaining the 3-loop rationale: "3 loops gives 2 co-adaptation cycles. Each LoRA stage early-stops at ~30 epochs; 50 allocated epochs provides headroom. Each NAMM stage warm-starts from the previous loop."
- Update the adapter checkpoint path: with `num_outer_loops: 3`, the final stage is `stage_2` (0-indexed), not `stage_1`.

---

## 2. Plumb Early Stopping into Joint LoRA Stages

The M4 readiness review found that early stopping is NOT passed through to the LoRA trainer inside `run_joint.py`. This means each LoRA stage runs all 50 allocated epochs even if the model converges at epoch 15.

### Investigate and act

Read `run_joint.py` — specifically the function that runs each LoRA stage. Find where it calls the LoRA trainer (likely `LoRAGradTrainer` from `grad_lora_finetuning/trainer.py`).

1. Does the trainer accept an `early_stopping_patience` argument?
2. Does `run_joint.py` pass it through?
3. If not, is it straightforward to add? (i.e., just passing `early_stopping_patience=config.get('early_stopping_patience', None)` to the trainer constructor)

**If it's a simple plumbing fix (< 20 lines):** Add it. Set `early_stopping_patience: 20` in `joint_lora_m4_5t.yaml` (or add a new key `lora_early_stopping_patience: 20`). This makes M4's LoRA stages behave like M1/M3.

**If it's complex (requires restructuring the training loop):** Don't add it. Document the asymmetry in `docs/m4_joint_training_analysis.md`:
- "M4 LoRA stages run for the full allocated epochs (50) because early stopping is not implemented in the joint training loop. M1/M3 early-stop at ~30. This means M4 trains longer per stage but may overfit in the tail of each stage. The `lora_eval_interval: 14` + best-checkpoint selection within each stage mitigates this — the saved checkpoint is the best seen, not the last."
- This is actually an acceptable argument: with eval_interval=14 and best-checkpoint saving, the model effectively gets early stopping via checkpoint selection even without explicit early stopping.

---

## 3. Update `.claude/rules/training.md`

Claude Code flagged that `.claude/rules/training.md` still documents M3 as `learning_rate=1e-4` (the old value). Update it to reflect the corrected configs:

Find any reference to M3 hyperparameters and update:
- `learning_rate: 5e-5` (matches M1)
- `lora_dropout: 0.1` (matches M1)
- Add note: "M3 hyperparameters MUST match M1 exactly except `namm_active: true` and `cache_size: 1024`"
- If the file mentions the M4 joint config, update it to reference `joint_lora_m4_5t.yaml` with 3 outer loops

Also update any references to `num_outer_loops: 2` → `3`, `namm_iterations_per_stage: 100` → `67`, `lora_epochs_per_stage: 75` → `50`.

---

## 4. Investigate M1_recency All-Zeros Bug

The experiment spec reports that M1_recency (M1 LoRA checkpoint evaluated with recency eviction at cache_size=1024) produces all-zero F1. This is an important missing data point — it would show whether NAMM-eviction + LoRA outperforms recency-eviction + LoRA.

### Trace the code path

Read `scripts/run_eval.py` and trace what happens when you:
1. Load an ES/LoRA checkpoint (`--es_checkpoint`)
2. Use a recency eviction policy (`--run_config recency_baseline_llama32_1b`)
3. Set `--cache_size 1024`

Specifically:
1. Is the LoRA checkpoint actually loaded when a recency run_config is used? Or does the recency config override/skip the checkpoint loading?
2. When recency eviction is active, how does it interact with the model's forward pass? Does it evict tokens BEFORE or AFTER the LoRA-adapted attention computation?
3. Is the generation output empty (producing empty strings → F1=0)? Or are the predictions non-empty but completely wrong?
4. Check the eval log for M1_recency if it exists — are there any error messages or warnings?

### Likely causes (in order of probability)

1. **LoRA weights not loaded with recency config:** The `recency_baseline_llama32_1b` Hydra preset might configure a different model wrapper that doesn't apply LoRA adapters. The base model without LoRA is much weaker, and recency eviction on top might collapse to zero.

2. **Generation collapse:** Recency eviction at 1024 tokens on a model trained with full context might cause attention to break — the model expects to attend to early context tokens that recency eviction removes. This produces degenerate repetitive output or empty strings.

3. **Cache interaction bug:** The LoRA changes attention patterns. Recency eviction (which doesn't know about the learned attention) removes tokens the LoRA-adapted model critically needs, producing worse-than-random output.

### Write findings

Write to `docs/m1_recency_investigation.md`:
- The code path analysis
- What you found
- Whether this is a code bug (fixable) or an expected behavioral result (the LoRA-adapted model genuinely fails under naive recency eviction)
- If it's expected behavior, this is actually a **useful finding for the paper**: it shows that LoRA-adapted attention patterns are incompatible with recency eviction, motivating learned eviction (NAMM)

---

## 5. Create Master Run Script

Create `scripts/run_all_experiments.sh` — a sequential script that runs every remaining experiment in dependency order.

### Remaining experiments to run

Based on the completion summary:

| Priority | Experiment | Command | Dependency | Est. time |
|----------|-----------|---------|------------|-----------|
| 1 | M3-matched cs1024 | See `docs/m3_rerun_plan.md` | M2 cs1024 checkpoint | ~6-10h |
| 2 | M3-matched cs2048 | See `docs/m3_rerun_plan.md` | M2 cs2048 checkpoint | ~8-12h |
| 3 | M1-r4 | `run_lora.py --config lora_rh_m1_instruct_5t.yaml --run_name m1_r4 --lora_rank 4 --lora_alpha 8` | none | ~6-10h |
| 4 | M1-r16 | `run_lora.py --config lora_rh_m1_instruct_5t.yaml --run_name m1_r16 --lora_rank 16 --lora_alpha 32` | none | ~6-10h |
| 5 | M4 joint | `run_joint.py --config joint_lora_m4_5t.yaml --run_name m4_joint_lora` | none | ~12-20h |
| 6 | A4-on (M4) | eval with M4 LoRA + M4 NAMM, cs1024 | M4 complete | ~1h |
| 7 | A4-off (M4) | eval with M4 LoRA only, no NAMM, full cache | M4 complete | ~1h |

### Script requirements

```bash
#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="${1:?Usage: $0 <experiment_dir>}"
```

The script should:

1. **Accept arguments:** `EXPERIMENT_DIR` (e.g., `experiments/experiment_5`), and optionally `--skip-smoke` to skip smoke tests.

2. **Run smoke tests first** (abort if any fail):
   ```bash
   # M1 smoke
   python scripts/run_lora.py --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
       --run_name smoke_m1 --num_epochs 1 --eval_interval 5 --no-gcs
   
   # M4 joint smoke
   python scripts/run_joint.py --config scripts/configs/joint_lora_m4_5t.yaml \
       --run_name smoke_m4 --num_outer_loops 2 --namm_iterations_per_stage 3 \
       --lora_epochs_per_stage 1 --population_size 2 --mini_batch_size 2
   ```

3. **Run experiments in dependency order** — experiments without dependencies can be listed sequentially (the script doesn't need to parallelise, just run one after another).

4. **After each training run, trigger eval:**
   ```bash
   python scripts/run_eval.py --config scripts/configs/eval_main_table.yaml \
       --es_checkpoint <path_to_best_ckpt> \
       --namm_checkpoint <if_applicable> \
       --cache_size <if_applicable> \
       --output_dir $EXPERIMENT_DIR/<condition>/eval
   ```

5. **After all runs, generate the cross-experiment report:**
   ```bash
   python scripts/generate_report.py \
       --experiment_dir $EXPERIMENT_DIR \
       --output $EXPERIMENT_DIR/paper_results.csv
   ```

6. **Log everything** to `$EXPERIMENT_DIR/run_all.log` (tee to stdout and file).

7. **Error handling:** If a run fails, log the failure and continue to the next independent run. Don't abort the whole script because M1-r4 failed — M4 can still run.

8. **Print summary at end** showing which experiments succeeded/failed.

### Checkpoint path conventions

Read `run_lora.py` and `run_joint.py` to understand where checkpoints are saved. The eval commands need to reference the correct paths. Common patterns:
- M1/M3 LoRA: `$EXPERIMENT_DIR/m1_lora_only/m1_r8/checkpoints/best_ckpt.pt`
- M4 joint LoRA: `$EXPERIMENT_DIR/joint_lora/m4_joint_lora/adapter/stage_2/` (stage_2 because 3 loops, 0-indexed)
- M4 joint NAMM: `$EXPERIMENT_DIR/joint_lora/m4_joint_lora/namm/latest.pt`

Verify these paths from the code before hardcoding them.

---

## 6. Verify Test Sample Count (69 vs 70)

The experiment spec is inconsistent: some places say 69 test samples, others say 70. The results table header says 70. Claude Code's previous run updated it to 69.

Read the data splitting code in `run_lora.py` (or wherever the 70/15/15 split happens) and compute the exact count:
- 5-task QA subset, filtered to [4096, 6500] tokens
- train_frac=0.7, val_frac=0.15, split_seed=42
- What's the total after filtering? What's ceil(total × 0.15) for val? What's the remainder for test?

If the existing `results.json` files say `num_samples: 70` or `num_samples: 69`, which is correct?

Write a one-paragraph finding to add to `docs/metric_investigation.md` (append, don't overwrite).

---

## 7. Write `docs/m4_joint_training_analysis.md`

Consolidate all M4-related findings into a single document:

1. **Schedule decision:** 3 loops, 67 NAMM + 50 LoRA per loop. Rationale from this prompt.
2. **Warm-start behavior:** (from m4_readiness_review) — NAMM warm-starts, LoRA continues, LR restarts.
3. **Early stopping status:** Whether it was plumbed through (from step 2 of this prompt), or documented as limitation.
4. **Compute budget:** Allocated = 201 NAMM + 150 LoRA (≈ M1+M2). Effective ≈ 201 NAMM + 90 LoRA (3 × ~30 epochs early-stop).
5. **Asymmetries with M1/M3:**
   - Stage-0 NAMM trains on un-adapted model (inherent to joint training)
   - LR warmup restarts each stage (3 warmup phases of ~1.5 epochs each vs M1's single 4.5-epoch warmup)
   - Early stopping: present or absent depending on step 2
6. **What to watch in WandB:**
   - Loop 0 NAMM val F1: if much worse than M2 at iteration 67, the budget is too tight
   - Per-loop LoRA val F1: should improve across loops
   - Per-loop eval_after_each_loop F1: the main metric, should show monotonic improvement
7. **Exact run command** with the final config values.

---

## Output Summary

```
scripts/configs/joint_lora_m4_5t.yaml    # Updated: 3 loops, 67 NAMM, 50 LoRA
scripts/configs/joint_default.yaml       # Updated: matching loop counts
scripts/run_all_experiments.sh           # New: master run script

.claude/rules/training.md               # Updated: M3 lr=5e-5, M4 3 loops

experiment_specification.md              # Updated: M4 section (3 loops, stage_2 paths)

docs/
├── m4_joint_training_analysis.md        # New: consolidated M4 analysis
├── m1_recency_investigation.md          # New: all-zeros bug investigation
├── metric_investigation.md              # Appended: sample count verification
```

If early stopping plumbing was added to run_joint.py:
```
scripts/run_joint.py                     # Modified: early_stopping_patience passed to LoRA trainer
scripts/configs/joint_lora_m4_5t.yaml    # Added: early_stopping_patience: 20
```
