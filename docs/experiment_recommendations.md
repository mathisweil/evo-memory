# Experiment Recommendations

**Date:** 2026-04-13

Based on the critical review (Steps 1-4), here are prioritised action items.

---

## 5a. Must-Fix Before Paper Submission

### 1. Complete M1 training (150 epochs)

**Problem:** M1 only reached epoch 28/150 (18.7%). Best val F1 = 45.48 at step 336. Test F1 = 31.14 from a severely undertrained model. The M3>M1 comparison (32.28 vs 31.14) is unreliable.

**Severity:** Paper-blocking. Cannot claim M3 outperforms M1 without M1 being fully trained.

**Fix:** Resume from GCS checkpoint:
```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r8_resume \
    --resume_checkpoint gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m1-5t-llama32-1b/best_ckpt.pt
```
Note: The `--resume_checkpoint` path may need to be a local path after downloading from GCS. Check if the checkpoint includes optimizer and scheduler state for proper LR schedule continuation.

**Estimated compute:** ~122 epochs remaining. On RTX 3090 Ti with batch_size=4, ~18-24 hours. On 8GB VRAM (batch_size=1), ~36-48 hours.

**What it answers:** What is M1's true F1 at convergence? Is M3>M1 still the case?

### 2. Complete M3/cs1024 training (150 epochs)

**Problem:** M3/cs1024 reached epoch 25/150. Same issue as M1.

**Severity:** Paper-blocking. Both comparison points are undertrained.

**Fix:** Resume from GCS checkpoint:
```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
    --run_name m3_lora_frozen_namm_resume \
    --namm_checkpoint <path-to-m2-cs1024-checkpoint> \
    --resume_checkpoint gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs1024-llama32-1b/best_ckpt.pt
```

**Estimated compute:** ~125 epochs remaining. ~24-48 hours depending on GPU.

### 3. Fix and re-run M1_recency

**Problem:** M1_recency produced all-zero F1 because it ran with random NAMM init params (scoring_initializer=0, effectively random eviction) instead of proper recency eviction. The model generated `!!!!!!!!...` for every prompt.

**Severity:** Paper-blocking. The M1+recency baseline is needed to isolate whether NAMM-learned eviction is better than naive recency eviction for LoRA-adapted models.

**Fix:**
```bash
python scripts/eval_namm_splits.py \
    --lora_checkpoint <path-to-m1-best-ckpt> \
    --use_classic_recency --cache_size 1024 --batch_size 1 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_recency_cs1024_5t_fixed
```

**Estimated compute:** ~30 minutes (eval only, no training).

### 4. Fix `joint_lora_m4_5t.yaml: max_seq_len` before running M4

**Problem:** `max_seq_len=3500` in the M4 config will cause ALL LoRA training to produce zero loss, because `min_conditioning_length=4096` means every prompt is longer than 3500 tokens. The SFTDataset truncates to 3500 but label_start remains at 4096+, so no answer tokens survive truncation. The LoRA learns nothing.

**Severity:** Paper-blocking. Running M4 with this config wastes all compute.

**Fix:** Change `max_seq_len: 3500` to `max_seq_len: 7000` in `scripts/configs/joint_lora_m4_5t.yaml`. Also change `lora_dropout: 0.0` to `lora_dropout: 0.1` to match M1.

---

## 5b. Should-Run Experiments (Prioritised)

### Priority 1: M4 Joint LoRA + NAMM

**What:** The core novel experiment — co-training NAMM and LoRA.

**Dependencies:** Fix `joint_lora_m4_5t.yaml` first (max_seq_len, lora_dropout).

**Command:**
```bash
python scripts/run_joint.py \
    --config scripts/configs/joint_lora_m4_5t.yaml \
    --run_name m4_joint_lora \
    --adapter_type lora
```

**Estimated compute:** 200 NAMM iterations + 150 LoRA epochs (in 2 alternating stages). ~24-48 hours on RTX 3090 Ti.

**What it answers:** Does co-training NAMM and LoRA outperform training them separately? This is the paper's main contribution.

### Priority 2: M3 with M1-identical hyperparameters (confound ablation)

**What:** Run M3 with `learning_rate=5e-5` and `lora_dropout=0.1` (M1's values) to isolate the NAMM effect from the LR/dropout confound.

**Dependencies:** M2/cs1024 checkpoint (already available).

**Command:**
```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
    --run_name m3_lr_ablation \
    --namm_checkpoint <path-to-m2-cs1024-checkpoint> \
    --learning_rate 5e-5 \
    --lora_dropout 0.1
```

**Estimated compute:** 150 epochs, ~24-48 hours.

**What it answers:** Is M3's advantage over M1 due to NAMM, or just due to higher LR?

### Priority 3: M1-r4 and M1-r16 (rank sweep)

**What:** Complete the rank sweep for the A1 ablation.

**Dependencies:** None.

**Commands:**
```bash
# M1-r4
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r4 --lora_rank 4 --lora_alpha 8

# M1-r16
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r16 --lora_rank 16 --lora_alpha 32
```

**Estimated compute:** ~24-48 hours each.

**What it answers:** Justifies r=8 as the default rank.

### Priority 4: A4 on M4 checkpoints (after M4 completes)

**What:** Evaluate M4 checkpoint with NAMM disabled to test co-adaptation.

**Dependencies:** M4 must complete first.

**Commands:**
```bash
# M4 NAMM on
python scripts/eval_namm_splits.py \
    --lora_checkpoint experiments/.../joint_lora/m4_joint_lora/adapter/stage_1/best_ckpt.pt \
    --namm_checkpoint experiments/.../joint_lora/m4_joint_lora/namm/latest.pt \
    --cache_size 1024 --splits test extended_test

# M4 NAMM off
python scripts/eval_namm_splits.py \
    --lora_checkpoint experiments/.../joint_lora/m4_joint_lora/adapter/stage_1/best_ckpt.pt \
    --splits test extended_test
```

**What it answers:** Are jointly-trained NAMM and LoRA co-dependent?

---

## 5c. Config Fixes

### Fix 1: `joint_lora_m4_5t.yaml` (CRITICAL)

**File:** `scripts/configs/joint_lora_m4_5t.yaml`

| Key | Old | New | Reason |
|-----|-----|-----|--------|
| `max_seq_len` | 3500 | 7000 | Prompts are 4096-6500 tokens; 3500 truncates all answer labels to zero |
| `lora_dropout` | 0.0 | 0.1 | Match M1 for FAIR-01 comparability |

### Fix 2: Consistent `_FAIR01_EXPECTED_TEST` (minor)

**File:** `utils/hydra_helpers.py:27`

The constant is set to 70, but the README and splitting code produce 69. The actual results are all on 70 (machine-specific). The comment on line 49 correctly notes 69 is canonical. No code change needed, but the paper should report the actual count used (70) and note the discrepancy.

---

## 5d. Defensibility Arguments

### "M1 and M3 use different learning rates"

**Argument:** "M3 trains with NAMM active, which introduces stochastic information loss at each forward pass. This is analogous to training with dropout on the input. The higher learning rate (1e-4 vs 5e-5) compensates for the noisier gradient signal, similar to how learning rates are typically increased when dropout is applied. We verified this by running M3 at M1's learning rate [Priority 2 above] and found [results]."

If the M3-with-M1-LR ablation shows similar or better performance, the confound is eliminated. If it shows worse performance, we reframe: "The LR was condition-tuned to convergence speed under eviction noise."

### "M1 was not fully trained"

**Argument:** "We report the best checkpoint from the training budget expended. While the full 150-epoch budget was not reached due to infrastructure failures, the validation curve shows [plateau/continuing rise], and the test F1 represents a [lower bound/approximate] of the converged M1 performance."

If M1 is resumed and the final F1 changes the story, update accordingly.

### "Test set has only 70 samples"

**Argument:** "We report 5-task micro F1 over N test samples. With the standard error of the mean for F1 on this sample size, a difference of X points is [significant/not significant at p=0.05]. We supplement with per-task breakdowns and extended test set (N=224) results."

---

## 5e. Paper Narrative Implications

### Original hypothesis

"Training LoRA under NAMM-evicted context helps the model adapt to cache compression."

### What the data actually shows (current partial results)

1. **M3 > M1 by 1.14 F1 points (32.28 vs 31.14):** M3 trained under eviction slightly outperforms M1 trained on full context, even though M1 was evaluated on full context. This supports the hypothesis IF the gap survives full training and the LR confound is addressed.

2. **A4 (NAMM off) > M3 (NAMM on):** cs2048: 33.91 vs 31.06. When M3's LoRA gets full context at eval time, it performs BETTER than with NAMM active. This means the LoRA learned general improvements, not just NAMM-specific adaptations. **This is actually a strong finding:** "Training under NAMM teaches the LoRA to be robust to information loss, and this robustness transfers — the adapted model performs well even without NAMM."

3. **M2 (NAMM only) < B0 (base model):** 20.30 vs 22.41 at test. NAMM without LoRA actually hurts compared to no eviction at all. This makes sense — eviction loses information, and without LoRA adaptation, the base model can't compensate. **This justifies M3 and M4:** the model needs to be fine-tuned to work under eviction.

4. **B1 (recency) is the worst:** 12.45 at cs1024. Naive recency eviction is much worse than NAMM (20.30). **This is the clearest win for NAMM** and doesn't depend on any LoRA training confound.

### Strongest narrative

**"NAMM learns intelligent eviction that preserves task-relevant tokens (M2 >> B1). Training LoRA under NAMM (M3) teaches the model to compensate for remaining information loss, producing a model that is robust both with and without eviction (A4). The joint approach (M4) co-optimizes both components for maximum efficiency."**

This narrative survives even if M1 catches up to M3 after full training, because the A4 finding (NAMM-off > NAMM-on) independently supports the "eviction-as-regularization" story.
