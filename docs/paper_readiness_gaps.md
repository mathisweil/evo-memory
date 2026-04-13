# Paper Readiness — Gap Analysis

Generated: 2026-04-13

---

## 1 Missing Experiments

| Experiment | Status | Blocking? | Estimated GPU-hours |
|-----------|--------|-----------|---------------------|
| M1-r4 | Not started | A1 ablation | ~6h (same as M1-r8) |
| M1-r16 | Not started | A1 ablation | ~6h |
| M4-LoRA (joint) | Not started | A4 proper, main table | ~12h (6h NAMM + 6h LoRA) |
| A1 (rank sweep analysis) | Blocked by M1-r4, M1-r16 | Discussion section | 0h (analysis only) |
| A4 on M4 (proper) | Blocked by M4 | Discussion section | ~1h (eval only) |

**Total remaining GPU-hours:** ~31h

Currently A4 was run on M3 checkpoints instead of M4. Once M4 is trained, A4 must be
re-run on the M4 checkpoint to answer the actual modularity question (RQ4).

---

## 2 Missing Configs (created in this audit)

| Config | Purpose | Created? |
|--------|---------|----------|
| `scripts/configs/lora_rh_m1_instruct_5t.yaml` | M1 FAIR-01 | YES |
| `scripts/configs/lora_rh_m4_instruct_5t.yaml` | M3 FAIR-01 | YES |
| `scripts/configs/joint_lora_m4_5t.yaml` | M4 FAIR-01 | YES |
| `scripts/configs/eval_main_table.yaml` | Unified eval | YES |

---

## 3 Missing Analyses

### 3a. Section 8 — Probing for residual knowledge (NOT DONE)

**Goal:** Train linear probes on frozen M1/M3 representations to test whether
information lost to eviction is still encoded in the residual stream.

**Feasibility:** Moderate. Requires:
- Extracting hidden states from specific layers (hook-based, ~100 LOC)
- Training sklearn linear probes on those features
- Designing probe tasks (e.g., "is entity X mentioned in evicted tokens?")

**Estimated effort:** 2-3 days coding + 4h GPU for extraction.

**Risk:** Probe design is subjective. If probes show no residual knowledge, the
negative result is hard to interpret (could be probe power, not model behavior).

### 3b. Section 9 — Gradient flow attribution (NOT DONE)

**Goal:** Trace how gradients flow through the model during M3 training to identify
which layers adapt most under eviction.

**Feasibility:** Hard. Requires:
- Gradient hooks on every layer during training (memory-intensive)
- Meaningful aggregation of per-layer gradient norms across training
- Comparison with M1 gradient patterns

**Estimated effort:** 3-4 days coding + significant GPU time for gradient collection.

**Risk:** High memory overhead with NAMM active. May need gradient checkpointing
adjustments that conflict with existing setup.

**Recommendation:** Section 8 is more feasible and directly addresses the paper's
core question. Section 9 is nice-to-have but high-risk for the time investment.
Consider dropping it if time is tight.

---

## 4 Results Inconsistencies

The experiment spec has two status sections that were written at different times:

- **Section 1 (Recommended Execution Order):** Implied B0/B1 not started.
- **Section 6 (Completed Runs):** Shows B0/B1 done with F1 scores.

**Ground truth (from §6):**

| Experiment | Status | Evidence |
|-----------|--------|---------|
| B0 | Done | F1 = 22.41, results in `results/main_table_5t/B0/` |
| B1 | Done | cs1024: 12.45, cs2048: 13.78 |
| M1-r8 | Done | F1 = 31.14, GCS checkpoint exists |
| M1-r4, M1-r16 | Not done | No WandB runs |
| M2 | Done | cs1024 (iter 105, val 27.90), cs2048 (iter 40, val 27.67) |
| M3 | Done | cs1024 (step 340, val 45.59), cs2048 (step 244, val 44.86) |
| M4 | Not started | No WandB runs |
| A4 | Done on M3 | cs1024: 28.82, cs2048: 33.91 (M3 ckpts, not M4) |

Section 6 is the ground truth. Section 1 is a template, not a status tracker.

---

## 5 Broken Runs

### M1_recency (all zeros)

`experiment_specification.md` reports M1_recency/cs1024 F1 = 0.00.

**Hypothesis:** The M1 LoRA checkpoint was trained with full cache (`namm_active=false`).
When evaluated under recency eviction at cs=1024, the model sees a radically different
token distribution than it was trained on. The LoRA adapter has learned to attend to
tokens that are now evicted, and the surviving tokens (most recent 1024) don't contain
the answer-relevant information the adapter expects.

However, all-zeros suggests a deeper issue — not just degraded performance, but total
failure. Possible causes:

1. **Cache validity mask bug:** The recency policy may not correctly set the cache
   validity mask, causing the model to attend to garbage memory positions.
2. **Position ID mismatch:** After recency eviction, position IDs may be discontinuous.
   The LoRA adapter (trained on contiguous positions) may produce degenerate attention.
3. **Generation collapse:** With wrong attention patterns, the model generates empty
   strings or repeated tokens that score 0 F1.

**Recommended investigation:** Run `scripts/check_eviction_stats.py` with the M1
adapter + recency policy to verify the cache state looks sane.

---

## 6 M3 Training Completeness

| Cache | Crashed at epoch | Best checkpoint step | Val F1 at best | Plateau? |
|-------|-----------------|---------------------|----------------|----------|
| 1024 | 25 | 340 (~14 ep) | 45.59 | Likely yes — best at epoch 14 of 25 run |
| 2048 | 15 | 244 (~10 ep) | 44.86 | Unclear — only 15 epochs completed |

For cs1024: best val F1 was at step 340 (epoch ~14), and the run continued to epoch 25
before crashing. If val F1 didn't improve from epoch 14 to 25, the model had plateaued
and more training would not help. Early stopping patience of 5 evals (at eval_interval=2,
that's 10 steps = ~0.5 epochs) would have stopped much earlier.

For cs2048: crashed at epoch 15, best at epoch ~10. Only 5 epochs of non-improvement.
Could benefit from more training, but the cs1024 pattern suggests early plateau is
typical for this setup.

**Recommendation:** The existing best checkpoints are likely sufficient. If time permits,
re-run cs2048 with `--resume_checkpoint` to confirm plateau.

---

## 7 Estimated Total Compute

| Task | GPU-hours (est.) |
|------|-----------------|
| M1-r4 | 6h |
| M1-r16 | 6h |
| M4 joint (2 x NAMM + 2 x LoRA) | 12h |
| A4 re-eval on M4 | 1h |
| A1 analysis (CPU) | 0h |
| Section 8 probing (extraction) | 4h |
| Buffer / debugging | 4h |
| **Total** | **~33h** |

Based on: M1-r8 WandB logs (28 epochs in ~segment 3, extrapolating to 150 epochs),
M2 NAMM ~6h for 200 gens at cs1024 (from README estimates).

---

## 8 Priority Order

1. **M1-r4, M1-r16** — unblocks A1, no dependencies, can run in parallel
2. **M4 joint** — unblocks proper A4, longest single run
3. **A4 on M4** — quick eval after M4 completes
4. **Section 8 probing** — if time permits
5. **Section 9 gradient flow** — drop if tight on time
