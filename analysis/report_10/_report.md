# Analysis 10: Maskfix Comparison — Buggy vs Fixed Attention Mask

## Overview

This analysis compares NAMM and LoRA+NAMM training between the original
implementation (with the attention mask bug causing entropy collapse from
chunk 9 onward) and the maskfix version (with correct attention throughout
prefill). All comparisons are at cache_size=1024.

> **M3 maskfix is still running** (step 294 of ~684, ~43% complete).
> The results below reflect the best checkpoint so far, not the final result.

## Results

### Best validation F1 (cs1024)

| Condition     | Qasper | 2WikiMQA | Qasper-E | HotpotQA-E | 2WikiMQA-E | Mean  |
| ------------- | -----: | -------: | -------: | ---------: | ---------: | ----: |
| B0            | 14.69 | 15.83 | 12.57 | 40.00 | 35.68 | 22.59 |
| M1            | 30.67 | 50.83 | 32.67 | 44.00 | 69.23 | 45.48 |
| M2 buggy      | 14.18 | 29.00 | 17.82 | 39.54 | 38.97 | 27.90 |
| M2 maskfix    | 8.62 | 20.00 | 8.81 | 14.00 | 23.08 | 14.90 |
| M3 buggy      | 21.60 | 51.11 | 39.40 | 59.67 | 56.15 | 45.59 |
| M3 maskfix*   | 21.86 | 63.06 | 25.76 | 74.00 | 75.64 | 52.06 |

*M3 maskfix is still running (step 294/~684). These are interim results.

### Key findings

1. **M2 maskfix (14.90) is substantially WORSE than M2 buggy (27.90).**
   The NAMM-only eviction policy performs worse with correct attention.
   This is surprising — better attention should help, not hurt.
   Possible explanations:
   - The CMA-ES optimisation landscape is different with correct attention,
     and the same hyperparameters (pop_size=8, sigma=0.065) may not be
     sufficient. The buggy regime's uniform attention may have been easier
     to optimise over (fewer "modes" to learn).
   - HotpotQA-E drops from 39.54 to 14.00 — the biggest single-task
     regression. Under buggy attention, NAMM's late-chunk scoring was
     effectively random, which may have accidentally preserved useful
     distractor-removal behaviour for HotpotQA-E.

2. **M3 maskfix (52.06) substantially EXCEEDS M3 buggy (45.59) and M1 (45.48).**
   Even at only 43% through training, the LoRA + fixed NAMM already
   outperforms both the buggy M3 and the full-context M1 by ~6.5 points.
   The biggest gains are on multi-hop tasks:
   - HotpotQA-E: 74.00 (maskfix) vs 59.67 (buggy) — +14.3 points
   - 2WikiMQA-E: 75.64 vs 56.15 — +19.5 points
   - 2WikiMQA: 63.06 vs 51.11 — +12.0 points

3. **Correct attention primarily helps the LoRA, not the NAMM policy.**
   M2 (NAMM-only) gets worse while M3 (LoRA+NAMM) gets much better.
   This suggests the LoRA adapter is the main beneficiary of correct
   prefill attention — it can now properly cross-reference question
   tokens with context during training, producing better gradient signal.

4. **The multi-hop tasks benefit most from the fix.** HotpotQA-E and
   2WikiMQA variants see the largest gains. These require comparing
   information across distant passages — exactly the kind of reasoning
   that requires functioning cross-attention during prefill.

## Figures

| File | Description |
| ---- | ----------- |
| `m2_buggy_vs_maskfix.png` | M2 mean val F1 over CMA-ES iterations |
| `m3_buggy_vs_maskfix.png` | M3 mean val F1 over training steps |
| `m2_per_task_buggy_vs_maskfix.png` | M2 per-task val F1 comparison |
| `m3_per_task_buggy_vs_maskfix.png` | M3 per-task val F1 comparison |
| `best_val_comparison.png` | Bar chart of best val F1 across all conditions |
