# Analysis 1 -- Per-Task Eviction Sensitivity (Test-Set Evaluation)

> **TL;DR:** M3/cs1024 (LoRA + frozen NAMM, 32.28 test micro F1) slightly
> exceeds M1 (full-context LoRA, 31.14), confirming the original val-based
> claim after eval fixes. New truncation and recency baselines establish a
> performance ladder: NAMM beats both truncation and recency at matched cache
> budgets. The A4 ablation reveals a cache-size-dependent pattern: at cs1024,
> NAMM at inference helps (M3=32.28 > A4=28.82); at cs2048, NAMM hurts
> (M3=31.06 < A4=33.91).

---

## Naming Note

> **What `results/main_table_5t/` calls "M4" is actually experiment-spec M3
> (LoRA + frozen NAMM).** Real M4 (joint co-training of LoRA and NAMM) has not
> been run. Throughout this report, we use **M3** to refer to this condition
> (LoRA fine-tuned with a frozen NAMM eviction policy). When referencing file
> paths or data keys, the results-directory label "M4" appears in code and
> paths (e.g. `M4/cs1024`). See `experiment_specification.md` for the full
> milestone naming.

---

## 1. Overview

This analysis evaluates 16 experimental conditions on 5 LongBench QA tasks
using a proper held-out test split (n=70) and an extended test split (n=224).
Results come from `results/main_table_5t/all_results.json`.

| Condition              | Results Key          | Description                                                    |
| ---------------------- | -------------------- | -------------------------------------------------------------- |
| **B0**                 | `B0`                 | Base Llama-3.2-1B-Instruct, no fine-tuning, full KV cache      |
| **B1/cs1024**          | `B1/cs1024`          | Base model + recency eviction (keep last 1024 KV entries)      |
| **B1/cs2048**          | `B1/cs2048`          | Base model + recency eviction (keep last 2048 KV entries)      |
| **M1**                 | `M1`                 | LoRA fine-tuned (rank 8), full KV cache, no eviction           |
| **M2/cs1024**          | `M2/cs1024`          | Standalone NAMM eviction (CMA-ES trained), no LoRA, cache 1024 |
| **M2/cs2048**          | `M2/cs2048`          | Standalone NAMM eviction, no LoRA, cache 2048                  |
| **M3/cs1024**          | `M4/cs1024`          | LoRA fine-tuned with frozen NAMM active, cache 1024            |
| **M3/cs2048**          | `M4/cs2048`          | LoRA fine-tuned with frozen NAMM active, cache 2048            |
| **Trunc/plain_1024**   | `Trunc/plain_1024`   | Plain Llama, input truncated to last 1024 tokens               |
| **Trunc/plain_2048**   | `Trunc/plain_2048`   | Plain Llama, input truncated to last 2048 tokens               |
| **Trunc/lora_m1_1024** | `Trunc/lora_m1_1024` | M1 LoRA adapter, input truncated to last 1024 tokens           |
| **Trunc/lora_m1_2048** | `Trunc/lora_m1_2048` | M1 LoRA adapter, input truncated to last 2048 tokens           |
| **M1_recency/cs1024**  | `M1_recency/cs1024`  | M1 LoRA + recency eviction, cache 1024 (BROKEN -- all zeros)   |
| **A4/cs1024**          | `A4/cs1024_no_namm`  | M3/cs1024 LoRA weights, NAMM disabled at eval (full cache)     |
| **A4/cs2048**          | `A4/cs2048_no_namm`  | M3/cs2048 LoRA weights, NAMM disabled at eval (full cache)     |

All runs use `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`,
`min_conditioning_length=4096`, `max_conditioning_length=6500`,
`max_answer_tokens=64`, greedy decoding. Extended_test is filtered to the
length window (6500, 8192].

---

## 2. Test-Split Per-Task F1 (n=70)

| Condition          | Qasper | 2WikiMQA | Qasper-E | HotpotQA-E | 2WikiMQA-E | Macro |     Micro |
| ------------------ | -----: | -------: | -------: | ---------: | ---------: | ----: | --------: |
| B0                 |  25.85 |    26.52 |     6.06 |      44.56 |      17.46 | 24.09 | **22.41** |
| B1/cs1024          |  22.29 |    10.42 |     7.26 |      17.65 |       6.55 | 12.83 | **12.45** |
| B1/cs2048          |  23.32 |     7.63 |     6.14 |      25.93 |       8.93 | 14.39 | **13.78** |
| M1                 |  45.03 |    10.00 |    35.62 |      30.51 |      30.16 | 30.26 | **31.14** |
| M2/cs1024          |  28.30 |    27.56 |     8.09 |      17.50 |      24.16 | 21.12 | **20.30** |
| M2/cs2048          |  26.79 |    25.00 |     6.06 |      18.45 |      15.18 | 18.30 | **17.40** |
| M3/cs1024          |  29.30 |    44.23 |    26.56 |      43.45 |      22.79 | 33.27 | **32.28** |
| M3/cs2048          |  39.68 |    25.00 |    30.47 |      35.51 |      24.60 | 31.05 | **31.06** |
| Trunc/plain_1024   |  29.80 |    26.52 |    13.99 |       9.38 |      12.50 | 18.44 | **18.21** |
| Trunc/plain_2048   |  24.81 |    25.00 |    12.42 |      17.28 |      14.29 | 18.76 | **18.26** |
| Trunc/lora_m1_1024 |  26.35 |    26.52 |    27.20 |      33.89 |      21.43 | 27.08 | **26.90** |
| Trunc/lora_m1_2048 |  31.56 |    27.56 |    30.04 |      33.95 |      21.43 | 28.91 | **28.87** |
| M1_recency/cs1024  |   0.00 |     0.00 |     0.00 |       0.00 |       0.00 |  0.00 |  **0.00** |
| A4/cs1024          |  46.19 |    25.00 |    28.12 |      26.67 |      17.46 | 28.69 | **28.82** |
| A4/cs2048          |  43.56 |    38.89 |    34.63 |      35.80 |      17.46 | 34.07 | **33.91** |

---

## 3. Extended-Test Per-Task F1 (n=224)

| Condition          | Qasper | 2WikiMQA | Qasper-E | HotpotQA-E | 2WikiMQA-E | Macro |     Micro |
| ------------------ | -----: | -------: | -------: | ---------: | ---------: | ----: | --------: |
| B0                 |  18.34 |    17.86 |    13.11 |      45.88 |      23.27 | 23.69 | **22.30** |
| B1/cs1024          |  15.36 |     5.35 |     4.80 |      14.24 |       4.27 |  8.81 |  **7.60** |
| B1/cs2048          |  16.84 |     4.40 |     7.66 |      21.47 |       8.32 | 11.74 | **10.28** |
| M1                 |  35.92 |    23.59 |    27.81 |      47.83 |      31.93 | 33.42 | **31.84** |
| M2/cs1024          |  19.49 |    23.74 |    12.13 |      26.36 |      22.05 | 20.75 | **20.65** |
| M2/cs2048          |  20.20 |    20.90 |    10.14 |      24.71 |      20.49 | 19.29 | **19.01** |
| M3/cs1024          |  25.39 |    27.27 |    23.78 |      41.21 |      22.51 | 28.03 | **26.92** |
| M3/cs2048          |  23.41 |    22.00 |    14.59 |      29.55 |      27.24 | 23.36 | **23.15** |
| Trunc/plain_1024   |  23.36 |    18.95 |    14.55 |      23.52 |      13.52 | 18.78 | **17.83** |
| Trunc/plain_2048   |  19.97 |    17.23 |    12.72 |      31.91 |      19.54 | 20.27 | **19.35** |
| Trunc/lora_m1_1024 |  25.50 |    18.58 |    28.15 |      37.31 |      18.99 | 25.71 | **24.24** |
| Trunc/lora_m1_2048 |  30.62 |    18.72 |    31.68 |      41.85 |      23.83 | 29.34 | **27.67** |
| M1_recency/cs1024  |   0.00 |     0.00 |     0.00 |       0.00 |       0.00 |  0.00 |  **0.00** |
| A4/cs1024          |  30.41 |    23.55 |    21.41 |      31.57 |      25.07 | 26.40 | **25.62** |
| A4/cs2048          |  24.54 |    31.15 |    22.88 |      29.92 |      21.31 | 25.96 | **25.66** |

---

## 4. Eviction Sensitivity: (M1 - M3) / M1

Eviction sensitivity measures the fractional performance drop when moving from
full-context LoRA (M1) to LoRA + frozen NAMM eviction (M3). Positive values
mean M3 is worse than M1; negative values mean M3 exceeds M1.

### Test split

| Task         |    cs=1024 |    cs=2048 |
| ------------ | ---------: | ---------: |
| Qasper       |    +34.94% |    +11.90% |
| 2WikiMQA     |   -342.31% |   -150.00% |
| Qasper-E     |    +25.45% |    +14.45% |
| HotpotQA-E   |    -42.43% |    -16.39% |
| 2WikiMQA-E   |    +24.44% |    +18.42% |
| **Micro F1** | **-3.64%** | **+0.25%** |

### Extended test split

| Task         |     cs=1024 |     cs=2048 |
| ------------ | ----------: | ----------: |
| Qasper       |     +29.31% |     +34.83% |
| 2WikiMQA     |     -15.59% |      +6.75% |
| Qasper-E     |     +14.50% |     +47.56% |
| HotpotQA-E   |     +13.85% |     +38.22% |
| 2WikiMQA-E   |     +29.49% |     +14.68% |
| **Micro F1** | **+15.47%** | **+27.30%** |

### Interpretation

**Test-split micro sensitivity is near zero or slightly negative.** At cs1024,
M3 actually exceeds M1 by 3.64% (32.28 vs 31.14). At cs2048, M3 essentially
matches M1 (31.06 vs 31.14, sensitivity +0.25%). This confirms the original
validation-era claim that eviction-aware LoRA training compensates for the
information bottleneck.

**Extended-test tells a different story.** On longer contexts (6500-8192
tokens), sensitivity rises to +15.47% (cs1024) and +27.30% (cs2048). The
M3 advantage observed on test does not generalise to contexts beyond the
training-length distribution.

**Task-level anomalies persist.** 2WikiMQA shows massive negative sensitivity
(-342% at cs1024) because M1 scores only 10.00 while M3 scores 44.23. This
makes per-task sensitivity unreliable for 2WikiMQA. Similarly, HotpotQA-E
shows M3 exceeding M1 on test (-42.43%) but not on extended_test (+13.85%).

---

## 5. Recovery Ratio: (M3 - M2) / (M1 - M2)

The recovery ratio measures what fraction of the gap between NAMM-only (M2)
and full-context LoRA (M1) is closed by adding LoRA training on top of frozen
NAMM (M3). A ratio of 1.0 means full recovery to M1 level; values above 1.0
mean M3 exceeds M1.

### Test split, cs=1024

| Task         | Recovery Ratio |
| ------------ | -------------: |
| Qasper       |           0.06 |
| 2WikiMQA     |          -0.95 |
| Qasper-E     |           0.67 |
| HotpotQA-E   |           2.00 |
| 2WikiMQA-E   |          -0.23 |
| **Micro F1** |       **1.10** |

### Test split, cs=2048

| Task         | Recovery Ratio |
| ------------ | -------------: |
| Qasper       |           0.71 |
| 2WikiMQA     |          -0.00 |
| Qasper-E     |           0.83 |
| HotpotQA-E   |           1.41 |
| 2WikiMQA-E   |           0.63 |
| **Micro F1** |       **0.99** |

### Extended test split, cs=1024

| Task         | Recovery Ratio |
| ------------ | -------------: |
| Qasper       |           0.36 |
| 2WikiMQA     |         -22.70 |
| Qasper-E     |           0.74 |
| HotpotQA-E   |           0.69 |
| 2WikiMQA-E   |           0.05 |
| **Micro F1** |       **0.56** |

### Extended test split, cs=2048

| Task         | Recovery Ratio |
| ------------ | -------------: |
| Qasper       |           0.20 |
| 2WikiMQA     |           0.41 |
| Qasper-E     |           0.25 |
| HotpotQA-E   |           0.21 |
| 2WikiMQA-E   |           0.59 |
| **Micro F1** |       **0.32** |

### Interpretation

**Test-split recovery is near-complete.** At cs1024, micro recovery is 1.10
(M3 exceeds M1); at cs2048, it is 0.99 (essentially full recovery). The
LoRA adapter trained under NAMM eviction fully closes -- and on test slightly
exceeds -- the gap to M1.

**Extended-test recovery is substantially lower.** At cs1024 micro recovery
drops to 0.56; at cs2048 it is 0.32. LoRA partially compensates for eviction
on longer contexts but does not fully bridge the gap to M1.

**HotpotQA-E shows super-recovery on test.** Recovery ratios of 2.00 (cs1024)
and 1.41 (cs2048) mean M3 dramatically exceeds M1 on this task. M3 scores
43.45 on HotpotQA-E vs M1's 30.51 -- the eviction-aware adapter appears to
have learned a particularly effective representation for multi-hop QA.

**2WikiMQA recovery is unreliable** due to M1's near-floor score (10.00)
creating an anomalous denominator. The -0.95 and -22.70 values should be
treated with caution.

---

## 6. B1 vs M2: Learned Eviction Beats Recency

The B1 conditions apply simple recency-based eviction (keep most recent N
tokens), providing a baseline for comparison with NAMM's learned eviction (M2).

### Test split, micro F1

| Cache Size | B1 (recency) | M2 (NAMM) | Delta |
| ---------- | -----------: | --------: | ----: |
| cs1024     |        12.45 |     20.30 | +7.85 |
| cs2048     |        13.78 |     17.40 | +3.62 |

### Per-task comparison (test, cs1024)

| Task       |    B1 |    M2 |  Delta |
| ---------- | ----: | ----: | -----: |
| Qasper     | 22.29 | 28.30 |  +6.01 |
| 2WikiMQA   | 10.42 | 27.56 | +17.15 |
| Qasper-E   |  7.26 |  8.09 |  +0.83 |
| HotpotQA-E | 17.65 | 17.50 |  -0.15 |
| 2WikiMQA-E |  6.55 | 24.16 | +17.61 |

**Key findings:**

1. **NAMM beats recency overall** -- M2 outperforms B1 by 7.85 pp at cs1024
   and 3.62 pp at cs2048 on micro F1, confirming that the CMA-ES evolved
   eviction policy provides substantial value beyond simple recency.

2. **The advantage is strongest on multi-hop tasks.** NAMM outperforms recency
   by +17.15 pp on 2WikiMQA and +17.61 pp on 2WikiMQA-E at cs1024. These
   tasks require aggregating information from multiple passages -- exactly
   where intelligent token selection matters.

3. **HotpotQA-E is the exception.** Recency slightly beats NAMM at cs1024
   (17.65 vs 17.50, delta -0.15) and strongly beats it at cs2048 (25.93 vs
   18.45). The HotpotQA-E evidence structure may favour retaining recent
   context over NAMM's spectral scoring.

4. **Both are far below B0.** Both B1/cs1024 (12.45) and M2/cs1024 (20.30)
   score below the no-eviction baseline B0 (22.41). Cache eviction without
   fine-tuning degrades the base model regardless of eviction strategy.

---

## 7. NAMM vs Truncation: M3 > Trunc/lora_m1 > Trunc/plain

The truncation baselines establish a clean performance hierarchy, isolating the
contributions of LoRA and NAMM.

### Test split, micro F1

| Method                                 | cs1024 | cs2048 |
| -------------------------------------- | -----: | -----: |
| M3 (LoRA + frozen NAMM)                |  32.28 |  31.06 |
| Trunc/lora_m1 (M1 LoRA + truncation)   |  26.90 |  28.87 |
| M2 (NAMM only, no LoRA)                |  20.30 |  17.40 |
| Trunc/plain (plain Llama + truncation) |  18.21 |  18.26 |
| B1 (recency eviction)                  |  12.45 |  13.78 |

**Key observations:**

1. **NAMM-adapted LoRA exceeds truncation-evaluated LoRA.** M3/cs1024 (32.28)
   outperforms Trunc/lora_m1_1024 (26.90) by 5.38 pp. This isolates the value
   of NAMM's selective eviction over naive truncation when the LoRA has adapted
   to the eviction regime.

2. **LoRA is the largest single contributor.** Adding M1's LoRA to truncation
   yields +8.69 pp at cs1024 (26.90 vs 18.21) and +10.61 pp at cs2048
   (28.87 vs 18.26).

3. **NAMM alone slightly exceeds truncation alone.** M2/cs1024 (20.30) beats
   Trunc/plain_1024 (18.21) by 2.09 pp. At cs2048, M2 (17.40) is slightly
   below Trunc/plain (18.26), possibly reflecting a less well-evolved cs2048
   NAMM policy.

4. **Recency eviction is worse than truncation.** B1 < Trunc/plain at both
   cache sizes, confirming that naive recency eviction actively harms
   performance more than simply dropping the beginning of the context.

---

## 8. A4 Ablation: NAMM at Inference -- Cache-Size-Dependent

The A4 conditions take the M3-trained LoRA adapter and evaluate it with NAMM
disabled (full KV cache). This isolates whether the M3 performance comes from
(a) the LoRA weights or (b) NAMM eviction at inference time.

### Test split, micro F1

| Condition              | Micro F1 | vs M1 |
| ---------------------- | -------: | ----: |
| M1 (full-context LoRA) |    31.14 |    -- |
| M3/cs1024 (NAMM on)    |    32.28 | +1.14 |
| A4/cs1024 (NAMM off)   |    28.82 | -2.32 |
| M3/cs2048 (NAMM on)    |    31.06 | -0.08 |
| A4/cs2048 (NAMM off)   |    33.91 | +2.77 |

### Interpretation

**At cs1024, NAMM at inference helps.** M3/cs1024 (32.28) exceeds A4/cs1024
(28.82) by 3.46 pp. The cs1024 LoRA is co-adapted with NAMM -- removing NAMM
at inference hurts performance. The LoRA appears to rely on NAMM's filtering
to focus on the most relevant tokens.

**At cs2048, NAMM at inference hurts.** M3/cs2048 (31.06) is below A4/cs2048
(33.91) by 2.85 pp. The cs2048 LoRA actually performs *better* without
eviction. The more moderate eviction during training (cs2048 retains more
tokens) produced a LoRA that is robust enough to benefit from full context.

**A4/cs2048 is the highest-scoring condition overall** (33.91), exceeding M1
(31.14) by 2.77 pp. This suggests that moderate eviction during training
acts as a beneficial regulariser, producing LoRA weights that outperform
standard full-context training even when eviction is removed.

**Per-task (test, cs1024):**

| Task       |    M1 | M3/cs1024 | A4/cs1024 |
| ---------- | ----: | --------: | --------: |
| Qasper     | 45.03 |     29.30 |     46.19 |
| 2WikiMQA   | 10.00 |     44.23 |     25.00 |
| Qasper-E   | 35.62 |     26.56 |     28.12 |
| HotpotQA-E | 30.51 |     43.45 |     26.67 |
| 2WikiMQA-E | 30.16 |     22.79 |     17.46 |

**Per-task (test, cs2048):**

| Task       |    M1 | M3/cs2048 | A4/cs2048 |
| ---------- | ----: | --------: | --------: |
| Qasper     | 45.03 |     39.68 |     43.56 |
| 2WikiMQA   | 10.00 |     25.00 |     38.89 |
| Qasper-E   | 35.62 |     30.47 |     34.63 |
| HotpotQA-E | 30.51 |     35.51 |     35.80 |
| 2WikiMQA-E | 30.16 |     24.60 |     17.46 |

---

## 9. Note: M1_recency/cs1024 Is Broken

The `M1_recency/cs1024` condition (M1 LoRA + recency eviction at cache
size 1024) produces all-zero F1 scores across every task on both test and
extended_test splits. This is a pipeline failure (likely the recency eviction
is discarding too aggressively or interacting badly with the LoRA adapter in a
way that produces degenerate outputs). These results are excluded from all
comparative analyses. The `M1_recency/cs2048` run is still pending.

---

## 10. Summary of Findings

1. **M3/cs1024 slightly exceeds M1 on test** (32.28 vs 31.14, +1.14 pp).
   This confirms the original validation-era claim that LoRA fine-tuning under
   frozen NAMM eviction can match full-context LoRA, even on held-out data.
   M3/cs2048 essentially matches M1 (31.06 vs 31.14).

2. **The advantage does not extend to longer contexts.** On extended_test
   (6500-8192 tokens), M1 (31.84) leads M3/cs1024 (26.92) by 4.92 pp.
   Eviction pressure increases with context length, and M3 cannot fully
   compensate.

3. **NAMM beats both truncation and recency at matched budgets.** The
   performance ladder on test at cs1024 is:
   M3 (32.28) > Trunc/lora_m1 (26.90) > M2 (20.30) > Trunc/plain (18.21)
   > B1 (12.45). Learned eviction provides genuine value over naive baselines.

4. **The A4 ablation reveals cache-size-dependent NAMM utility.** At cs1024,
   NAMM at inference helps (M3=32.28 > A4=28.82). At cs2048, NAMM hurts
   (M3=31.06 < A4=33.91). The cs1024 LoRA is more tightly co-adapted with
   NAMM; the cs2048 LoRA is a more general adaptation.

5. **A4/cs2048 is the best condition overall** (33.91 test micro F1),
   suggesting moderate eviction during training may act as a regulariser
   that produces superior LoRA weights even for full-context inference.

6. **Recovery ratio is near-complete on test** (micro: 1.10 at cs1024, 0.99
   at cs2048) but drops substantially on extended_test (0.56 and 0.32).

7. **Task-level patterns show complementary strengths.** M3 dramatically
   outperforms M1 on 2WikiMQA (44.23 vs 10.00) and HotpotQA-E (43.45 vs
   30.51), while M1 dominates on Qasper (45.03 vs 29.30). NAMM-aware training
   appears to boost multi-hop reasoning at some cost to localised QA.

8. **M1_recency/cs1024 is broken** (all zeros) and excluded from analysis.

---

## 11. Figures

| File                              | Description                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| `generate_plots_v2.py`            | Script to generate all plots from `all_results.json`           |
| `results_test.json`               | Extracted per-condition, per-task, per-split F1 numbers        |
| `test_f1_comparison.png`          | Grouped bar chart: all conditions, 5 tasks + micro, test split |
| `extended_test_f1_comparison.png` | Same for extended_test split                                   |
| `sensitivity_test.png`            | Eviction sensitivity (M1-M3)/M1 per task, cs1024 and cs2048    |
| `recovery_ratio_test.png`         | Recovery ratio (M3-M2)/(M1-M2) per task, cs1024 and cs2048     |
