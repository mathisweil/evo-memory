# Analysis 1 (Maskfix) -- Per-Task Eviction Sensitivity

> **Status**: M3 maskfix is still running (~43% complete, step 298/~608).
> All values are **validation F1** (not test).
> Naming follows M0--M3 convention throughout.

## 1. Per-Task Best Validation F1

| Task         |    B0 |    M1 | M2 buggy | M2 maskfix | M3 buggy | M3 maskfix |
|:-------------|------:|------:|---------:|-----------:|---------:|-----------:|
| qasper       | 14.69 | 30.67 |    14.18 |       8.62 |    21.60 |      21.86 |
| 2wikimqa     | 15.83 | 50.83 |    29.00 |      20.00 |    51.11 |      63.06 |
| qasper_e     | 12.57 | 32.67 |    17.82 |       8.81 |    39.40 |      25.76 |
| hotpotqa_e   | 40.00 | 44.00 |    39.54 |      14.00 |    59.67 |      74.00 |
| 2wikimqa_e   | 35.68 | 69.23 |    38.97 |      23.08 |    56.15 |      75.64 |
| **Mean**     | **22.59** | **45.48** | **27.90** | **14.90** | **45.59** | **52.06** |

Best step / best iter:

- M1: step 0 (LoRA alone, no eviction)
- M2 buggy: iter 105
- M2 maskfix: iter 170
- M3 buggy: step 340 (of 608)
- M3 maskfix: step 260 (of ~608, still running)

## 2. Eviction Sensitivity: (M1 - M3) / M1

This measures the fraction of M1 performance *lost* when NAMM eviction is applied
during joint training. Negative values mean M3 *exceeded* M1, i.e. joint training
recovered more than what was lost.

| Task         | Buggy (%) | Maskfix (%) |
|:-------------|----------:|------------:|
| qasper       |    +29.56 |      +28.71 |
| 2wikimqa     |     -0.55 |      -24.04 |
| qasper_e     |    -20.60 |      +21.14 |
| hotpotqa_e   |    -35.61 |      -68.18 |
| 2wikimqa_e   |    +18.89 |       -9.26 |
| **Mean**     |  **-0.24** | **-14.48** |

Positive = performance lost; negative = performance gained beyond M1.

On average, M3 buggy essentially matches M1 (sensitivity -0.24%), while M3 maskfix
*surpasses* M1 by 14.48% relative. The maskfix attention correction enables the
joint-training loop (LoRA + NAMM) to not just recover from eviction damage but to
improve beyond the LoRA-only baseline.

## 3. Recovery Ratio: (M3 - M2) / (M1 - M2)

This measures how much of the gap between M2 (NAMM-only, no LoRA) and M1 (LoRA-only,
no eviction) is closed by M3 (joint LoRA+NAMM). Values above 100% mean M3 exceeded M1.

| Task         | Buggy (%) | Maskfix (%) |
|:-------------|----------:|------------:|
| qasper       |     45.01 |       60.06 |
| 2wikimqa     |    101.28 |      139.69 |
| qasper_e     |    145.33 |       71.05 |
| hotpotqa_e   |    451.13 |      200.00 |
| 2wikimqa_e   |     56.78 |      113.89 |
| **Mean**     | **100.61** | **121.53** |

Both buggy and maskfix M3 fully recover the M1-M2 gap on average, but maskfix
achieves a substantially higher recovery ratio (121.53% vs 100.61%), meaning it
overshoots M1 by a wider margin.

## 4. Key Findings

1. **M3 maskfix (52.06) >> M3 buggy (45.59) >> M1 (45.48).** Fixing the attention
   mask during joint training yields a +6.47 point absolute improvement over the
   buggy M3, and +6.58 points over M1. This is a 14.2% relative gain over M3 buggy.

2. **M2 maskfix (14.90) << M2 buggy (27.90).** The maskfix M2 performs substantially
   *worse* than the buggy M2. This is expected: the buggy NAMM was optimised under
   a broken attention regime where the model could partially attend to supposedly
   evicted tokens. The NAMM policy learned to exploit this leak, achieving
   artificially higher scores. With correct masking (maskfix), eviction is real:
   truly evicted tokens are invisible, so a NAMM-only policy (no LoRA adaptation)
   does much worse.

3. **The buggy M2 advantage is a misleading signal.** The 13-point M2 buggy > M2
   maskfix gap does not indicate that the buggy setup is better. It shows the buggy
   attention mask allowed information leakage that flattered the NAMM-only score.
   Once the model is allowed to jointly adapt (M3), the maskfix condition
   dramatically outperforms.

4. **Task-level variance is large.** hotpotqa_e shows the most dramatic maskfix
   gain (+14.33 points in M3, from 59.67 to 74.00), while qasper_e actually
   regresses under maskfix M3 (39.40 to 25.76). This suggests some tasks are more
   sensitive to the eviction strategy than others.

## 5. Caveats

- M3 maskfix has completed only ~43% of training (step 298 of ~608). The final best
  val F1 may change.
- All numbers are validation F1, not test F1. Final conclusions require test-set
  evaluation.
- The buggy M2 was optimised end-to-end in the broken attention regime, so its
  evolved NAMM policy is specifically adapted to exploit the attention leak. The
  two M2 conditions are not directly comparable in the usual sense -- they were
  optimised in different environments.
