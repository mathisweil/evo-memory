# Analysis 1 -- Per-Task Eviction Sensitivity

> All values are **validation F1** (not test).
> Naming follows M0--M3 convention throughout.
> M3 checkpoint is step 260 (best at step 260 of 425 before kill); final metrics may shift.

## 1. Per-Task Best Validation F1

| Task         |    B0 |    M1 |    M2 |    M3 |
|:-------------|------:|------:|------:|------:|
| qasper       | 14.69 | 30.67 |  8.62 | 21.86 |
| 2wikimqa     | 15.83 | 50.83 | 20.00 | 63.06 |
| qasper_e     | 12.57 | 32.67 |  8.81 | 25.76 |
| hotpotqa_e   | 40.00 | 44.00 | 14.00 | 74.00 |
| 2wikimqa_e   | 35.68 | 69.23 | 23.08 | 75.64 |
| **Mean**     | **22.59** | **45.48** | **14.90** | **52.06** |

Best step / best iter:

- M1: step 0 (LoRA alone, no eviction)
- M2: iter 170 (WandB `z5bo4n8k`)
- M3: step 260 (run killed at step 425) (WandB `h0bzg6on`)

## 2. M3 Gain Over M1: (M3 - M1) / M1

This measures the relative change in performance when training with NAMM
eviction active (M3) vs full context (M1). Positive = M3 is better.

| Task         | M3 Gain (%)     |
|:-------------|----------------:|
| qasper       |          -28.71 |
| 2wikimqa     |          +24.04 |
| qasper_e     |          -21.14 |
| hotpotqa_e   |          +68.18 |
| 2wikimqa_e   |           +9.26 |
| **Mean**     |      **+14.48** |

On average, M3 surpasses M1 by 14.48% relative. Eviction-aware training does
not merely recover from eviction damage -- it improves beyond the LoRA-only
baseline. Three of five tasks (2WikiMQA, HotpotQA-E, 2WikiMQA-E) show gains,
with HotpotQA-E showing a particularly large +68% improvement. Qasper tasks
are the exception, losing 21-29% relative to M1.

## 3. Recovery Ratio: (M3 - M2) / (M1 - M2)

This measures how much of the gap between M2 (NAMM-only, no LoRA) and M1
(LoRA-only, no eviction) is closed by M3 (joint LoRA+NAMM). Values above 100%
mean M3 exceeded M1.

| Task         | Recovery (%) |
|:-------------|-------------:|
| qasper       |        60.06 |
| 2wikimqa     |       139.69 |
| qasper_e     |        71.05 |
| hotpotqa_e   |       200.00 |
| 2wikimqa_e   |       113.89 |
| **Mean**     |   **121.53** |

M3 fully recovers the M1-M2 gap on average (121.53%), meaning it overshoots M1.
hotpotqa_e achieves 200% recovery -- M3 gains twice the M1-M2 gap at that task.

## 4. Key Findings

1. **M3 (52.06) exceeds M1 (45.48) by +6.58 points.** Joint LoRA+NAMM training
   does not just recover from eviction -- it produces a model that outperforms the
   LoRA-only baseline. This is a 14.5% relative gain over M1.

2. **M2 (14.90) shows the cost of eviction without adaptation.** NAMM-only eviction
   (no LoRA adaptation) drops performance well below even the B0 baseline (22.59).
   The model cannot compensate for token removal without LoRA fine-tuning. This
   confirms that joint training (M3) is essential -- eviction alone destroys too
   much information.

3. **Task-level variance is large.** hotpotqa_e shows the most dramatic M3 gain
   (74.00, up from M1's 44.00), while qasper shows the largest sensitivity to
   eviction (M3 21.86 vs M1 30.67). This suggests some tasks are more sensitive
   to the eviction strategy than others.

4. **Recovery ratio exceeds 100% on three of five tasks.** 2wikimqa (139.69%),
   hotpotqa_e (200.00%), and 2wikimqa_e (113.89%) all show M3 exceeding M1,
   indicating that the NAMM eviction policy can act as a beneficial regulariser
   when the model is allowed to adapt jointly.

## 5. Comparison with Pre-Correction (Buggy) Runs

Early M2/M3 runs used a broken attention mask that allowed partial attention to
evicted tokens. The bug inflated M2 scores (27.90 mean F1 vs 14.90 after
correction) because the NAMM policy learned to exploit information leaking
through "evicted" positions. M3 buggy achieved 45.59 mean F1 -- substantially
lower than the corrected M3 (52.06). Once joint adaptation is allowed, the
corrected attention mask yields a +6.47 point improvement, confirming that the
buggy M2 advantage was a misleading signal caused by information leakage, not
genuine eviction quality.

| Condition    | M2 Val F1 | M3 Val F1 |
|:-------------|----------:|----------:|
| Buggy        |     27.90 |     45.59 |
| Corrected    |     14.90 |     52.06 |

## 6. Caveats

- M3 training was killed at step 425; best val F1 at step 260. The run
  did not complete and final metrics may differ.
- All numbers are validation F1, not test F1. Final conclusions require test-set
  evaluation.

## Plots

| Plot                                                     | Description                                  |
| -------------------------------------------------------- | -------------------------------------------- |
| [`best_val_f1_comparison.png`](plots/best_val_f1_comparison.png) | Per-task best val F1 grouped bar (B0--M3)  |
| [`sensitivity_bar.png`](plots/sensitivity_bar.png)             | M3 gain over M1 per task                     |
| [`recovery_ratio.png`](plots/recovery_ratio.png)               | Recovery ratio per task (M3/M1)              |
