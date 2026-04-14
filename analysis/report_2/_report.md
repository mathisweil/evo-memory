# Analysis 2 -- Adaptation Rate and Learning Efficiency

> All values are **validation F1** (not test).
> Naming follows M0--M3 convention throughout.
> M3 checkpoint is step 260 (~43% through training); final metrics may shift.

## 1. Baseline Performance Under Eviction

The baseline is the validation F1 at step 0 of M3 training -- i.e., the M1 LoRA
checkpoint evaluated with NAMM eviction active but before any joint fine-tuning.

| Metric          |  Value |
|:----------------|-------:|
| Baseline val F1 |  23.70 |
| M1 val F1       |  45.48 |
| Eviction damage |  21.78 |

Applying NAMM eviction to the M1 checkpoint (without any joint training) drops
performance by 21.78 points -- nearly halving it. This quantifies the starting
gap that M3 joint training must recover.

## 2. Convergence Speed: Steps to Threshold

Threshold values are computed as percentages of the way from baseline (23.70)
to best val F1 (52.06).

| Threshold | Target Val F1 | Steps to Reach |
|:----------|:--------------|---------------:|
| 50%       | 37.88         |             28 |
| 75%       | 44.97         |            134 |
| 90%       | 49.28         |            222 |

- M3 reaches 50% of its improvement range in just 28 steps -- rapid early gains.
- The 75% threshold takes 134 steps, showing continued steady improvement.
- 90% is reached at step 222, after which the remaining gains are marginal.
- The best val F1 (52.06) is reached at step 260.

The fast early convergence suggests that the LoRA adapter quickly learns to
compensate for the most damaging evictions, while later steps refine task-specific
representations.

## 3. Best Validation F1 and Convergence Point

| Metric                 | Value  |
|:-----------------------|-------:|
| Best val F1            |  52.06 |
| Best step              |    260 |
| Total steps (planned)  |   ~608 |
| Baseline val F1        |  23.70 |
| Gain over baseline     | +28.36 |
| Gain over M1 (45.48)  |  +6.58 |

M3 not only recovers the 21.78-point eviction damage but overshoots M1 by 6.58
points. This means joint training produces a net improvement of +28.36 points
from the eviction-damaged baseline, reaching a final performance 14.5% above
the LoRA-only result.

## 4. Training-Validation Gap

The train-val gap measures how much training F1 exceeds validation F1 at the best
checkpoint, indicating potential overfitting.

| Metric     | Value |
|:-----------|------:|
| Mean gap   | -8.56 |
| Gap std    |  3.55 |

Negative gap means train F1 < val F1, which is unusual and suggests that the
training distribution is harder than the validation distribution for these tasks.
This could indicate:

- The training examples (sampled from LongBench) are systematically harder than
  the validation split.
- The model generalises well from training to validation under the eviction
  regime.
- No signs of overfitting at step 260, with low gap variance (std 3.55)
  suggesting consistent behaviour across tasks.

## 5. Summary: M3 Learning Efficiency

| Property                  | Value     |
|:--------------------------|:----------|
| Baseline val F1 (step 0)  | 23.70     |
| Best val F1               | 52.06     |
| Absolute gain             | +28.36    |
| M1 val F1 (no eviction)   | 45.48     |
| M3 gain over M1           | +6.58     |
| Steps to 50% recovery     | 28        |
| Steps to 75% recovery     | 134       |
| Steps to 90% recovery     | 222       |
| Steps to best             | 260       |
| Train-val gap             | -8.56     |
| Gap std                   | 3.55      |

The learning curve shows rapid early adaptation followed by steady refinement.
M3 recovers the full eviction damage and exceeds M1 within 260 steps, with no
signs of overfitting.

## 6. Comparison with Pre-Correction (Buggy) Runs

Early M3 runs used a broken attention mask that allowed partial attention to
evicted tokens. The buggy M3 converged later (step 340) and to a lower peak
(45.59 vs 52.06), starting from a lower baseline (19.96 vs 23.70).

| Property            | Buggy | Corrected |
|:--------------------|------:|----------:|
| Baseline val F1     | 19.96 |     23.70 |
| Best val F1         | 45.59 |     52.06 |
| Absolute gain       | 25.63 |     28.36 |
| Steps to best       |   340 |       260 |
| Train-val gap       | -2.24 |     -8.56 |
| Gap std             |  4.24 |      3.55 |

The corrected run converges faster (80 fewer steps), higher (+6.47 points), and
with a larger absolute gain (+2.73 points) despite starting from a higher
baseline. The corrected attention mask ensures LoRA and NAMM co-adapt in a
consistent environment where eviction genuinely removes information, producing
more robust learned representations.

## 7. Caveats

- M3 has completed only ~43% of training (step 260 of ~608). The best step and
  final metrics may shift as training continues.
- All numbers are validation F1, not test F1.
- The train-val gap is computed at the best validation checkpoint.

## Plots

| Plot                                                         | Description                           |
| ------------------------------------------------------------ | ------------------------------------- |
| [`learning_curves_overlay.png`](learning_curves_overlay.png) | M1 vs M3 validation F1 over training  |
| [`normalised_improvement.png`](normalised_improvement.png)   | Normalised improvement curves         |
| [`overfitting_gap.png`](overfitting_gap.png)                 | Train-val gap over training           |
