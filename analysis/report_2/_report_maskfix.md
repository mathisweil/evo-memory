# Analysis 2 (Maskfix) -- Adaptation Rate and Learning Efficiency

> **Status**: M3 maskfix is still running (~43% complete, step 298/~608).
> All values are **validation F1** (not test).
> Naming follows M0--M3 convention throughout.

## 1. Baseline Performance Under Eviction

The baseline is the validation F1 at step 0 of M3 training -- i.e., the M1 LoRA
checkpoint evaluated with NAMM eviction active but before any joint fine-tuning.

| Condition   | Baseline Val F1 |
|:------------|----------------:|
| M3 maskfix  |           23.70 |
| M3 buggy    |           19.96 |

The maskfix baseline is **3.74 points higher** than the buggy baseline. This means
that even at step 0 (before any joint training), the correctly-masked model retains
more information under eviction. In the buggy regime, the model's internal
representations were trained (during M1) with a broken attention mask, so when
eviction is applied in M3, the model is less robust to the actual token removal.

With correct masking, the model's representations are never allowed to depend on
tokens that should be invisible, so the zero-shot-under-eviction performance is
inherently better.

## 2. Convergence Speed: Steps to Threshold

Threshold values are computed as percentages of the way from baseline to best val F1.
Data available for M3 maskfix only.

| Threshold | Target Val F1 | Steps to Reach |
|:----------|:--------------|---------------:|
| 50%       | 37.88         |             28 |
| 75%       | 44.97         |            134 |
| 90%       | 49.28         |            222 |

- M3 maskfix reaches 50% of its improvement range in just 28 steps -- rapid early
  gains.
- The 75% threshold takes 134 steps, showing continued steady improvement.
- 90% is reached at step 222, after which the remaining gains are marginal.

For M3 buggy, threshold steps were not pre-extracted, but given its best step (340)
and total steps (608), it reaches its peak later in training.

## 3. Best Validation F1 and Convergence Point

| Metric         | M3 buggy | M3 maskfix |
|:---------------|:---------|:-----------|
| Best val F1    | 45.59    | 52.06      |
| Best step      | 340      | 260        |
| Total steps    | 608      | 298*       |
| Baseline       | 19.96    | 23.70      |
| Gain over base | +25.63   | +28.36     |

*M3 maskfix is still running (298 of ~608 steps completed).

Key observations:

- **Maskfix converges faster**: best val F1 at step 260 vs step 340 for buggy.
  That is 80 fewer steps to reach the peak, a 23.5% reduction in steps-to-best.
- **Maskfix converges higher**: 52.06 vs 45.59, a +6.47 absolute / +14.2%
  relative improvement.
- **Maskfix has a larger absolute gain**: +28.36 points from baseline vs +25.63
  for buggy, despite starting from a higher baseline.
- **Maskfix may improve further**: with ~57% of training remaining, there is
  headroom for additional gains (or potential overfitting).

## 4. Training-Validation Gap

The train-val gap measures how much training F1 exceeds validation F1 at the best
checkpoint, indicating potential overfitting.

| Metric     | M3 buggy         | M3 maskfix        |
|:-----------|:-----------------|:------------------|
| Mean gap   | -2.24            | -8.56             |
| Gap std    | 4.24             | 3.55              |

Negative gap means train F1 < val F1, which is unusual and suggests that the
training distribution is harder than the validation distribution for these tasks.

- M3 maskfix has a **larger negative gap** (-8.56 vs -2.24), meaning validation
  F1 exceeds training F1 by a wider margin. This could indicate:
  - The maskfix model generalises better from training to validation examples.
  - The training examples (sampled from LongBench) are systematically harder than
    the validation split.
  - The maskfix model's higher capacity under correct attention means it benefits
    more from the validation distribution.
- M3 maskfix has **lower gap variance** (std 3.55 vs 4.24), suggesting more
  consistent behaviour across tasks.
- Neither condition shows signs of overfitting at their respective best steps.

## 5. Summary: Learning Efficiency Comparison

| Property                   | M3 buggy | M3 maskfix | Advantage    |
|:---------------------------|:---------|:-----------|:-------------|
| Baseline val F1            | 19.96    | 23.70      | maskfix +3.74 |
| Best val F1                | 45.59    | 52.06      | maskfix +6.47 |
| Absolute gain              | 25.63    | 28.36      | maskfix +2.73 |
| Steps to best              | 340      | 260        | maskfix -80   |
| Train-val gap              | -2.24    | -8.56      | --           |
| Gap std                    | 4.24     | 3.55       | maskfix lower |

The maskfix correction improves learning efficiency on every measured axis: higher
starting point, faster convergence, and higher final performance. The correct
attention mask ensures that the LoRA adapter and NAMM policy co-adapt in a
consistent environment where eviction genuinely removes information, producing
more robust learned representations.

## 6. Caveats

- M3 maskfix has completed only ~43% of training. The best step and final metrics
  may shift as training continues.
- Steps-to-threshold data is only available for M3 maskfix. A direct threshold
  comparison with M3 buggy is not possible from the pre-extracted data.
- All numbers are validation F1, not test F1.
- The train-val gap is computed at the best validation checkpoint, not necessarily
  at the same training step for both conditions.
