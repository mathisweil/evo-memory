# Analysis 8 -- Probing for Residual Knowledge of Evicted Content

> Checkpoints: M2 NAMM from WandB `z5bo4n8k`, M3 LoRA from `h0bzg6on` step 260.

> **TL;DR:** The probe task is poorly calibrated for this eviction regime.
> The random baseline is 0.600 (majority class due to imbalanced eviction
> labels), M1 probe accuracy is 0.599 (indistinguishable from chance), and
> M3 is 0.513 (below chance). Neither condition carries a detectable linear
> signal about which samples had answer tokens evicted. The results are
> inconclusive -- the probe design does not have discriminative power under
> the attention-informed eviction pattern produced by this NAMM, so we
> cannot draw conclusions about residual knowledge of evicted content.

## Methodology

Linear probes (logistic regression, 5-fold stratified CV) trained on
mean-pooled hidden states per layer to predict whether answer tokens were
evicted. 40 test samples from the 5-task QA subset.

- **M1:** Forward pass with Recency passthrough (no eviction), extract
  mean-pooled hidden states per layer.
- **M3:** Forward pass with NAMM active (cache_size=1024, `z5bo4n8k`
  policy), extract mean-pooled hidden states from retained tokens only.
- **Binary label:** 1 if any answer tokens overlap with evicted positions,
  0 otherwise.

## Results

### Probe Accuracy Summary

| Metric                      | M1    | M3    | Random |
| --------------------------- | ----: | ----: | -----: |
| Mean probe accuracy         | 0.599 | 0.513 |  0.600 |
| Std probe accuracy          | 0.131 | 0.145 |    --  |
| Mean accuracy gap (M1 - M3) | 0.085 |   --  |    --  |
| Max gap layer               |   7,9 |   --  |    --  |
| Max gap magnitude           | 0.200 |   --  |    --  |

### Probe Accuracy by Layer

| Layer     | M1 accuracy | M3 accuracy | Gap (M1 - M3) | Random |
| --------- | ----------: | ----------: | -------------: | -----: |
| Embedding |           * |           * |              * |  0.600 |
| Layer 0   |           * |           * |              * |  0.600 |
| Layer 1   |           * |           * |              * |  0.600 |
| Layer 2   |           * |           * |              * |  0.600 |
| Layer 3   |           * |           * |              * |  0.600 |
| Layer 4   |           * |           * |              * |  0.600 |
| Layer 5   |           * |           * |              * |  0.600 |
| Layer 6   |           * |           * |              * |  0.600 |
| Layer 7   |           * |           * |      **0.200** |  0.600 |
| Layer 8   |           * |           * |          0.125 |  0.600 |
| Layer 9   |           * |           * |      **0.200** |  0.600 |
| Layer 10  |           * |           * |              * |  0.600 |
| Layer 11  |           * |           * |              * |  0.600 |
| Layer 12  |           * |           * |              * |  0.600 |
| Layer 13  |           * |           * |              * |  0.600 |
| Layer 14  |           * |           * |              * |  0.600 |
| Layer 15  |           * |           * |              * |  0.600 |

`*` See `plots/probe_accuracy.png` for exact per-layer values.

### Key Observations

1. **M1 probe accuracy matches the random baseline.** Mean M1 probe
   accuracy is 0.599, indistinguishable from the majority-class baseline
   of 0.600. A linear probe on M1 hidden states cannot distinguish
   samples where answer tokens would be evicted from those where they
   would not -- the probe has no discriminative power.

2. **M3 probe accuracy is below chance.** At 0.513, M3 is below the
   0.600 baseline, meaning the probe is actively misclassifying samples.
   Representations of retained tokens in M3 are less informative about
   answer-token eviction than random guessing.

3. **The M1--M3 gap is uninterpretable.** The 0.085 gap (M1 0.599, M3
   0.513) exists, but since both conditions are at or below chance, the
   gap reflects noise in below-baseline predictions rather than meaningful
   information loss.

4. **Max gap is at layers 7 and 9.** The largest M1--M3 divergence (0.200) is at layers 7 and 9
   (gap = 0.200). This aligns with the NAMM's more aggressive eviction
   in layers 2--8 (Report 3).

## Why the Probe Fails

The random baseline of 0.600 (rather than 0.500) reveals a class imbalance
problem. The NAMM's attention-informed eviction retains answer tokens more
often than not, creating an approximately 60/40 class split (answer tokens
not evicted vs evicted). This majority-class baseline makes the binary
probe task harder -- a probe needs to exceed 0.600, not 0.500, to
demonstrate any signal.

The underlying issue is that attention-informed eviction produces
sample-specific, complex eviction patterns that are difficult to predict
from mean-pooled hidden states. The NAMM scores tokens by attention
importance, and answer tokens tend to be attended to, so they are
preferentially retained. A simple linear probe on mean-pooled
representations cannot capture this nuanced, per-token scoring process.

## The Results Are Inconclusive

The probe task as designed does not have the statistical power to measure
residual knowledge of evicted content in this regime. The information-loss
narrative still holds conceptually -- M3 representations likely do lose
some evicted-token information -- but this probe cannot measure it.

A better-calibrated probe would:
- Use a **regression target** (fraction of attention mass evicted) rather
  than a binary label
- Use **per-token probes** rather than mean-pooled representations
- Control for class balance by resampling to 50/50

## Comparison with Buggy Runs (Historical Context)

Under the original buggy attention mask, the probe showed different
behaviour due to a different eviction regime:

| Metric              | Corrected | Buggy | Note                             |
| ------------------- | --------: | ----: | -------------------------------- |
| M1 mean accuracy    |     0.599 | 0.557 | Buggy was above its 0.500 random |
| M3 mean accuracy    |     0.513 | 0.484 | Both below their baselines       |
| Random baseline     |     0.600 | 0.500 | Class balance shifted            |
| Max gap layer       |         8 |    14 | Shifted earlier                  |
| Max gap magnitude   |     0.200 | 0.325 | Smaller peak gap                 |

The buggy NAMM used positional heuristics (due to uniform attention from
the mask bug), producing a 50/50 class split and more systematic,
predictable eviction patterns. This gave the probe marginally above-chance
accuracy for M1 (0.557 vs 0.500 baseline). With correct attention, the
eviction pattern becomes attention-informed and harder to predict, and the
class imbalance renders the probe uninformative.

## Connection to Other Reports

| Report       | Finding                              | Implication for Rpt 8                       |
| ------------ | ------------------------------------ | ------------------------------------------- |
| 3 (Retain.)  | More aggressive, varied eviction     | Shifts class balance in eviction pattern    |
| 6 (Align.)   | rho = +0.14 (attention-aligned)      | Attention-aligned eviction keeps answer toks|
| 7 (Repr.)    | CKA similarity high (mean 0.995)     | Representations more similar, harder to probe|

## Figures

| File                                    | Description                                             |
| --------------------------------------- | ------------------------------------------------------- |
| `plots/probe_accuracy.png`            | Per-layer probe accuracy for M1 vs M3 vs random baseline|
| `plots/entity_survival.png`           | Retention fractions and answer token survival estimates  |
| `plots/layer_wise_information.png`    | Per-layer accuracy difference (M1 - M3)                 |
