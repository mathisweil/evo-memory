# Analysis 8 (Maskfix) -- Probing for Residual Knowledge of Evicted Content

> **Status**: M3 maskfix is still running (~43% complete, step 260/~608).
> Naming follows M0--M3 convention throughout.
> Maskfix checkpoints: M2 from `z5bo4n8k`, M3 from `h0bzg6on` step 260.

> **TL;DR:** With maskfix, M1 probe accuracy rises to 0.599 (from buggy 0.557)
> and M3 to 0.513 (from buggy 0.484). However, the random baseline is 0.600
> (majority class), meaning M1 is **at chance** and M3 is **below chance**.
> The probe task appears poorly calibrated for the maskfix regime: the
> maskfix NAMM makes different eviction decisions that shift the class
> balance, rendering the binary probe uninformative. The M1--M3 gap is
> smaller than buggy (0.085 vs 0.073) and the maximum gap shifts from
> layer 14 to layer 8.

## Methodology

Same setup as the original Report 8: linear probes (logistic regression,
5-fold stratified CV) trained on mean-pooled hidden states per layer to
predict whether answer tokens were evicted. 40 test samples from the
5-task QA subset.

- **M1:** Forward pass with Recency passthrough (no eviction), extract
  mean-pooled hidden states per layer.
- **M3 maskfix:** Forward pass with NAMM active (cache_size=1024,
  `z5bo4n8k` policy), extract mean-pooled hidden states from retained
  tokens only.
- **Binary label:** 1 if any answer tokens overlap with evicted positions,
  0 otherwise.

## Results

### Probe Accuracy Summary

| Metric                      | Buggy M1 | Buggy M3 | Maskfix M1 | Maskfix M3 | Random |
| --------------------------- | -------: | -------: | ---------: | ---------: | -----: |
| Mean probe accuracy         |    0.557 |    0.484 |      0.599 |      0.513 |  0.600 |
| Std probe accuracy          |      --  |      --  |      0.131 |      0.145 |    --  |
| Mean accuracy gap (M1 - M3) |    0.073 |      --  |      0.085 |        --  |    --  |
| Max gap layer               |       14 |      --  |          8 |        --  |    --  |
| Max gap magnitude           |    0.325 |      --  |      0.200 |        --  |    --  |

### Probe Accuracy by Layer (Maskfix)

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
| Layer 7   |           * |           * |              * |  0.600 |
| Layer 8   |           * |           * |      **0.200** |  0.600 |
| Layer 9   |           * |           * |              * |  0.600 |
| Layer 10  |           * |           * |              * |  0.600 |
| Layer 11  |           * |           * |              * |  0.600 |
| Layer 12  |           * |           * |              * |  0.600 |
| Layer 13  |           * |           * |              * |  0.600 |
| Layer 14  |           * |           * |              * |  0.600 |
| Layer 15  |           * |           * |              * |  0.600 |

`*` See `probe_accuracy_maskfix.png` for exact per-layer values.

### Key Observations

1. **M1 probe accuracy matches the random baseline.** With maskfix, the
   mean M1 probe accuracy is 0.599, which is indistinguishable from the
   majority-class baseline of 0.600. This means a linear probe on M1
   hidden states cannot distinguish samples where answer tokens would
   be evicted from those where they would not -- the probe has no
   discriminative power under the maskfix eviction regime.

2. **M3 probe accuracy is below chance.** At 0.513, M3 is below the
   0.600 baseline, meaning the probe is actively misclassifying samples.
   Under the maskfix NAMM, representations of retained tokens in M3 are
   *less* informative about answer-token eviction than random guessing.

3. **The M1--M3 gap is real but uninterpretable.** The 0.085 gap (M1
   0.599, M3 0.513) exists, but since both conditions are at or below
   chance, the gap reflects noise in below-baseline predictions rather
   than meaningful information loss.

4. **Max gap shifts from layer 14 to layer 8.** Under buggy eviction,
   the largest M1--M3 divergence was at layer 14 (gap = 0.325). With
   maskfix, it shifts to layer 8 (gap = 0.200). This aligns with the
   maskfix NAMM's more aggressive eviction at earlier layers (Report 3
   maskfix: retention drops sharply in layers 2--8).

## Buggy vs Maskfix Comparison

| Aspect                  | Buggy        | Maskfix       | Interpretation                        |
| ----------------------- | -----------: | ------------: | ------------------------------------- |
| M1 mean accuracy        |        0.557 |         0.599 | Higher, but now at chance (0.600)     |
| M3 mean accuracy        |        0.484 |         0.513 | Higher, but still below chance        |
| M1 - M3 gap             |        0.073 |         0.085 | Slightly larger                       |
| Max gap layer           |           14 |             8 | Shifts to earlier layers              |
| Max gap magnitude       |        0.325 |         0.200 | Smaller peak gap                      |
| Random baseline         |        0.500 |         0.600 | **Changed** -- class balance shifted  |
| M1 vs random            | Above chance | At chance     | Probe is uninformative under maskfix  |

### Why the Random Baseline Changed

The random baseline shifted from 0.500 (buggy) to 0.600 (maskfix) because
the maskfix NAMM makes **different eviction decisions** than the buggy NAMM.
Under buggy eviction (driven by positional heuristics due to uniform
attention), the class split was roughly 50/50 (answer tokens evicted vs
not). Under maskfix eviction (driven by actual attention-informed scoring),
the class balance shifts to approximately 60/40 -- the maskfix NAMM is
**more likely to retain answer tokens** (since it scores tokens by
attention importance, and answer tokens tend to be attended to). This
creates a majority class at 60%, making the binary probe task harder.

### Why M1 Drops to Chance Under Maskfix

The M1 condition uses no eviction (full context). The probe tests whether
M1 hidden states can predict which tokens *would have been* evicted by
NAMM. Under buggy NAMM, the eviction pattern was driven by position
(systematic, predictable from representations), giving M1 above-chance
accuracy (0.557). Under maskfix NAMM, the eviction pattern is driven by
attention-based scoring (complex, sample-specific), which is harder to
predict from mean-pooled hidden states alone. The probe loses its
discriminative signal.

## Interpretation

### The Probe Task Is Not Well-Calibrated for Maskfix

The central finding is that the binary probing approach used in the
original Report 8 does not transfer cleanly to the maskfix regime:

1. **Different eviction patterns produce different class balances.** The
   50/50 split that made the buggy probe task feasible is gone.

2. **Attention-informed eviction is harder to predict.** The buggy NAMM's
   positional-heuristic eviction was systematic and predictable. The
   maskfix NAMM's attention-based eviction varies per sample and per layer,
   making it harder for a simple linear probe to detect.

3. **The information-loss narrative still holds conceptually** -- M3
   representations likely do lose some evicted-token information -- but
   this probe design cannot measure it reliably when both M1 and M3 are
   at or below chance.

### What Would Fix the Probe

A better-calibrated probe for maskfix would:
- Use a **regression target** (fraction of attention mass evicted) rather
  than a binary label
- Use **per-token probes** rather than mean-pooled representations
- Control for class balance by resampling to 50/50

### Connection to Other Maskfix Reports

| Report              | Finding                                     | Implication for Report 8                         |
| ------------------- | ------------------------------------------- | ------------------------------------------------ |
| 3 (Retention)       | Maskfix has more aggressive, varied eviction | Different eviction pattern shifts class balance   |
| 6 (Alignment)       | Maskfix rho = +0.14 (attention-aligned)      | Attention-aligned eviction retains answer tokens |
| 7 (Representations) | Maskfix CKA similarity is higher             | Representations are more similar, harder to probe |

## Figures

| File                              | Description                                                |
| --------------------------------- | ---------------------------------------------------------- |
| `probe_accuracy_maskfix.png`      | Per-layer probe accuracy for M1 vs M3 vs random baseline   |
| `entity_survival_maskfix.png`     | Retention fractions and answer token survival estimates     |
| `layer_wise_information_maskfix.png` | Per-layer accuracy difference (M1 - M3)                 |
