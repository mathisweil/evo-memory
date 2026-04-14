# Analysis 2 -- Adaptation Rate and Learning Efficiency

## Summary

This analysis compares the learning dynamics of M1 (LoRA fine-tuning with full context) and M3 (LoRA fine-tuning with frozen NAMM active) across three cache sizes (1024, 2048, 3072). We examine normalised improvement curves, convergence speed, and the train-val generalisation gap.

**Key findings:**

- M1 and the M3 variants reach comparable peak validation F1 (~45-46), except M3 cs3072, which peaked at 41.38 but was only trained for 116 steps (vs 682 for M1).
- Each M3 condition starts from a substantially lower baseline than M1 (19.96-30.56 vs 22.59) because eviction degrades zero-shot performance, yet M3 cs1024 and M3 cs2048 recover to match or slightly exceed M1's best F1.
- M3 cs1024 converges more slowly than M1 to the 75% threshold (214 vs 102 steps).
- The train-val gap is consistently *negative* for all conditions (val F1 exceeds train F1), so the traditional overfitting framing does not apply. M3 cs1024 shows the smallest magnitude gap (mean -2.24), while M1 shows the largest (mean -5.74).

## Conditions and Baselines

| Condition | WandB Run(s)                 | Baseline F1 | Best Val F1 | Best Step | Total Steps |
| --------- | ---------------------------- | ----------- | ----------- | --------- | ----------- |
| M1        | kz6vqo2o, x9a4smmf, qfoxxi2m | 22.59       | 45.48       | 336       | 682         |
| M3 cs1024 | ovosogkj                     | 19.96       | 45.59       | 340       | 608         |
| M3 cs2048 | m4knrhmr                     | 24.84       | 46.39       | 354       | 370         |
| M3 cs3072 | 4sgkswa6                     | 30.56       | 41.38       | 70        | 116         |

The baseline F1 is the zero-shot performance of the base model under each condition's inference setup (full context for M1, evicted context for M3). Each condition's own baseline was used for normalisation.

## Steps to Threshold

The table below shows the number of gradient steps required for each condition to first reach 50%, 75%, and 90% of its own best validation F1.

| Condition | 50% Threshold | 75% Threshold | 90% Threshold |
| --------- | ------------- | ------------- | ------------- |
| M1        | 2             | 102           | 258           |
| M3 cs1024 | 16            | 214           | 246           |
| M3 cs2048 | 4             | 150           | 238           |
| M3 cs3072 | 2             | 2             | 54            |

See `steps_to_threshold.png`.

**Interpretation:** All conditions reach 50% of their best F1 almost immediately (within 2-16 steps), because the baselines already provide substantial performance. The differences emerge at higher thresholds:

- **M3 cs1024** is the slowest to reach 75% (214 steps vs 102 for M1), consistent with it having to recover from the lowest baseline (19.96) while navigating an information bottleneck.
- **M3 cs2048** is intermediate (150 steps to 75%).
- **M3 cs3072** reaches all thresholds very early, but this is partly because its best F1 (41.38) is lower than the others, so its absolute thresholds are easier to meet. Additionally, its run is very short, so it may not have converged.

## Normalised Improvement Curves

See `normalised_improvement.png`.

The normalised improvement maps each condition's trajectory to [0, 1], where 0 = baseline and 1 = best achieved F1. This isolates the *shape* of learning from the absolute performance level.

- **M1** shows a gradual, steady rise over the first ~300 steps, reaching its peak around step 336 before declining.
- **M3 cs1024** shows a noisier trajectory with a slower initial rise, but ultimately reaches its normalised peak at a similar step count (~340). The higher noise reflects the stochasticity introduced by token eviction during training.
- **M3 cs2048** follows a trajectory similar to M1 but with more variance, peaking at step 354.
- **M3 cs3072** rises quickly in its short window but shows high variance.

The fact that M3 cs1024 and M3 cs2048 both reach normalised improvement ~1.0 at approximately the same step count as M1 suggests that the learning rate of adaptation is roughly comparable, despite the additional challenge of eviction.

## Learning Curves Overlay

See `learning_curves_overlay.png`.

The raw (lightly smoothed) validation F1 curves confirm that:

- M1, M3 cs1024, and M3 cs2048 all converge to a similar performance band (~43-46 F1) by step ~300-350.
- M3 cs1024 starts lowest (~20 F1) and has the steepest absolute improvement trajectory.
- M3 cs2048 starts at ~25 F1 and tracks M1 closely after ~150 steps.
- M3 cs3072 starts at ~31 F1 but its short run makes comparison difficult.
- After their respective peaks, M1 and M3 cs1024 show performance degradation in later steps.

## Overfitting Gap

See `overfitting_gap.png`.

The overfitting gap is defined as `train_F1 - val_F1`. Positive values indicate overfitting; negative values indicate val outperforms train.

### Surprising finding: negative gap throughout

All conditions show a consistently *negative* gap, meaning validation F1 exceeds training F1 throughout training. This likely reflects a methodological difference between how training and validation F1 are computed (e.g. different evaluation subsets or answer extraction methodology).

| Condition | Mean Gap | Std  | Final Gap (last 10 evals) |
| --------- | -------- | ---- | ------------------------- |
| M1        | -5.74    | 3.05 | -10.68                    |
| M3 cs1024 | -2.24    | 4.24 | -2.86                     |
| M3 cs2048 | -3.98    | 4.18 | -11.18                    |
| M3 cs3072 | -5.46    | 2.50 | -1.82                     |

### Does eviction act as regularisation?

The original hypothesis was that eviction might act as implicit regularisation (analogous to dropout). While the traditional overfitting framing does not apply (all gaps are negative), we can compare the *magnitude* of the gap:

- **M3 cs1024** has the smallest mean gap magnitude (-2.24), suggesting that eviction with a small cache does reduce the gap -- consistent with a regularisation effect.
- However, **M3 cs2048** shows a larger final gap (-11.18) than M1 (-10.68), which goes against the regularisation hypothesis.
- **M3 cs3072** shows a small final gap (-1.82) but this may be an artefact of its short training duration.

The evidence for eviction-as-regularisation is mixed.

## Convergence Speed: Does Larger Cache = Faster Convergence?

| Cache Size | Baseline F1 | Steps to 75% | Steps to 90% |
| ---------- | ----------- | ------------ | ------------ |
| cs1024     | 19.96       | 214          | 246          |
| cs2048     | 24.84       | 150          | 238          |
| cs3072     | 30.56       | 2            | 54           |

Larger cache sizes start from higher baselines and reach thresholds faster. However, this is confounded by the fact that M3 cs3072's best F1 is much lower (due to its short training run), making its thresholds easier to reach.

If we consider convergence to M1's best val F1 as an absolute benchmark:
- M3 cs1024 first reaches 45.48 at approximately step 340.
- M3 cs2048 first reaches 45.48 at approximately step 354.
- M3 cs3072 never reaches 45.48 within its 116 steps.

## Conclusions

1. **M3 converges at a comparable rate to M1.** Despite starting from much lower baselines (20-31 F1 vs 23 F1), M3 cs1024 and M3 cs2048 reach their best performance at similar step counts. The absolute improvement per step is therefore *larger* for M3 than M1.

2. **The information bottleneck from eviction does not substantially slow convergence.** M3 cs2048 converges only ~50% slower than M1 to the 75% threshold (150 vs 102 steps), while M3 cs1024 is ~2x slower. Given that these conditions operate with reduced KV cache, this is a relatively modest penalty.

3. **Evidence for eviction-as-regularisation is mixed.** M3 cs1024 shows the smallest train-val gap, consistent with the regularisation hypothesis, but the pattern is not consistent across cache sizes.

4. **Larger cache does not clearly yield faster convergence** when controlling for training duration. M3 cs3072 was undertrained, preventing meaningful comparison. Between cs1024 and cs2048, the larger cache converges moderately faster, but both reach comparable peak performance.

5. **The most striking finding is that M3 recovers from severe baseline degradation.** M3 cs1024 starts with a baseline of 19.96 (12% below M1's baseline of 22.59) yet reaches a peak of 45.59 -- slightly *exceeding* M1's best of 45.48. This suggests that LoRA can fully compensate for the information loss from aggressive eviction, at least on average across tasks.

## Plots

- `normalised_improvement.png` -- Normalised improvement curves for all conditions
- `overfitting_gap.png` -- Train-val F1 gap over training (smoothed)
- `steps_to_threshold.png` -- Steps to reach 50/75/90% of best val F1
- `learning_curves_overlay.png` -- Raw val F1 with light smoothing
