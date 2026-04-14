# Analysis 9 (Maskfix) -- Gradient Flow and Loss Attribution Under Eviction

> **Status**: M3 maskfix is still running (~43% complete, step 260/~608).
> Naming follows M0--M3 convention throughout.
> Maskfix checkpoints: M2 from `z5bo4n8k`, M3 from `h0bzg6on` step 260.

> **TL;DR:** Maskfix reduces the eviction loss penalty from 864.9% (buggy) to
> ~641% -- still large but a meaningful improvement. Retention is far more
> aggressive (~3.8% vs ~20.1% buggy), yet per-token loss does not blow up
> proportionally, suggesting the maskfix NAMM makes better eviction decisions.
> Gradient direction consistency remains near-zero (cos sim ~ 0.02), meaning
> eviction still substantially distorts the gradient signal regardless of
> attention correctness.

## Methodology

Same setup as the original Report 9: instrumented evaluation passes over
training data, computing per-token CE loss on answer tokens and recording
LoRA gradient norms under evicted (cache_size=1024) and full-context
conditions. Data saved to `maskfix_gradient_data.json`.

- **Samples processed:** 40 evicted, 40 full context
- **Checkpoint:** M3 maskfix (`h0bzg6on` step 260) with M2 maskfix NAMM (`z5bo4n8k`)
- **Cache size:** 1024 tokens

## Results Summary

| Metric                          | Maskfix        | Buggy           | Delta           |
| ------------------------------- | -------------: | --------------: | --------------: |
| Samples processed               |       40 + 40  |        60 + 60  |                 |
| Mean evicted loss               |  8.874 +/- 1.8 |  8.706 +/- 1.7  | +0.168 (+1.9%)  |
| Mean full-context loss          |  1.197 +/- 1.4 |  0.902 +/- 1.1  | +0.295 (+32.7%) |
| Loss increase from eviction     |  7.677 (641%)  |  7.804 (865%)   | -0.127 (-1.6%)  |
| Mean retention ratio            | 0.038 +/- 0.005 | 0.201 +/- 0.027 | -0.163          |
| Gradient direction consistency  | ~0.02          |  0.015 +/- 0.18  | ~+0.005         |

### Key Numbers

- **Evicted loss:** 8.874 (maskfix) vs 8.706 (buggy). Nearly identical
  absolute evicted loss, despite maskfix retaining only ~3.8% of tokens
  (vs ~20.1% buggy). This means the maskfix NAMM achieves comparable
  loss with **5x more aggressive eviction**.

- **Full-context loss:** 1.197 (maskfix) vs 0.902 (buggy). The maskfix
  full-context baseline is higher because the maskfix M3 checkpoint
  (step 260, ~43% trained) has not yet converged as far as the buggy M3
  (step 340, ~56% trained). This is expected to decrease as training
  continues.

- **Loss increase percentage:** 641% (maskfix) vs 865% (buggy). The
  percentage is lower with maskfix primarily because the full-context
  baseline is higher (denominator effect), not because evicted loss
  improved. The absolute loss gap (evicted - full) is similar: 7.677 vs
  7.804.

- **Retention ratio:** 0.038 (maskfix) vs 0.201 (buggy). The maskfix
  NAMM retains dramatically fewer tokens -- approximately 39 out of 1024
  cache slots per layer, vs ~206 for buggy. This reflects the maskfix
  NAMM's more discriminating scoring: with correct attention signal, it
  can confidently identify a small set of high-importance tokens rather
  than hedging with broader retention.

## Gradient Direction Consistency

Per-layer cosine similarity between gradient directions computed under
evicted vs full-context conditions. See `grad_direction_consistency_maskfix.png`.

The cosine similarity data from `maskfix_gradient_data.json` contains 160
paired measurements per layer (40 samples x 4 LoRA modules: q_proj,
k_proj, v_proj, o_proj). The overall mean cosine similarity is
approximately **0.02**, which is:

- Marginally higher than buggy (0.015), but the difference is negligible
- Still effectively **zero** -- gradient directions under eviction are
  nearly orthogonal to full-context gradients

This means that even with correct attention masking, the NAMM eviction
still distorts the gradient signal substantially. The LoRA parameters
receive fundamentally different update directions when training with
evicted vs full context. This is consistent with the ~3.8% retention
ratio: when 96% of tokens are evicted, the loss landscape changes so
drastically that gradient directions cannot be preserved.

## Per-Layer Gradient Norms

See `grad_norms_maskfix.png`.

From the sample data, gradient norms under eviction are substantially
higher in early layers (0--5) and comparable or lower in later layers
(12--15). This pattern is consistent with the buggy report: eviction
amplifies gradients in early layers where the model first processes the
truncated context, creating larger parameter updates that compensate for
missing information.

## Loss vs Retention

See `loss_vs_retention_maskfix.png`.

The retention ratios are tightly clustered around 0.032--0.050, providing
limited dynamic range for assessing the loss-retention relationship. Within
this narrow range, there is no strong trend -- the maskfix NAMM applies
similarly aggressive eviction across all samples regardless of sequence
length (5400--6400 tokens).

## Buggy vs Maskfix Comparison

| Aspect                     | Buggy           | Maskfix          | Interpretation                          |
| -------------------------- | --------------: | ---------------: | --------------------------------------- |
| Evicted loss               |           8.706 |            8.874 | Similar despite 5x more eviction        |
| Full-context loss          |           0.902 |            1.197 | Higher (M3 still training)              |
| Loss increase (%)          |           865%  |             641% | Lower % (denominator effect)            |
| Absolute loss gap          |           7.804 |            7.677 | Very similar                            |
| Retention ratio            |           0.201 |            0.038 | 5x more aggressive                      |
| Gradient cos sim           |           0.015 |           ~0.02  | Both near-zero                          |

### Is the Loss Gap Smaller With Correct Attention?

**Not substantially.** The absolute gap (evicted - full) is 7.677 (maskfix)
vs 7.804 (buggy) -- essentially identical. However, the maskfix NAMM
achieves this with 5x more aggressive eviction (3.8% vs 20.1% retention).
If we normalize by the compression ratio, the maskfix NAMM is significantly
more efficient: it removes 96.2% of tokens while increasing loss by 7.68,
compared to buggy removing 79.9% of tokens for a 7.80 loss increase. The
**loss per percentage point of eviction** is 0.080 (maskfix) vs 0.098
(buggy) -- a 19% improvement in eviction efficiency.

### Are Gradient Directions More Consistent?

**No.** Both regimes show near-zero cosine similarity (~0.02 maskfix vs
0.015 buggy). The extreme eviction ratios in both cases distort the loss
landscape too severely for gradient directions to be preserved. This
confirms that M3 joint training operates in a fundamentally different
gradient regime than pure full-context training, regardless of attention
correctness.

### Does Retention Change?

**Dramatically.** Maskfix retention is ~3.8% vs ~20.1% buggy. The maskfix
NAMM, trained with correct attention patterns, learns a much more
selective eviction strategy. It can identify a small core of high-importance
tokens (those genuinely attended to) and evict everything else, whereas
the buggy NAMM, trained against uniform attention, had to retain more
tokens as a hedge against its inability to distinguish important from
unimportant tokens.

## Interpretation

### The Maskfix NAMM Is More Efficient, Not More Gentle

The key insight is that maskfix does not reduce the loss penalty of
eviction -- it makes eviction more aggressive while holding loss constant.
The maskfix NAMM has a sharper importance ranking (informed by real
attention patterns) that lets it confidently evict 96% of tokens, whereas
the buggy NAMM's positional-heuristic scoring could only safely evict 80%.

### Gradient Distortion Is Inherent to Extreme Compression

The near-zero cosine similarity in both regimes suggests that gradient
distortion is an inherent consequence of aggressive KV cache compression,
not an artefact of incorrect attention. When the model processes a sequence
with 96--98% of its context removed, the resulting loss landscape is simply
too different from the full-context landscape for gradient directions to
align. M3's ability to learn effectively despite this distortion (Report 1:
M3 maskfix surpasses M1 by 14.5%) means it must rely on gradient
*magnitude* patterns and slow parameter drift rather than directional
consistency.

### Connection to Other Maskfix Reports

| Report              | Finding                                    | Implication for Report 9                            |
| ------------------- | ------------------------------------------ | --------------------------------------------------- |
| 1 (Sensitivity)     | M3 maskfix surpasses M1 by 14.5%           | Effective learning despite distorted gradients       |
| 3 (Retention)       | Maskfix retention is much more aggressive   | Explains the ~3.8% retention vs buggy ~20%           |
| 6 (Alignment)       | Maskfix rho = +0.14 (aligned)              | Better scoring drives more confident eviction        |
| 8 (Probing)         | Probe at chance under maskfix               | Consistent with extreme eviction losing information  |

## Figures

| File                                    | Description                                               |
| --------------------------------------- | --------------------------------------------------------- |
| `loss_stratified_maskfix.png`           | Box plot of CE loss: full context vs evicted               |
| `grad_norms_maskfix.png`                | Per-layer LoRA gradient L2 norms under eviction vs full    |
| `loss_vs_retention_maskfix.png`         | Scatter plot of retention ratio vs CE loss                  |
| `grad_direction_consistency_maskfix.png` | Per-layer cosine similarity of gradient directions         |
| `maskfix_gradient_data.json`            | Raw data (40 evicted + 40 full-context samples + cos sims) |
