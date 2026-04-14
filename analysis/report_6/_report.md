# Analysis 6 (Maskfix) -- Token Importance Alignment (NAMM Scores vs Attention)

> **Status**: M3 maskfix is still running (~43% complete, step 260/~608).
> Naming follows M0--M3 convention throughout.
> Maskfix checkpoints: M2 from `z5bo4n8k`, M3 from `h0bzg6on` step 260.

> **TL;DR:** The NAMM-attention correlation **flipped** from buggy to maskfix:
> buggy rho = -0.137 (anti-correlated) vs maskfix rho = +0.135 (positively
> correlated). With correct attention masking, NAMM scores ARE weakly aligned
> with attention weights -- NAMM retains tokens the model attends to. The
> buggy anti-correlation was an artefact of the uniform-attention bug, not a
> genuine "complementary signals" phenomenon. M1 and M3 maskfix alignment is
> nearly identical (0.135 vs 0.140), meaning M3 fine-tuning still does not
> reshape attention to better agree with NAMM.

## Methodology

Same setup as the original Report 6 (two-pass: full-context attention
extraction, then NAMM `update_cache(analyze=True)`), but using maskfix
checkpoints where the attention mask is correctly applied during both
M2 evolution and M3 joint training.

- **Checkpoints:** M1 (LoRA full-context), M3 maskfix (LoRA + frozen NAMM,
  `h0bzg6on` step 260), both using M2 maskfix NAMM (`z5bo4n8k`).
- **Test data:** 3 samples per task from the 5-task test set (15 samples total).
- **Cache size:** 1024 tokens.

## Findings

### Score-Attention Correlation

| Metric                         | M1 (maskfix) | M3 maskfix | M1 (buggy) | M3 (buggy) |
| ------------------------------ | -----------: | ---------: | ---------: | ---------: |
| Mean Spearman rho (all layers) |       +0.135 |     +0.140 |     -0.137 |     -0.136 |
| Std Spearman rho               |        0.139 |      0.135 |      0.070 |      0.071 |

See `score_attention_correlation_maskfix.png`.

The sign flip is the headline result: with correct attention masking, NAMM
scores are **positively** correlated with attention weights at rho ~ +0.14.
This means NAMM preferentially retains tokens that the model attends to --
exactly what one would expect from a well-functioning eviction policy.

### Why the Buggy Correlation Was Negative

Under the attention-mask bug, the causal mask was not applied during the
forward pass used for M2 NAMM evolution. This produced **uniform attention**
(every token attends equally to every other token), which had two
consequences:

1. **All tokens looked equally attended.** With no meaningful attention
   gradient, NAMM's scoring was driven by the positional/temporal features
   in the BAM spectrogram (token age, position) rather than by actual
   attention patterns. Older tokens received lower scores, producing a
   systematic anti-correlation with the (correct) attention weights that
   assign varying importance to tokens.

2. **The "complementary signals" narrative was wrong.** The original
   Report 6 interpreted the negative correlation as evidence that NAMM
   operates on a fundamentally different importance signal than attention.
   In reality, it was describing a bug: NAMM was trained against a
   corrupted attention signal and learned to ignore attention entirely,
   falling back to positional heuristics.

### Eviction Regret

| Metric                         | M1 (maskfix) | M3 maskfix | M1 (buggy) | M3 (buggy) |
| ------------------------------ | -----------: | ---------: | ---------: | ---------: |
| Mean total regret (all layers) |        0.289 |    similar |      0.070 |      0.067 |

See `eviction_regret_maskfix.png`.

Total regret is **higher** with maskfix (~0.289 vs ~0.070 in buggy). This
is not paradoxical -- with correct attention, attention weights are no
longer near-uniform, so evicting any token removes a larger fraction of
the total attention mass. Under the buggy uniform attention, every token
had approximately equal (and tiny) per-token attention, so evicting 80% of
tokens only removed ~7% of total attention mass. With correct (peaked)
attention, the same eviction removes more.

### Alignment Shift (M1 vs M3)

See `alignment_shift_maskfix.png`.

| Metric                  | M1 (maskfix) | M3 maskfix | Delta |
| ----------------------- | -----------: | ---------: | ----: |
| Mean Spearman rho       |       +0.135 |     +0.140 | +0.005 |

M1 and M3 maskfix alignment is nearly identical, meaning M3 joint
training does not reshape attention to better agree with NAMM's scoring.
This is consistent with the buggy result (where M1 and M3 were also
identical at rho ~ -0.137). In both regimes, the LoRA adaptation does not
converge toward the NAMM importance ranking.

## Buggy vs Maskfix Comparison

| Aspect                    | Buggy                            | Maskfix                          |
| ------------------------- | -------------------------------- | -------------------------------- |
| NAMM-attention rho        | -0.137 (anti-correlated)         | +0.135 (positively correlated)   |
| Interpretation            | NAMM uses "complementary signal" | NAMM retains attended tokens     |
| M1 vs M3 shift            | Negligible                       | Negligible                       |
| Total eviction regret     | 0.070 (low)                      | 0.289 (higher, expected)         |
| Rho std across layers     | 0.070 (tight)                    | 0.139 (wider spread)             |

The wider standard deviation under maskfix reflects genuine
layer-by-layer variation in alignment. Under buggy uniform attention,
layers were all similarly (negatively) correlated because the attention
signal was uniformly uninformative. With correct attention, some layers
(e.g. early layers with broad attention) show different alignment than
deep layers with focused attention.

## Interpretation

### NAMM Does Track Attention -- Weakly

The positive correlation (rho = 0.135) confirms that NAMM's learned
scoring function, when trained against correct attention patterns, does
capture some of the same importance signal as attention weights. However,
the correlation is weak (rho ~ 0.14, R-squared ~ 2%), meaning NAMM
relies heavily on additional features beyond raw attention magnitude --
the BAM spectrogram temporal patterns, positional information, and
cross-layer statistics.

### M3 Still Does Not Reshape Attention Toward NAMM

Even with correct attention, M3 fine-tuning does not increase the
NAMM-attention correlation (0.140 vs 0.135, negligible difference). This
confirms that the LoRA adaptation strategy is not "learn to attend to
what NAMM retains" but rather "learn to extract sufficient information
regardless of what NAMM retains." The Report 5 maskfix finding of +4%
attention entropy supports this: M3 broadens attention as a hedging
strategy rather than sharpening it toward NAMM's retained tokens.

### Connection to Other Maskfix Reports

| Report              | Finding                                  | Implication for Report 6                          |
| ------------------- | ---------------------------------------- | ------------------------------------------------- |
| 1 (Sensitivity)     | M3 maskfix surpasses M1 by 14.5%         | Positive alignment + hedging = effective strategy  |
| 3 (Retention)       | Maskfix retention is much more aggressive | Higher regret is expected with fewer retained tokens |
| 5 (Attention)       | M3 maskfix has higher entropy             | Hedging strategy, not alignment strategy           |

## Figures

- `score_attention_correlation_maskfix.png` -- Spearman rho between NAMM scores and mean attention received, per layer, for M1 and M3 maskfix
- `eviction_regret_maskfix.png` -- Total and per-token attention mass on evicted tokens, per layer
- `alignment_shift_maskfix.png` -- Per-task comparison of mean alignment (Spearman rho) between M1 and M3 maskfix
