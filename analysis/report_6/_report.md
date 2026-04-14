# Analysis 6 -- Token Importance Alignment (NAMM Scores vs Attention)

> Naming follows M0--M3 convention throughout.
> Checkpoints: M2 NAMM from WandB `z5bo4n8k`, M3 LoRA from `h0bzg6on` step 260.

> **TL;DR:** NAMM scores are weakly positively correlated with attention
> weights (Spearman rho = +0.135 for M1, +0.140 for M3). NAMM
> preferentially retains tokens the model attends to, consistent with a
> well-functioning eviction policy. However, the correlation is weak
> (R-squared ~2%), indicating NAMM relies heavily on additional features
> beyond raw attention magnitude. M3 fine-tuning does not reshape
> attention to better agree with NAMM -- the hedging strategy (Report 5)
> broadens attention rather than sharpening it toward retained tokens.

## Methodology

Two-pass analysis: full-context attention extraction, then NAMM
`update_cache(analyze=True)` to obtain token importance scores.

- **Checkpoints:** M1 (LoRA full-context, val F1 45.48), M3 (LoRA + frozen NAMM,
  `h0bzg6on` step 260, val F1 52.06), both using M2 NAMM (`z5bo4n8k`).
- **Test data:** 3 samples per task from the 5-task test set (15 samples total).
- **Cache size:** 1024 tokens.

## Findings

### Score-Attention Correlation

| Metric                         | M1     | M3     |
| ------------------------------ | -----: | -----: |
| Mean Spearman rho (all layers) | +0.135 | +0.140 |
| Std Spearman rho               |  0.139 |  0.135 |

See `plots/score_attention_correlation.png`.

NAMM scores are positively correlated with attention weights at rho ~+0.14.
This means NAMM preferentially retains tokens that the model attends to --
exactly what one would expect from a well-functioning eviction policy.

### M3 Does Not Reshape Attention Toward NAMM

| Metric            | M1     | M3     | Delta  |
| ----------------- | -----: | -----: | -----: |
| Mean Spearman rho | +0.135 | +0.140 | +0.005 |

See `plots/alignment_shift.png`.

M1 and M3 alignment is nearly identical, meaning M3 joint training does
not reshape attention to better agree with NAMM's scoring. The LoRA
adaptation strategy is not "learn to attend to what NAMM retains" but
rather "learn to extract sufficient information regardless of what NAMM
retains." This is consistent with the Report 5 finding that M3 broadens
attention as a hedging strategy rather than sharpening it toward NAMM's
retained tokens.

### Eviction Regret

| Metric                         | M1    | M3      |
| ------------------------------ | ----: | ------: |
| Mean total regret (all layers) | 0.289 | similar |

See `plots/eviction_regret.png`.

Total regret measures the attention mass assigned to evicted tokens --
higher regret means the model is attending to tokens that NAMM discards.
The regret of 0.289 reflects the fact that with correct (peaked)
attention distributions, evicting a large fraction of the cache
inevitably removes some attended tokens. This is expected and does not
indicate a failure of the eviction policy; it instead reflects the
fundamental trade-off between cache compression and information loss.

### Layer-by-Layer Variation

The standard deviation of Spearman rho across layers (0.139 for M1,
0.135 for M3) reflects genuine layer-by-layer variation in alignment.
Early layers with broad attention show different alignment than deep
layers with focused attention, consistent with the hierarchical nature
of transformer representations.

## Interpretation

### NAMM Tracks Attention -- Weakly

The positive correlation (rho = 0.135) confirms that NAMM's learned
scoring function captures some of the same importance signal as attention
weights. However, the correlation is weak (rho ~0.14, R-squared ~2%),
meaning NAMM relies heavily on additional features beyond raw attention
magnitude -- the BAM spectrogram temporal patterns, positional
information, and cross-layer statistics.

### The Hedging-Alignment Trade-off

M3 could in principle have learned to sharpen its attention toward tokens
that NAMM retains, increasing alignment and reducing regret. Instead, it
broadens attention (Report 5: +5.0% entropy). This suggests the hedging
strategy is more effective than the alignment strategy: rather than
trusting NAMM's retention decisions and focusing on retained tokens, M3
hedges against NAMM's errors by attending broadly. The 14.5% F1
improvement over M1 (Report 1) validates this approach.

### Connection to Other Reports

| Report          | Finding                             | Implication                                       |
| --------------- | ----------------------------------- | ------------------------------------------------- |
| 1 (Sensitivity) | M3 surpasses M1 by 14.5%           | Positive alignment + hedging = effective strategy  |
| 3 (Retention)   | Aggressive retention pattern        | Higher regret expected with fewer retained tokens  |
| 5 (Attention)   | M3 has +5.0% higher entropy         | Hedging strategy, not alignment strategy           |

## Historical note: buggy correlation was misleading

Prior to the attention mask fix, NAMM-attention correlation was negative
(rho = -0.137 for both M1 and M3). The original Report 6 interpreted
this as evidence that NAMM operates on a "complementary signal"
fundamentally different from attention. In reality, the bug produced a
misleading anti-correlation: without a correct causal mask during M2
evolution, attention was near-uniform, so NAMM's scoring was driven by
positional heuristics (token age, position) rather than actual attention
patterns. Fixing the mask revealed the true positive correlation. The
buggy eviction regret was also misleadingly low (0.070 vs 0.289) because
near-uniform attention meant evicting any token removed negligible
attention mass.

## Figures

- `plots/score_attention_correlation.png` -- Spearman rho between NAMM scores and mean attention received, per layer, for M1 and M3
- `plots/eviction_regret.png` -- Total and per-token attention mass on evicted tokens, per layer
- `plots/alignment_shift.png` -- Per-task comparison of mean alignment (Spearman rho) between M1 and M3
