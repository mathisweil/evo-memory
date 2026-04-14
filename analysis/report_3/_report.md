# Analysis 3 -- Per-Layer Retention Pattern Analysis

> All values are **validation F1** (not test) where applicable.
> Retention patterns are **training-time** values extracted from WandB logs.
> Naming follows M0--M3 convention throughout.
> M3 checkpoint is step 260 (~43% through training); retention patterns may shift.

## 1. Per-Layer Retention

Retention is the fraction of KV-cache tokens retained (not evicted) at each
Transformer layer during M3 training. Higher retention = fewer tokens evicted.

| Layer | Retention |
|:------|----------:|
| 0     |    0.2169 |
| 1     |    0.2097 |
| 2     |    0.2169 |
| 3     |    0.2169 |
| 4     |    0.2169 |
| 5     |    0.2129 |
| 6     |    0.1872 |
| 7     |    0.2169 |
| 8     |    0.1495 |
| 9     |    0.1567 |
| 10    |    0.1600 |
| 11    |    0.2156 |
| 12    |    0.1965 |
| 13    |    0.2166 |
| 14    |    0.2036 |
| 15    |    0.2046 |

## 2. Aggregate Statistics

| Statistic                      | Value              |
|:-------------------------------|-------------------:|
| Mean retention                 |             0.1997 |
| Coefficient of Variation (CV)  |             0.1149 |
| Minimum retention              | 0.1495 (layer 8)  |
| Maximum retention              | 0.2169 (layer 0)  |
| Range (max - min)              |             0.0674 |

## 3. Eviction Pattern Structure

The NAMM policy learns a structured eviction pattern that is not uniform across
layers. Three distinct regimes emerge:

**High retention (> 0.21):** Layers 0--5, 7, 11, 13. These early and select
later layers retain the most tokens (~0.217), preserving positional and
token-identity information critical for downstream layers. The policy treats
these layers as largely non-compressible.

**Moderate retention (0.19--0.21):** Layers 6, 12, 14, 15. These layers see
moderate eviction, with retention between 0.187 and 0.205. They represent an
intermediate level of compressibility.

**Aggressive eviction (< 0.17):** Layers 8, 9, 10. These mid-network layers
receive the most eviction, retaining only 0.15--0.16 of tokens. They carry the
most redundant or compressible KV-cache information.

## 4. Layer-Level Observations

### Layers with the most aggressive eviction (lowest retention)

| Rank | Layer           | Retention |
|:-----|:----------------|----------:|
| 1    | Layer 8         |    0.1495 |
| 2    | Layer 9         |    0.1567 |
| 3    | Layer 10        |    0.1600 |

Layers 8--10 form a contiguous block in the middle of the 16-layer Transformer
that the NAMM policy identifies as most compressible. This is consistent with the
observation that mid-network representations tend to be more redundant: early
layers establish token-level features, final layers refine task-specific
representations, and mid-layers carry transitional information that can be
compressed without large downstream impact.

### Layers with the least eviction (highest retention)

Layers 0, 2, 3, 4, and 7 all retain ~0.217 tokens, near the maximum. Layer 0
is notable as the first layer -- the policy preserves its full output, likely
because all subsequent layers depend on it.

### Low CV indicates uniform eviction

The coefficient of variation (0.1149) is relatively low, meaning the NAMM policy
distributes eviction fairly evenly rather than concentrating it aggressively in a
few layers. No single layer is stripped below 0.15 retention, ensuring that
information is preserved at every layer to some degree.

## 5. Interpretation

The retention patterns tell a coherent story about how NAMM learns to compress the
KV cache of Llama-3.2-1B:

1. **The policy preserves early layers.** Layers 0--5 and 7 retain the most tokens,
   consistent with these layers carrying positional encoding and token-identity
   information that cannot be safely removed.

2. **Mid-network layers are the primary compression target.** Layers 8--10 are
   evicted most aggressively, suggesting they carry the most redundant information
   in the KV cache. This aligns with the general principle that intermediate
   Transformer representations are more compressible.

3. **Eviction is distributed, not concentrated.** The low CV (0.1149) and narrow
   range (0.0674) show that the policy spreads eviction across layers rather than
   stripping any single layer bare. This measured strategy correlates with the
   strong M3 val F1 of 52.06 -- the model preserves enough information at every
   layer to maintain downstream quality.

4. **The learned policy reflects genuine information structure.** Because eviction
   under correct attention masking has real consequences (evicted tokens are truly
   invisible to subsequent computation), the retention pattern reflects the actual
   compressibility of each layer rather than artifacts of information leakage.

## 6. Comparison with Pre-Correction (Buggy) Runs

Early M3 runs used a broken attention mask that allowed partial attention to
evicted tokens. Comparing the retention patterns reveals how the attention bug
distorted the learned eviction policy.

| Layer | Buggy | Corrected | Delta   |
|:------|------:|----------:|--------:|
| 0     | 0.2197|    0.2169 | -0.0029 |
| 1     | 0.1917|    0.2097 | +0.0181 |
| 2     | 0.2197|    0.2169 | -0.0029 |
| 3     | 0.2197|    0.2169 | -0.0029 |
| 4     | 0.2197|    0.2169 | -0.0029 |
| 5     | 0.2172|    0.2129 | -0.0043 |
| 6     | 0.1537|    0.1872 | +0.0335 |
| 7     | 0.2197|    0.2169 | -0.0029 |
| 8     | 0.1159|    0.1495 | +0.0336 |
| 9     | 0.1143|    0.1567 | +0.0424 |
| 10    | 0.1743|    0.1600 | -0.0143 |
| 11    | 0.2197|    0.2156 | -0.0041 |
| 12    | 0.2194|    0.1965 | -0.0229 |
| 13    | 0.2196|    0.2166 | -0.0031 |
| 14    | 0.1956|    0.2036 | +0.0081 |
| 15    | 0.2001|    0.2046 | +0.0045 |

The buggy policy was more aggressive at layers 6, 8, 9 (retaining only
0.11--0.15 of tokens) because the attention leak preserved access to "evicted"
tokens. Under correct masking, the corrected policy backs off at these layers
(+0.03--0.04 more tokens retained) and produces a substantially more uniform
pattern (CV 0.1149 vs 0.1826, a 37.1% reduction). The corrected, more uniform
eviction strategy correlates with the higher M3 val F1 (52.06 vs 45.59).

## 7. Caveats

- These are training-time retention patterns from WandB logs, not inference-time.
  Retention behaviour may differ at evaluation time.
- M3 has completed only ~43% of training. Retention patterns may shift as
  training continues and the NAMM policy evolves further.
- Retention values are averages over the training data distribution and may vary
  substantially across individual examples.

## Plots

| Plot                                                         | Description                               |
| ------------------------------------------------------------ | ----------------------------------------- |
| [`layer_retention_profile.png`](layer_retention_profile.png) | Mean retention per layer (bar chart)       |
| [`retention_heatmap.png`](retention_heatmap.png)             | Retention by layer and training step       |
| [`retention_over_training.png`](retention_over_training.png) | Per-layer retention curves                 |
| [`retention_vs_f1.png`](retention_vs_f1.png)                 | Retention vs val F1 scatter                |
