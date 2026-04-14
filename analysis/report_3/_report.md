# Analysis 3 (Maskfix) -- Per-Layer Retention Pattern Analysis

> **Status**: M3 maskfix is still running (~43% complete, step 298/~608).
> All values are **validation F1** (not test) where applicable.
> Retention patterns are **training-time** values extracted from WandB logs.
> Naming follows M0--M3 convention throughout.

## 1. Per-Layer Retention: Buggy vs Maskfix

Retention is the fraction of KV-cache tokens retained (not evicted) at each
Transformer layer during M3 training. Higher retention = fewer tokens evicted.

| Layer  | Buggy Retention | Maskfix Retention | Difference (M-B) |
|:-------|----------------:|------------------:|------------------:|
| 0      |          0.2197 |            0.2169 |           -0.0029 |
| 1      |          0.1917 |            0.2097 |           +0.0181 |
| 2      |          0.2197 |            0.2169 |           -0.0029 |
| 3      |          0.2197 |            0.2169 |           -0.0029 |
| 4      |          0.2197 |            0.2169 |           -0.0029 |
| 5      |          0.2172 |            0.2129 |           -0.0043 |
| 6      |          0.1537 |            0.1872 |           +0.0335 |
| 7      |          0.2197 |            0.2169 |           -0.0029 |
| 8      |          0.1159 |            0.1495 |           +0.0336 |
| 9      |          0.1143 |            0.1567 |           +0.0424 |
| 10     |          0.1743 |            0.1600 |           -0.0143 |
| 11     |          0.2197 |            0.2156 |           -0.0041 |
| 12     |          0.2194 |            0.1965 |           -0.0229 |
| 13     |          0.2196 |            0.2166 |           -0.0031 |
| 14     |          0.1956 |            0.2036 |           +0.0081 |
| 15     |          0.2001 |            0.2046 |           +0.0045 |

## 2. Aggregate Statistics

| Statistic                      | Buggy  | Maskfix |
|:-------------------------------|-------:|--------:|
| Coefficient of Variation (CV)  | 0.1826 |  0.1149 |
| Minimum retention              | 0.1143 (layer 9) | 0.1495 (layer 8) |
| Maximum retention              | 0.2197 (layer 0) | 0.2169 (layer 0) |
| Range (max - min)              | 0.1054 |  0.0674 |

## 3. Is Eviction More Uniform with Correct Attention?

**Yes.** The maskfix NAMM produces substantially more uniform eviction across layers:

- **CV drops from 0.1826 to 0.1149** -- a 37.1% reduction in relative variability.
  The maskfix policy distributes eviction more evenly rather than concentrating it
  in a few layers.
- **The retention range narrows from 0.1054 to 0.0674** -- the gap between the
  most and least aggressive layers shrinks by 36.1%.
- **The minimum retention rises from 0.1143 to 0.1495** -- the most aggressively
  evicted layer retains 30.8% more tokens under maskfix. This means no single layer
  is stripped as severely.

The buggy attention mask allowed the NAMM policy to rely on information leaking
through "evicted" tokens, meaning it could afford to aggressively evict at certain
layers (8, 9) without true information loss. Under correct masking, aggressive
eviction at any layer has real consequences, so the policy learns a more
conservative and uniform eviction pattern.

## 4. Layer-Level Observations

### Layers with the most aggressive eviction (lowest retention)

| Rank | Buggy              | Maskfix            |
|:-----|:-------------------|:-------------------|
| 1    | Layer 9 (0.1143)   | Layer 8 (0.1495)   |
| 2    | Layer 8 (0.1159)   | Layer 9 (0.1567)   |
| 3    | Layer 6 (0.1537)   | Layer 10 (0.1600)  |

Both conditions identify layers 8--9 as the primary eviction targets -- these are
mid-network layers that apparently carry the most redundant or compressible
information in the KV cache. Layer 6 is the third-most-evicted in both conditions
(third for buggy at 0.1537, fourth for maskfix at 0.1872).

### Layers with the least eviction (highest retention)

Both conditions retain the most tokens at the early layers (0, 2, 3, 4) and
layer 7, with retention values near 0.22 (buggy) and 0.217 (maskfix). This
suggests the NAMM policy in both regimes learns to preserve early-layer
representations, which carry positional and token-identity information critical
for downstream layers.

### Layers where buggy and maskfix diverge most

| Layer | Buggy  | Maskfix | Delta   | Direction             |
|:------|:-------|:--------|:--------|:----------------------|
| 9     | 0.1143 | 0.1567  | +0.0424 | Maskfix retains more  |
| 8     | 0.1159 | 0.1495  | +0.0336 | Maskfix retains more  |
| 6     | 0.1537 | 0.1872  | +0.0335 | Maskfix retains more  |
| 12    | 0.2194 | 0.1965  | -0.0229 | Maskfix retains less  |
| 1     | 0.1917 | 0.2097  | +0.0181 | Maskfix retains more  |

The largest divergences are at layers 6, 8, and 9 -- exactly the layers where
buggy eviction was most aggressive. Under maskfix, the policy "backs off" at
these layers, retaining 0.03--0.04 more tokens. This is consistent with the
interpretation that the buggy mask allowed over-eviction at these layers because
the information leaked through anyway.

Layer 12 is the only layer where maskfix evicts *more* aggressively than buggy
(0.1965 vs 0.2194). This may indicate that with correct attention, layer 12
representations become more compressible -- or that the maskfix policy
redistributes some eviction budget from mid-layers to this later layer.

## 5. Interpretation

The retention patterns tell a coherent story:

1. **Under broken attention, NAMM learns a false sense of what is safe to evict.**
   Layers 8--9 are evicted very aggressively (retaining only ~11% of tokens) because
   the attention leak preserves access to "evicted" tokens.

2. **Under correct attention, NAMM must be more conservative.** The same layers
   still receive the most eviction (they carry the most compressible information),
   but retention rises to ~15%, and the overall eviction pattern becomes more
   uniform.

3. **More uniform eviction correlates with better downstream performance.** The
   maskfix M3 achieves 52.06 val F1 vs 45.59 for buggy, suggesting that the more
   measured eviction strategy (CV 0.11 vs 0.18) preserves information more
   effectively overall.

4. **The NAMM policy structure is qualitatively similar in both regimes.** The same
   layers are targeted for eviction in both conditions; the difference is one of
   degree, not kind. This suggests the underlying layer-wise information structure
   of Llama-3.2-1B is consistent, and the maskfix correction primarily affects how
   aggressively the policy can exploit that structure.

## 6. Caveats

- These are training-time retention patterns from WandB logs, not inference-time.
  Retention behaviour may differ at evaluation time.
- M3 maskfix has completed only ~43% of training. Retention patterns may shift as
  training continues and the NAMM policy evolves further.
- Retention values reflect the NAMM's evolved policy at the logged training step
  and may not represent the final converged policy.
- The retention values are averages over the training data distribution and may vary
  substantially across individual examples.
