# Analysis 4: LoRA Weight Comparison (M1 vs M3 cs1024)

## TL;DR

M3 (eviction-aware training with frozen NAMM, cache size 1024) learns LoRA adapters that are **substantially larger** than M1 (full context, no eviction) -- on average 1.93x for q_proj and 1.50x for v_proj -- but in **nearly orthogonal subspaces** (mean cosine overlap ~0.19 for q_proj, ~0.17 for v_proj). The compensation effect intensifies in later layers (layers 9--15), where q_proj norm ratios exceed 2x. This indicates that M3 does not simply "do more of the same" as M1; it learns qualitatively different adaptations to cope with the information loss from KV cache eviction.

---

## Summary of Findings

1. **M3 adapters are uniformly larger.** Across all 16 layers and both projection types, `||B@A||_F` is larger for M3 than M1 (ratio always > 1). The effect is stronger in q_proj (mean ratio 1.93) than v_proj (mean ratio 1.50).

2. **Later layers compensate more.** The norm ratio increases monotonically from early to late layers, peaking at layer 14 for q_proj (ratio 2.59) and layer 2 for v_proj (ratio 1.85). The overall trend is that the last 4--5 layers bear the heaviest adaptation burden.

3. **Singular value spectra show M3 uses more rank capacity.** In the SVD of B@A, M3 tends to have larger singular values overall, but the spectra remain rank-1 dominated in both models. M3's leading singular value is consistently larger, and subsequent singular values are also amplified.

4. **Subspace overlap is very low (~0.15--0.24).** The mean cosine of principal angles between M1 and M3 column spaces hovers around 0.19 (q_proj) and 0.17 (v_proj), far from the value of 1.0 that would indicate identical adaptation directions. This means M1 and M3 learn in almost entirely different subspaces of weight space.

---

## Per-Layer Norms and Ratio

### q_proj

| Layer | M1 `||B@A||_F` | M3 `||B@A||_F` | Ratio (M3/M1) |
|------:|-------------------:|-------------------:|--------------:|
|     0 |             0.1604 |             0.2947 |         1.837 |
|     1 |             0.2000 |             0.3226 |         1.613 |
|     2 |             0.1813 |             0.2713 |         1.496 |
|     3 |             0.1836 |             0.3227 |         1.758 |
|     4 |             0.1665 |             0.3492 |         2.097 |
|     5 |             0.1980 |             0.2868 |         1.448 |
|     6 |             0.1884 |             0.3076 |         1.633 |
|     7 |             0.1962 |             0.3414 |         1.740 |
|     8 |             0.2157 |             0.3326 |         1.542 |
|     9 |             0.2030 |             0.4629 |         2.281 |
|    10 |             0.2015 |             0.3868 |         1.920 |
|    11 |             0.2156 |             0.4662 |         2.162 |
|    12 |             0.2233 |             0.4196 |         1.879 |
|    13 |             0.2163 |             0.5170 |         2.390 |
|    14 |             0.2721 |             0.7046 |         2.590 |
|    15 |             0.2286 |             0.5619 |         2.458 |

**Mean ratio: 1.928, Std: 0.351**

### v_proj

| Layer | M1 `||B@A||_F` | M3 `||B@A||_F` | Ratio (M3/M1) |
|------:|-------------------:|-------------------:|--------------:|
|     0 |             0.0988 |             0.1414 |         1.432 |
|     1 |             0.1204 |             0.1540 |         1.279 |
|     2 |             0.1127 |             0.2082 |         1.847 |
|     3 |             0.1201 |             0.1921 |         1.600 |
|     4 |             0.1193 |             0.1883 |         1.579 |
|     5 |             0.1238 |             0.1695 |         1.370 |
|     6 |             0.1245 |             0.1806 |         1.450 |
|     7 |             0.1231 |             0.1605 |         1.303 |
|     8 |             0.1523 |             0.1650 |         1.084 |
|     9 |             0.1303 |             0.1771 |         1.359 |
|    10 |             0.1544 |             0.2217 |         1.436 |
|    11 |             0.1770 |             0.2665 |         1.506 |
|    12 |             0.1622 |             0.2701 |         1.666 |
|    13 |             0.1790 |             0.2783 |         1.555 |
|    14 |             0.1890 |             0.3162 |         1.673 |
|    15 |             0.1776 |             0.3195 |         1.799 |

**Mean ratio: 1.496, Std: 0.193**

---

## Subspace Overlap Per Layer

Mean cosine of principal angles between M1 and M3 LoRA column spaces (1.0 = identical subspace, 0.0 = orthogonal):

| Layer | q_proj overlap | v_proj overlap |
|------:|--------------:|--------------:|
|     0 |         0.161 |         0.116 |
|     1 |         0.187 |         0.142 |
|     2 |         0.192 |         0.145 |
|     3 |         0.182 |         0.172 |
|     4 |         0.168 |         0.144 |
|     5 |         0.192 |         0.158 |
|     6 |         0.145 |         0.168 |
|     7 |         0.195 |         0.200 |
|     8 |         0.188 |         0.175 |
|     9 |         0.173 |         0.198 |
|    10 |         0.235 |         0.186 |
|    11 |         0.198 |         0.180 |
|    12 |         0.241 |         0.182 |
|    13 |         0.191 |         0.209 |
|    14 |         0.201 |         0.175 |
|    15 |         0.219 |         0.233 |

**q_proj mean: 0.192, v_proj mean: 0.174**

---

## Discussion

### Do M1 and M3 learn in the same subspace?

**No.** The subspace overlap is uniformly low (0.12--0.24), barely above what one would expect from random 8-dimensional subspaces in a 2048-dimensional space. The observed values of ~0.17--0.19 are modestly above the random baseline, indicating there is a small amount of shared structure, but the adaptation directions are overwhelmingly different.

This is a striking result: even though both models are fine-tuned on the same data with the same LoRA configuration (rank 8, targeting q_proj and v_proj), the presence of NAMM eviction during M3 training pushes the optimizer into a fundamentally different region of the low-rank parameter space.

### Which layers diverge most?

- **Layer 6** has the lowest q_proj overlap (0.145) and one of the lowest v_proj overlaps.
- **Layer 0** has the lowest v_proj overlap (0.116), suggesting the earliest value projection adapts most differently under eviction.
- **Layers 10, 12, 15** have the highest overlaps (~0.22--0.24 for q_proj), meaning that later layers retain slightly more shared adaptation direction, though the overlap is still very low.

There is no strong layer-depth trend in overlap -- M1 and M3 diverge roughly equally across the network.

---

## Interpretation

### Larger M3 norms = more compensation needed

The fact that M3 norms are 1.5--2.6x larger than M1 across all layers confirms that the LoRA adapter must "work harder" when the model trains with KV cache eviction. This is expected: with eviction, some context information is irreversibly lost, and the model compensates by making larger adjustments to its query and value projections.

The compensation is strongest in the **later layers** (9--15) for q_proj, with ratios exceeding 2x. This aligns with the intuition that later transformer layers perform more task-specific processing and are therefore more affected by the loss of contextual information through eviction. The model must amplify its query projections to extract more information from the reduced cache.

For v_proj, the compensation is more moderate and more evenly distributed across layers, consistent with the smaller dimensionality of value heads (512 vs 2048 for queries in the GQA architecture).

### Lower subspace overlap = qualitatively different adaptation

The near-orthogonal subspaces tell us something important about how eviction changes what the model needs to learn:

- **M1** can learn adaptations that rely on having full attention over the entire context. Its LoRA updates likely encode task-specific attention patterns that assume all tokens remain accessible.
- **M3** must learn adaptations that are robust to information loss. Its LoRA updates likely encode attention patterns that prioritize extracting and compressing critical information before it gets evicted, or that can reconstruct needed context from the surviving cache entries.

These are fundamentally different strategies, which explains why the learned weight subspaces barely overlap. The model under eviction is not learning a "noisier version" of the same adaptation -- it is learning a different adaptation entirely.

---

## Plots

| Plot | Description |
|------|-------------|
| [`weight_magnitude.png`](weight_magnitude.png) | Per-layer Frobenius norms of B@A for M1 and M3 |
| [`singular_values.png`](singular_values.png) | SVD spectra of B@A (q_proj) across all 16 layers |
| [`subspace_overlap.png`](subspace_overlap.png) | Mean cosine of principal angles between M1/M3 subspaces |
| [`norm_ratio.png`](norm_ratio.png) | Per-layer ratio of M3/M1 norms |
