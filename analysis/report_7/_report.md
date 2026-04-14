# Analysis 7 -- CKA Representation Similarity

> **TL;DR:** M3 representations remain very close to M1 in CKA space (mean 0.995, min 0.990 at layer 10). The LoRA perturbation is small -- CKA exceeds 0.99 at every layer -- confirming that the eviction-aware adapter operates within the same representational neighbourhood as the full-context baseline. The point of maximum divergence at layer 10 suggests the adaptation concentrates in later, semantic-integration layers rather than early positional/syntactic ones. This is consistent with the smaller LoRA norms found in Report 4.

---

## Setup

All comparisons use **full-context inputs** (no NAMM eviction at inference). Prompts: 10 test samples at 1024 tokens from LongBench tasks.

- **M1:** LoRA fine-tuned, full context, no eviction (baseline)
- **M3:** LoRA + frozen NAMM, best checkpoint (step 260, val F1 52.06, ~43% through training)

Linear CKA measures representation similarity:
```
CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```
CKA = 1.0 means identical representations (up to linear transform); CKA = 0.0 means unrelated.

---

## Findings

### Layer-wise CKA: M3 vs M1

| Metric        | Value    |
| ------------- | -------: |
| Mean CKA      |   0.9952 |
| Min CKA       |   0.9901 |
| Min CKA layer | layer 10 |
| Max CKA       |   1.0000 |
| Max CKA layer |  layer 0 |

CKA starts at 1.0 at the embedding/early layers (shared tokenizer and minimal LoRA effect) and dips slightly at layer 10, the point of maximum divergence.

### The Divergence at Layer 10 Is Mild

The minimum CKA of 0.990 is well within the range typical of LoRA-adapted models of the same architecture (>0.95). With rank-8 LoRA in a 2048-dimensional space, the adapter produces at most a ~1% representational shift at its most divergent layer. The model computes slightly different things at layer 10 but stays firmly in the same representational neighbourhood as M1.

### Divergence Concentrates in Later Layers

Layer 10 is involved in higher-level semantic integration. The adaptation being concentrated here rather than in early layers suggests the LoRA adapter is performing task-specific processing under eviction constraints -- a natural locus for adapting to missing context, as opposed to low-level positional or syntactic corrections.

### Consistency with Report 4 (LoRA Weights)

The high CKA (mean 0.995) is consistent with Report 4's finding that M3 LoRA norms are moderate (1.42x base for q_proj). Smaller weight perturbations produce representations closer to the baseline model. The adapter achieves strong task performance (val F1 52.06, surpassing M1's 45.48) with very little representational departure from M1.

---

## Comparison with Buggy Runs (Historical Context)

Prior to fixing the attention mask bug, M3-buggy (step 600, val F1 45.59) showed mean CKA 0.992 and minimum CKA 0.979 at layer 3. Comparing with the corrected results:

| Metric        | M3 (corrected) | M3-buggy |
| ------------- | -------------: | -------: |
| Mean CKA      |         0.9952 |   0.9921 |
| Min CKA       |         0.9901 |   0.9788 |
| Min CKA layer |       layer 10 |  layer 3 |

The shift in divergence point from layer 3 to layer 10 is notable. The buggy mask forced the model to make early, aggressive corrections to compensate for corrupted attention patterns, pushing divergence to layer 3 (CKA 0.979). With correct masking, the adaptation defers to layer 10 where it serves a more natural semantic-integration purpose, and the divergence is less severe (0.990 vs 0.979). Both variants maintained very high CKA (>0.97 everywhere), confirming these are quantitative rather than qualitative differences.

---

## Cross-Report Connections

| Report            | Measures              | Finding                                            |
| ----------------- | --------------------- | -------------------------------------------------- |
| 4 (LoRA wts)      | Weight-space diff     | Norms 1.42x base; overlap marginally above chance  |
| 5 (Attention)      | Attention patterns    | Identical entropy/sinks; hedging not mask artefact |
| 7 (CKA, this)     | Repr. similarity      | M3 close to M1 (mean 0.995); diverge at layer 10  |

A coherent picture: the eviction-aware adapter achieves strong performance (val F1 52.06 vs M1 45.48) with moderate LoRA norms, near-identical attention patterns, and high representational similarity to M1. The adaptation is efficient -- it changes just enough in later layers to handle eviction constraints without distorting the base model's representations.

---

## Note on Checkpoint Maturity

The M3 checkpoint (step 260, ~43% through training) has not completed training. The representational profile may evolve as training continues -- the divergence point at layer 10 could shift or deepen. The current results should be interpreted as a snapshot of the best-performing checkpoint available.

---

## Plots

| Plot                                   | Description                     |
| -------------------------------------- | ------------------------------- |
| [`cka_by_layer.png`](cka_by_layer.png) | Layer-wise CKA, M1 vs M3       |
| [`cka_heatmap.png`](cka_heatmap.png)   | Cross-layer CKA heatmap, M3 vs M1 |
