# Analysis 4 (Maskfix): LoRA Weight Comparison (M1 vs M3-maskfix vs M3-buggy)

> **TL;DR:** With the attention mask bug fixed, M3's LoRA adapters are **smaller** than the buggy variant (1.42x vs 1.93x for q_proj, 1.16x vs 1.50x for v_proj) but still larger than M1. Subspace overlap with M1 is marginally higher for maskfix (~0.21 for q_proj vs ~0.19 buggy), though both remain near-orthogonal. The correct attention mask reduces the compensation burden on the LoRA -- the adapter no longer needs to work as hard to overcome degraded KV entries -- but the "qualitatively different adaptation" finding from the original report holds regardless of the bug.

---

## Setup

All comparisons use **full-context inputs** (no NAMM eviction at inference). Prompts: 10 test samples at 1024 tokens from LongBench tasks.

- **M1:** LoRA fine-tuned, full context, no eviction (baseline)
- **M3-maskfix:** LoRA + frozen NAMM, attention mask bug fixed, best checkpoint (step 260, val F1 52.06, ~43% through training)
- **M3-buggy:** LoRA + frozen NAMM, original buggy attention mask (step 600, val F1 45.59, end of training)

---

## Summary of Findings

1. **Maskfix LoRA norms are smaller than buggy.** With correct attention masking, the LoRA adapter compensates less aggressively: q_proj mean ratio drops from 1.93x (buggy) to 1.42x (maskfix), and v_proj from 1.50x to 1.16x. The corrected mask gives the model clean KV cache entries, reducing the need for large weight perturbations to work around corrupted attention patterns.

2. **Subspace overlap is slightly higher for maskfix.** q_proj overlap rises from 0.192 (buggy) to 0.206 (maskfix); v_proj from 0.174 to 0.181. The maskfix LoRA is marginally more aligned with M1's adaptation direction, though both remain overwhelmingly near-orthogonal.

3. **Both are still near-orthogonal (~0.18--0.21).** The "different adaptation" conclusion from the original report is robust to the bug fix. Eviction-aware training pushes the optimizer into a fundamentally different region of LoRA parameter space regardless of whether the attention mask is correct.

---

## Norm Ratio (M3/M1) Comparison

### q_proj

| Variant   | Mean ratio | Std   |
| --------- | ---------: | ----: |
| M3-maskfix |     1.4201 | 0.1760 |
| M3-buggy   |     1.9277 | 0.3506 |

### v_proj

| Variant   | Mean ratio | Std   |
| --------- | ---------: | ----: |
| M3-maskfix |     1.1630 | 0.1235 |
| M3-buggy   |     1.4960 | 0.1929 |

The maskfix adapter is **26% smaller** (q_proj) and **22% smaller** (v_proj) than the buggy adapter in terms of M3/M1 norm ratio. Both are still strictly > 1.0 across all layers, confirming that eviction-aware training produces larger adapters than full-context training even with correct masking.

---

## Subspace Overlap Comparison

Mean cosine of principal angles between M1 and M3 LoRA column spaces (1.0 = identical, 0.0 = orthogonal):

### q_proj

| Variant    | Mean overlap | Std    |
| ---------- | -----------: | -----: |
| M3-maskfix |       0.2057 | 0.0330 |
| M3-buggy   |       0.1918 | 0.0242 |

### v_proj

| Variant    | Mean overlap | Std    |
| ---------- | -----------: | -----: |
| M3-maskfix |       0.1812 | 0.0333 |
| M3-buggy   |       0.1738 | 0.0282 |

The differences are small (+0.014 for q_proj, +0.007 for v_proj). Both variants sit firmly in the "near-orthogonal" regime. The bug does not meaningfully change the conclusion that M1 and M3 learn in different subspaces.

---

## Discussion

### Why are maskfix norms smaller?

With the buggy attention mask, corrupted attention patterns produce degraded KV cache entries during training. The LoRA adapter must compensate for two things simultaneously: (1) the information loss from eviction, and (2) the additional noise from incorrect masking. Fixing the mask removes the second source of degradation, leaving only the genuine eviction effect, which requires less compensation.

The reduction is substantial -- roughly a quarter less compensation -- suggesting that a meaningful fraction of the buggy adapter's magnitude was spent correcting mask-induced artefacts rather than adapting to eviction per se.

### Why is subspace overlap marginally higher?

With correct masking, the M3 training signal is cleaner: the model learns to adapt purely to eviction rather than to eviction-plus-mask-noise. This purer signal produces an adaptation that is slightly more aligned with M1 (which also uses correct masking), but the alignment increase is modest. The dominant effect -- near-orthogonal subspaces -- persists because eviction itself, not the mask bug, is the primary driver of subspace divergence.

### Note on checkpoint maturity

The maskfix checkpoint is from step 260 (~43% through training), while the buggy checkpoint is from step 600 (end of training). The smaller norms for maskfix could partially reflect less training, though the maskfix val F1 (52.06) substantially exceeds the buggy val F1 (45.59), suggesting the maskfix adapter is more efficient rather than simply less trained.

---

## Plots

| Plot                                                           | Description                                                      |
| -------------------------------------------------------------- | ---------------------------------------------------------------- |
| [`weight_magnitude_maskfix.png`](weight_magnitude_maskfix.png) | Per-layer Frobenius norms of B@A for M1, M3-maskfix, and M3-buggy |
| [`singular_values_maskfix.png`](singular_values_maskfix.png)   | SVD spectra of B@A (q_proj) comparing maskfix vs buggy            |
| [`subspace_overlap_maskfix.png`](subspace_overlap_maskfix.png) | Subspace overlap with M1 for maskfix vs buggy                     |
| [`norm_ratio_maskfix.png`](norm_ratio_maskfix.png)             | Per-layer M3/M1 norm ratios, maskfix vs buggy side-by-side        |
