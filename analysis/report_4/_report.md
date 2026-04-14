# Analysis 4 -- LoRA Weight Comparison (M1 vs M3)

> **TL;DR:** M3's LoRA adapters are larger than M1's -- 1.42x for q_proj
> and 1.16x for v_proj -- but the two adaptations are near-orthogonal
> (subspace overlap ~0.21 for q_proj, ~0.18 for v_proj). Eviction-aware
> training pushes the optimizer into a fundamentally different region of
> LoRA parameter space: the model compensates for information loss from
> eviction by growing its adapter norms while learning in a qualitatively
> different subspace from full-context fine-tuning.

---

## Setup

All comparisons use **full-context inputs** (no NAMM eviction at inference). Prompts: 10 test samples at 1024 tokens from LongBench tasks.

- **M1:** LoRA fine-tuned, full context, no eviction (baseline; best val F1 45.48)
- **M3:** LoRA + frozen NAMM, best checkpoint (step 260, val F1 52.06, ~43% through training; WandB `h0bzg6on`)

---

## Findings

### 1. M3 LoRA norms are larger than M1

| Projection | Mean M3/M1 ratio | Std    |
| ---------- | ---------------: | -----: |
| q_proj     |           1.4201 | 0.1760 |
| v_proj     |           1.1630 | 0.1235 |

Both ratios are strictly > 1.0 across all layers. Eviction-aware training produces larger adapters than full-context training: the model compensates for the information loss from NAMM eviction by increasing its LoRA weight magnitudes, particularly in the query projection.

### 2. Subspace overlap is near-orthogonal

Mean cosine of principal angles between M1 and M3 LoRA column spaces (1.0 = identical, 0.0 = orthogonal):

| Projection | Mean overlap | Std    |
| ---------- | -----------: | -----: |
| q_proj     |       0.2057 | 0.0330 |
| v_proj     |       0.1812 | 0.0333 |

Both values sit firmly in the near-orthogonal regime. M3 does not simply "scale up" the M1 adaptation -- it learns in a different subspace entirely.

### 3. Larger norms + orthogonal subspace = qualitatively different adaptation

The combination of elevated norms and low overlap indicates that eviction-aware training does not merely intensify the same adaptation that full-context training finds. Instead, the optimizer discovers a distinct strategy for coping with the possibility that tokens will be evicted. This is consistent with the attention entropy findings (Report 5), where M3 shows a broader, more hedged attention pattern rather than a sharpened version of M1's.

---

## Discussion

### Why does eviction-aware training produce larger adapters?

During M3 training, the frozen NAMM evicts a fraction of KV cache entries. The LoRA adapter must compensate for this information loss by learning to extract sufficient signal from the retained entries. This requires larger perturbations to the base model's attention and value projections -- hence larger Frobenius norms.

The q_proj ratio (1.42x) exceeds the v_proj ratio (1.16x), suggesting that most of the compensation occurs in the query projection: the model reshapes *what it looks for* more than *what it outputs* per token.

### Note on checkpoint maturity

The M3 checkpoint is from step 260 (~43% through training). Despite being early-stopped, M3's val F1 (52.06) substantially exceeds M1 (45.48), suggesting the adapter is efficient rather than undertrained. If anything, the norm ratios might increase slightly with further training, though the subspace direction is unlikely to shift given how orthogonal the adaptation already is.

---

## Comparison with buggy runs (historical context)

Prior to the attention mask bug fix, M3-buggy (step 600, val F1 45.59) showed substantially larger norm ratios (q_proj 1.93x, v_proj 1.50x) and marginally lower subspace overlap (q_proj 0.192, v_proj 0.174). The inflated norms reflected the adapter compensating for *both* eviction and mask-induced noise; fixing the mask removed the second source of degradation, reducing the compensation burden by roughly a quarter. The near-orthogonal subspace finding is robust to the bug fix.

---

## Plots

| Plot                                                  | Description                             |
| ----------------------------------------------------- | --------------------------------------- |
| [`weight_magnitude.png`](plots/weight_magnitude.png) | Per-layer Frobenius norms (M1 vs M3)    |
| [`singular_values.png`](plots/singular_values.png) | SVD spectra of B@A q_proj (M1 vs M3)   |
| [`subspace_overlap.png`](plots/subspace_overlap.png) | Subspace overlap with M1 per layer      |
| [`norm_ratio.png`](plots/norm_ratio.png)    | Per-layer M3/M1 norm ratios             |
