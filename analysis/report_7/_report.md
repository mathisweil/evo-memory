# Analysis 7 -- CKA Representation Similarity

> **TL;DR:** All three model pairs (M1 vs M2, M1 vs M3, M2 vs M3) have
> CKA > 0.995 at every layer — the representations are nearly identical.
> M1's LoRA diverges most at early layers (L2), while M3's diverges most
> at later layers (L9-L11), near the eviction hotspot.  The two LoRAs
> change different parts of the network but neither moves far from the
> base model.

---

## Setup

Prompts: 365 samples (73 per task, balanced across all 5 tasks, drawn
from train+val+test splits).  Seq_len 4303-6464 tokens.  All comparisons
use full-context forward passes (no NAMM eviction at inference).

- **M1:** LoRA fine-tuned, full context (val F1 45.48)
- **M2:** Base model, no LoRA (no fine-tuning)
- **M3:** LoRA + frozen NAMM, best checkpoint (step 260, val F1 52.06)

Linear CKA measures representation similarity: 1.0 = identical geometry,
0.0 = unrelated.  Computed from mean-pooled hidden states across samples.

---

## Findings

### Three-way CKA comparison

| Pair     | Meaning                 | Mean CKA | Min CKA | Min layer |
| -------- | ----------------------- | -------: | ------: | --------: |
| M1 vs M2 | M1 LoRA effect          |   0.9988 |  0.9956 |        L2 |
| M2 vs M3 | M3 LoRA effect          |   0.9992 |  0.9984 |        L9 |
| M1 vs M3 | Between the two LoRAs   |   0.9986 |  0.9977 |       L11 |

All pairs exceed 0.995 everywhere.  The differences are tiny but reveal
where each LoRA acts:

**M1's LoRA** (M1 vs M2, orange in the plot) diverges most at **L2** —
an early layer.  M1 modifies how the model processes input
representations early in the network.

**M3's LoRA** (M2 vs M3, green) diverges most at **L9** — a deeper
layer near the eviction hotspot (layers 8-9 per Report 3).  M3's
adaptation targets the layers downstream of the heaviest eviction.

**Between the two LoRAs** (M1 vs M3, blue) the maximum divergence is at
**L11**, reflecting the cumulative effect of M1 changing early layers
and M3 changing later layers.

### No computational shifts

The cross-layer CKA heatmap shows the diagonal is always the best match
— M3 layer 9 still does roughly what M1 layer 9 does.  There is no
evidence that M3 has reorganised which layer performs what computation.

---

## Discussion

### Different LoRAs, different network depths

M1 and M3 have near-orthogonal LoRA subspaces (Report 4, overlap ~0.21)
and dramatically different weight norms (M3 is 1.42x larger).  Yet both
produce representations within 0.5% of the base model at every layer.
The LoRA perturbations are large in weight space but tiny in
representation space.

The key structural difference is *where* each LoRA acts.  M1 adapts
early (L2), M3 adapts late (L9).  This is consistent with Report 3's
finding that layers 8-9 perform the most aggressive eviction — M3's
LoRA concentrates its representational changes near the eviction site.

### Connection to other reports

| Report          | Finding                                  | Implication                                     |
| --------------- | ---------------------------------------- | ----------------------------------------------- |
| 4 (LoRA wts)    | Norms 1.42x, orthogonal subspaces        | Large weight diffs but small representation diffs |
| 5 (Attention)   | M2 ≈ M3 entropy; LoRA changes values     | CKA confirms: LoRA barely moves representations  |
| 3 (Retention)   | L8-9 most aggressive eviction             | M3 CKA dip at L9 aligns with eviction hotspot    |

---

## Plots

| Plot                                          | Description                                  |
| --------------------------------------------- | -------------------------------------------- |
| [`cka_by_layer.png`](plots/cka_by_layer.png)  | Per-layer CKA: M1 vs M2, M2 vs M3, M1 vs M3 |
| [`cka_heatmap.png`](plots/cka_heatmap.png)    | Cross-layer CKA heatmap (M1 vs M3)           |
