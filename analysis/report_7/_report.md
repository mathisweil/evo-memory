# Analysis 7 (Maskfix): CKA Representation Similarity -- Maskfix vs Buggy

> **TL;DR:** M3-maskfix representations are **closer to M1** than M3-buggy (mean CKA 0.995 vs 0.992). The point of maximum divergence shifts from **layer 3 (buggy, CKA 0.979)** to **layer 10 (maskfix, CKA 0.990)** -- the divergence moves deeper into the network and becomes less severe. Both variants maintain very high CKA (>0.97 everywhere), confirming the LoRA perturbation is small in representation space regardless of the bug. The maskfix adapter produces representations that are more M1-like, consistent with its smaller LoRA norms (Report 4 maskfix).

---

## Setup

All comparisons use **full-context inputs** (no NAMM eviction at inference). Prompts: 10 test samples at 1024 tokens from LongBench tasks.

- **M1:** LoRA fine-tuned, full context, no eviction (baseline)
- **M3-maskfix:** LoRA + frozen NAMM, attention mask bug fixed, best checkpoint (step 260, val F1 52.06, ~43% through training)
- **M3-buggy:** LoRA + frozen NAMM, original buggy attention mask (step 600, val F1 45.59, end of training)

Linear CKA measures representation similarity:
```
CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```
CKA = 1.0 means identical representations (up to linear transform); CKA = 0.0 means unrelated.

---

## Findings

### Layer-wise CKA vs M1

| Metric          | M3-maskfix | M3-buggy |
| --------------- | ---------: | -------: |
| Mean CKA        |     0.9952 |   0.9921 |
| Min CKA         |     0.9901 |   0.9788 |
| Min CKA layer   |   layer 10 |  layer 3 |
| Max CKA         |     1.0000 |   1.0000 |
| Max CKA layer   |    layer 0 |  layer 0 |

Both variants start at CKA = 1.0 at the embedding/early layers (shared tokenizer and minimal LoRA effect) and dip at different points deeper in the network.

### Side-by-side comparison

| Layer     | CKA (maskfix vs M1) | CKA (buggy vs M1) | Delta     |
| --------- | ------------------: | -----------------: | --------: |
| Layer 0   |              1.0000 |             1.0000 |    0.0000 |
| Layer 3   |                 --  |         **0.9788** |        -- |
| Layer 10  |          **0.9901** |                 -- |        -- |

(Full per-layer values available in the analysis scripts; the table above highlights the minimum-CKA layers for each variant.)

---

## Discussion

### Maskfix representations are more M1-like

The higher mean CKA (0.995 vs 0.992) is consistent with Report 4's finding that maskfix LoRA norms are smaller (1.42x vs 1.93x for q_proj). Smaller weight perturbations produce representations closer to the baseline model. The maskfix adapter achieves better task performance (val F1 52.06 vs 45.59) with less representational divergence from M1 -- a more efficient adaptation.

### The divergence point shifts from layer 3 to layer 10

This is a notable structural change:

- **M3-buggy** diverges most at **layer 3** (CKA 0.979) -- an early-middle layer where attention transitions from positional/syntactic to semantic processing. The original Report 7 interpreted this as a critical adaptation point where M3 redirects information flow.

- **M3-maskfix** diverges most at **layer 10** (CKA 0.990) -- a later layer involved in higher-level semantic integration. The divergence is also less severe (0.990 vs 0.979).

The shift suggests that the buggy mask forced the model to make early, aggressive corrections to compensate for corrupted attention patterns in the initial layers. With correct masking, the adaptation can be deferred to later layers where it serves a more natural purpose (task-specific processing under eviction constraints).

### The divergence at layer 10 is mild

The maskfix minimum CKA of 0.990 is substantially higher than the buggy minimum of 0.979. On the CKA scale, where fine-tuned models of the same architecture typically score >0.95, the difference between 0.990 and 0.979 is meaningful -- the maskfix model's representations are much closer to M1 at their point of maximum divergence.

### Both variants maintain very high CKA

Both exceed 0.97 at every layer. The LoRA perturbation (rank 8 in a 2048-dimensional space) produces at most a ~2% representational shift (buggy) or ~1% shift (maskfix). This confirms that the "different adaptation" finding from Report 4 (orthogonal LoRA subspaces) manifests as a small functional difference -- the models compute slightly different things but stay in the same representational neighbourhood.

---

## Connection to Other Maskfix Reports

| Report              | What it measures         | Maskfix vs Buggy                                                |
| ------------------- | ------------------------ | --------------------------------------------------------------- |
| 4 (LoRA weights)    | Weight-space difference  | Maskfix norms **26% smaller**; overlap marginally higher         |
| 5 (Attention)       | Attention patterns       | **Nearly identical** entropy/sinks; hedging is not a mask artefact |
| 7 (CKA, this report)| Representation similarity | Maskfix **closer to M1** (0.995 vs 0.992); divergence shifts deeper |

A coherent picture emerges:

1. **The mask bug inflated LoRA norms** (Report 4) and **pushed representational divergence to early layers** (this report), but **did not change the qualitative attention strategy** (Report 5).

2. **Maskfix is a more efficient adaptation.** It achieves higher task performance (52.06 vs 45.59 val F1) with smaller LoRA norms, closer representations to M1, and the same hedging attention pattern. The buggy adapter wasted capacity compensating for mask-induced noise.

3. **The core findings are robust to the bug fix.** Near-orthogonal LoRA subspaces, pre-emptive hedging in attention, and high-but-not-identical CKA all persist with correct masking. These are genuine properties of eviction-aware training, not artefacts.

---

## Note on Checkpoint Maturity

The maskfix checkpoint (step 260, ~43% through training) has not completed training. The representational profile may evolve as training continues -- the divergence point at layer 10 could shift or deepen. The current results should be interpreted as a snapshot of the best-performing maskfix checkpoint available, not a final comparison.

---

## Plots

| Plot                                                         | Description                                                     |
| ------------------------------------------------------------ | --------------------------------------------------------------- |
| [`cka_by_layer_maskfix.png`](cka_by_layer_maskfix.png)       | Layer-wise CKA vs M1 for maskfix and buggy, side-by-side        |
| [`cka_heatmap_maskfix.png`](cka_heatmap_maskfix.png)         | Cross-layer CKA heatmap for M3-maskfix vs M1                    |
| [`cka_heatmap_buggy_ref.png`](cka_heatmap_buggy_ref.png)     | Cross-layer CKA heatmap for M3-buggy vs M1 (reference)          |
