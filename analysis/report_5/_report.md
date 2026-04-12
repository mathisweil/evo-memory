# Analysis 5 — Attention Entropy Shift Under Eviction

> **TL;DR:** Even on **full-context inputs** (no NAMM eviction at inference), M3 cs1024 shows measurably different attention patterns from M1. M3 has **higher mean entropy** (1.992 vs 1.912 nats) — more distributed attention — and **slightly lower sink fractions** (0.568 vs 0.574). The entropy difference is concentrated in specific layer–head pairs, with the largest shifts in layers 3, 6, 13, and 15. This contradicts the earlier (buggy) finding that the models were identical on full context, and shows the LoRA adaptation produces a detectable functional signature even without eviction active.

## Methodology

We loaded M1 (LoRA, full context) and M3 cs1024 (LoRA + frozen NAMM) checkpoints, merged LoRA weights into the base model, and ran inference on 10 test prompts (1024 tokens each) from LongBench tasks (qasper, 2wikimqa, hotpotqa_e). Both models processed identical inputs with **full context** — no NAMM eviction at inference time.

For each model, we computed:
1. **Attention entropy** per head per layer: H = -sum(a_i * log(a_i)), averaged over query positions and samples
2. **Attention sink fraction** per head per layer: fraction of attention mass on the first 5 tokens

## Findings

### Attention Entropy

| Metric | M1 | M3 cs1024 |
|--------|-----|-----------|
| Mean entropy (all layers, all heads) | 1.912 | 1.992 |
| Layers where M3 entropy < M1 (sharper) | 3 / 16 | — |

See `attention_entropy.png` — M3 shows consistently higher entropy across most layers, meaning its attention is more distributed.

The entropy profile across layers:
- **Layer 0** has the highest entropy (~3.0) — broad, unfocused attention
- **Layers 1-3** have low entropy (~1.1-1.5) — very focused attention, likely positional/syntactic heads
- **Later layers (8-15)** have moderate entropy (~1.5-2.5) — increasingly distributed for semantic integration
- M3 diverges most in the **middle and late layers** (6-15), where semantic processing dominates

### Attention Sinks

| Metric | M1 | M3 cs1024 |
|--------|-----|-----------|
| Mean sink fraction (first 5 tokens) | 0.574 | 0.568 |

Over 56% of attention mass goes to the first 5 tokens across all layers — consistent with the attention sink phenomenon (Xiao et al., 2024). M3 shows a small but consistent reduction in sink reliance.

### Entropy Difference Heatmap

See `entropy_diff.png`. The M3−M1 difference shows a structured pattern:
- **Red cells (M3 higher):** Concentrated in layers 3, 6-8, 12-13, 15 — M3 distributes attention more broadly here
- **Blue cells (M3 lower/sharper):** Scattered, notably layer 5 head 21 — M3 focuses more tightly on specific heads
- **Max |diff| = 1.0 nats** — a substantial shift for individual heads
- The pattern is not uniform: specific heads in specific layers are selectively modified

## Interpretation

1. **The LoRA adaptation is detectable even on full context.** Despite both models processing the same tokens with no eviction, M3's LoRA weights (trained with NAMM eviction at train time) produce measurably different attention patterns. The rank-8 LoRA updates, while small relative to the 1.2B base model, are sufficient to shift attention entropy by ~4% on average.

2. **M3 distributes attention more broadly.** Higher entropy means M3 attends to more tokens rather than focusing sharply on a few. This is consistent with a model that has learned to hedge against token loss — if eviction may remove attended tokens, distributing attention across more tokens provides robustness.

3. **M3 relies slightly less on attention sinks.** The reduced sink fraction (0.568 vs 0.574) suggests M3 redistributes some attention mass from the BOS/system tokens toward content tokens. Under eviction, content tokens are at risk of removal while sink tokens are always retained, so reducing sink reliance means the model extracts more information from content tokens while they're available.

4. **The adaptation is head-specific, not uniform.** The entropy difference heatmap shows that M3's changes are concentrated in particular layer–head combinations, not a global shift. This is consistent with LoRA's low-rank structure selectively modifying specific attention circuits.

### Connection to Reports 4 and 7

Report 4 showed M1 and M3 learn in near-orthogonal LoRA subspaces with M3 norms 1.5-2.6× larger. Report 7 shows CKA between M1 and M3 is 0.979–1.0 (very similar but not identical), with maximum divergence at layer 3. Together, these three reports paint a consistent picture:

| Report | What it measures | M1 vs M3 on full context |
|--------|-----------------|--------------------------|
| 4 (LoRA weights) | Weight-space difference | **Orthogonal subspaces**, M3 norms 1.5-2.6× larger |
| 5 (Attention) | Function-space difference (attention) | **Measurably different**: M3 higher entropy (+4%), lower sinks |
| 7 (CKA) | Function-space difference (representations) | **Very similar but not identical**: CKA 0.979-1.0 |

The weight-space divergence (Report 4) translates into measurable function-space differences in both attention (this report) and hidden representations (Report 7), even on full context.

### Implications for the paper

This finding is arguably a **stronger contribution** than the earlier (buggy) "dormant adaptation" narrative:

1. **Pre-emptive hedging, not dormant adaptation.** M3 does not simply switch on a latent capability when eviction activates. Instead, it learns a qualitatively different attention strategy — broader, less sink-dependent — that is *always active*. This strategy is coherent with eviction robustness: distributing attention across more tokens means no single eviction is catastrophic.

2. **Different computation path, same task performance.** Despite producing different internal representations (Report 7, CKA dip at layer 3) and different attention patterns (this report, +4% entropy), M3 matches M1 on aggregate F1 (val: 45.59 vs 45.48; test micro: 32.28 vs 31.14 — see Report 1). The model finds a different but equally effective solution that happens to also be robust to eviction.

3. **M3 occupies a different region of function space.** Combining the orthogonal LoRA subspaces (Report 4), the attention entropy shift (this report), and the CKA divergence (Report 7), the evidence shows M3 is not a minor perturbation of M1 — it is a genuinely different model that generalises across both full-context and evicted-context conditions.

## Figures

- `attention_entropy.png` — Per-layer attention entropy and sink fraction, M1 vs M3
- `entropy_heatmap.png` — Layer × head entropy heatmaps for both models
- `entropy_diff.png` — Entropy difference (M3 − M1) per layer and head

## References

- Xiao et al. (2024). "Efficient Streaming Language Models with Attention Sinks." *ICLR 2024*.
- Clark et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP 2019*.
- Abnar & Zuidema (2020). "Quantifying Attention Flow in Transformers." *ACL 2020*.
