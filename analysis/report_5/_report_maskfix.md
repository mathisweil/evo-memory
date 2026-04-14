# Analysis 5 (Maskfix): Attention Entropy Shift -- Maskfix vs Buggy

> **TL;DR:** M3-maskfix and M3-buggy produce **nearly identical attention patterns** on full-context inputs: both show +5.0% higher entropy and -2.5% lower sink fractions relative to M1. The "pre-emptive hedging" strategy -- broader, less sink-dependent attention -- is **not an artefact of the attention mask bug**. It is a genuine adaptation to eviction-aware training that emerges regardless of whether the mask is correct.

---

## Setup

All comparisons use **full-context inputs** (no NAMM eviction at inference). Prompts: 10 test samples at 1024 tokens from LongBench tasks.

- **M1:** LoRA fine-tuned, full context, no eviction (baseline)
- **M3-maskfix:** LoRA + frozen NAMM, attention mask bug fixed, best checkpoint (step 260, val F1 52.06, ~43% through training)
- **M3-buggy:** LoRA + frozen NAMM, original buggy attention mask (step 600, val F1 45.59, end of training)

For each model, we computed per-head per-layer attention entropy (H = -sum(a_i log a_i)) and attention sink fraction (mass on first 5 tokens), averaged over query positions and samples.

---

## Findings

### Attention Entropy

| Model       | Mean entropy (nats) | Change vs M1 |
| ----------- | ------------------: | ------------- |
| M1          |              2.1494 | --            |
| M3-maskfix  |              2.2561 | +5.0%         |
| M3-buggy    |              2.2606 | +5.2%         |

M3-maskfix and M3-buggy are within 0.0045 nats of each other -- effectively indistinguishable. Both show the same entropy elevation relative to M1.

### Attention Sinks

| Model       | Mean sink fraction | Change vs M1 |
| ----------- | -----------------: | ------------- |
| M1          |             0.5278 | --            |
| M3-maskfix  |             0.5135 | -2.7%         |
| M3-buggy    |             0.5150 | -2.4%         |

Again, maskfix and buggy are nearly identical (delta = 0.0015). Both reduce attention mass on sink tokens by a similar margin.

---

## Discussion

### The hedging strategy is real, not a mask artefact

The original Report 5 found that M3-buggy distributes attention more broadly than M1 and interpreted this as "pre-emptive hedging" -- the model learns to attend to more tokens as insurance against eviction. A natural concern was that the buggy attention mask might itself cause broader attention, and the hedging would disappear once the mask is fixed.

This report rules out that concern. M3-maskfix shows the same entropy shift (+5.0% vs +5.2%) and the same sink reduction (-2.7% vs -2.4%). The pre-emptive hedging strategy is a genuine adaptation to eviction-aware training.

### Why are the two M3 variants so similar?

Despite having different attention masks during training and substantially different LoRA norms (Report 4 maskfix: 1.42x vs buggy 1.93x for q_proj), both M3 variants converge to nearly the same attention behaviour on full-context inputs. This suggests that the hedging pattern is a robust attractor in the eviction-aware training landscape: the model finds this strategy regardless of the precise training dynamics.

The LoRA norms differ because the buggy variant must compensate for mask-induced noise in addition to eviction, but the extra compensation does not change the qualitative attention pattern -- it just requires more weight magnitude to achieve a similar functional outcome.

### Comparison with original Report 5

The original report compared M1 (entropy 1.912, sinks 0.574) with M3-buggy. The numbers here differ slightly (M1 entropy 2.149, sinks 0.528) because the maskfix analysis used a different set of 10 test prompts. The relative patterns are consistent: M3 has higher entropy and lower sinks regardless of the specific samples.

---

## Key Takeaway

The attention mask bug affected the *magnitude* of LoRA compensation (Report 4) and likely the training efficiency (maskfix reaches 52.06 F1 at 43% training vs buggy's 45.59 F1 at 100%), but it did **not** change the qualitative nature of the learned attention strategy. The pre-emptive hedging interpretation stands.

---

## Plots

| Plot                                                         | Description                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| [`attention_entropy_maskfix.png`](attention_entropy_maskfix.png) | Per-layer attention entropy for M1, M3-maskfix, and M3-buggy  |
| [`entropy_heatmap_maskfix.png`](entropy_heatmap_maskfix.png)     | Layer x head entropy heatmaps, three-way comparison           |
| [`entropy_diff_maskfix.png`](entropy_diff_maskfix.png)           | Entropy difference heatmaps: maskfix-M1 vs buggy-M1 side-by-side |
