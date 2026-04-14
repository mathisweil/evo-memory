# Analysis 5 -- Attention Entropy Shift Under Eviction

> **TL;DR:** M3 distributes attention more broadly than M1: +5.0% higher
> entropy and -2.7% lower sink fractions. We interpret this as
> "pre-emptive hedging" -- the model learns to attend to more tokens as
> insurance against eviction, reducing its reliance on attention sinks
> that might be evicted. This pattern was also observed in the earlier
> buggy runs (with nearly identical magnitudes), confirming it is a
> robust adaptation to eviction-aware training rather than a training
> artefact.

---

## Setup

All comparisons use **full-context inputs** (no NAMM eviction at inference). Prompts: 10 test samples at 1024 tokens from LongBench tasks.

- **M1:** LoRA fine-tuned, full context, no eviction (baseline; best val F1 45.48)
- **M3:** LoRA + frozen NAMM, best checkpoint (step 260, val F1 52.06, ~43% through training; WandB `h0bzg6on`)

For each model, we computed per-head per-layer attention entropy (H = -sum(a_i log a_i)) and attention sink fraction (mass on first 5 tokens), averaged over query positions and samples.

---

## Findings

### Attention Entropy

| Model | Mean entropy (nats) | Change vs M1 |
| ----- | ------------------: | -----------: |
| M1    |              2.1494 | --           |
| M3    |              2.2561 | +5.0%        |

M3 produces consistently higher entropy across layers: its attention distributions are flatter, spreading probability mass over more tokens rather than concentrating on a few.

### Attention Sinks

| Model | Mean sink fraction | Change vs M1 |
| ----- | -----------------: | -----------: |
| M1    |             0.5278 | --           |
| M3    |             0.5135 | -2.7%        |

M3 places less attention mass on the first 5 tokens (attention sinks). Since sinks are typically the highest-priority tokens for retention, reducing sink dependence makes the model more robust to the eviction of mid-sequence tokens.

---

## Discussion

### Pre-emptive hedging as an eviction-aware strategy

The combination of higher entropy and lower sink fractions suggests a coherent adaptation: M3 has learned to "hedge" its attention, spreading mass more evenly so that no single token's eviction is catastrophic. Rather than sharpening attention toward the tokens NAMM retains (which would increase alignment with the eviction policy), M3 broadens attention as insurance against losing any particular token.

This is consistent with the Report 4 finding that M3's LoRA adaptation is near-orthogonal to M1's -- the model is not learning a stronger version of M1's strategy but a qualitatively different one.

### Connection to M3's performance advantage

M3's hedging strategy appears effective: despite using only ~43% of training steps, M3 (val F1 52.06) substantially outperforms M1 (val F1 45.48). By distributing attention more broadly, the model extracts information from a wider set of tokens, making it more resilient when NAMM evicts part of the KV cache.

### Layer-by-layer variation

The entropy increase is not uniform across layers. Some layers show larger shifts than others, reflecting the fact that different layers serve different roles in the transformer's information processing. See `attention_entropy.png` and `entropy_heatmap.png` for per-layer and per-head breakdowns.

---

## Robustness check: comparison with buggy runs

The earlier buggy M3 variant (step 600, val F1 45.59, trained with incorrect attention mask) showed nearly identical attention patterns: +5.2% entropy and -2.4% sink reduction vs M1. The two M3 variants are within 0.0045 nats of each other on entropy and 0.0015 on sink fraction -- effectively indistinguishable. This confirms that the hedging strategy is a robust attractor in the eviction-aware training landscape, emerging regardless of whether the attention mask is correct. The bug affected training efficiency (maskfix reaches higher F1 in fewer steps) and LoRA norm magnitudes (Report 4), but not the qualitative attention pattern.

---

## Plots

| Plot                                                             | Description                            |
| ---------------------------------------------------------------- | -------------------------------------- |
| [`attention_entropy.png`](attention_entropy.png) | Per-layer attention entropy (M1 vs M3) |
| [`entropy_heatmap.png`](entropy_heatmap.png)     | Layer x head entropy heatmaps          |
| [`entropy_diff.png`](entropy_diff.png)           | Entropy diff heatmap: M3 minus M1      |
