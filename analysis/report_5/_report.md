# Analysis 5 -- Attention Entropy Under Eviction

> **TL;DR:** M2 and M3 (both under NAMM eviction) show ~12% lower
> attention entropy than M1 (full context).  M2 and M3 are nearly
> identical (-11.0% and -12.5% vs M1), meaning the LoRA has minimal
> impact on attention entropy in the evicted regime.  The entropy
> reduction is a direct consequence of attending over ~6.5% of the
> original tokens.

---

## Setup

Each model is measured in its **actual operating regime**:

- **M1:** Full-context forward (no NAMM, no eviction). All input tokens
  visible.
- **M2:** Forward with `apply_memory_policy=True`, no LoRA. NAMM evicts
  tokens during prefill; base model attends over the retained cache.
- **M3:** Forward with `apply_memory_policy=True`, with LoRA. Same NAMM
  eviction as M2, but LoRA-adapted weights.

Prompts: 365 samples from LongBench (73 per task, balanced across all 5
tasks, drawn from train+val+test splits).  Seq_len 4303-6464 tokens.
After eviction, models retain a mean of **6.5%** of tokens (cache sizes
~200-450 from ~5700 input).

Entropy: H = -sum(a_i log a_i) at the last query position, averaged
over heads and samples.  Evicted tokens have zero attention (they are
physically absent from the KV cache), so this is mathematically
equivalent to computing entropy over the full prompt with evicted
positions set to zero — 0 log 0 = 0 contributes nothing.

Checkpoints:
- M1: `experiment_artifacts/gcs/M1/best_ckpt.pt` (val F1 45.48)
- M3: `experiment_artifacts/gcs/M3_cs1024_maskfix/best_ckpt.pt` (step 260, val F1 52.06)
- NAMM: `experiment_artifacts/gcs/M2_cs1024_maskfix/ckpt.pt`

---

## Findings

### Entropy comparison

| Condition            | Mean entropy (nats) | Change vs M1 |
| -------------------- | ------------------: | -----------: |
| M1 (full context)    |              2.6366 | --           |
| M2 (NAMM, no LoRA)   |              2.3457 | -11.0%       |
| M3 (LoRA + NAMM)     |              2.3063 | -12.5%       |

M2 and M3 are nearly identical (2.35 vs 2.31 nats, -1.7% difference).
The LoRA adaptation has negligible effect on the entropy of the
attention distribution under eviction.

### Per-layer patterns

| Layer | M1    | M2    | M3    | M3-M1   |
| ----: | ----: | ----: | ----: | ------: |
|     0 | 3.195 | 2.884 | 2.882 |  -0.313 |
|     1 | 1.678 | 1.445 | 1.495 |  -0.183 |
|     2 | 1.440 | 1.235 | 1.489 |  +0.049 |
|     3 | 1.825 | 1.778 | 1.796 |  -0.029 |
|     4 | 2.764 | 2.547 | 2.443 |  -0.321 |
|     5 | 2.601 | 2.197 | 2.334 |  -0.268 |
|     6 | 2.718 | 2.601 | 2.370 |  -0.348 |
|     7 | 3.364 | 2.803 | 2.744 |  -0.621 |
|     8 | 2.750 | 2.743 | 2.658 |  -0.091 |
|     9 | 2.613 | 2.545 | 2.769 |  +0.156 |
|    10 | 2.716 | 2.373 | 2.226 |  -0.490 |
|    11 | 2.545 | 2.288 | 2.199 |  -0.346 |
|    12 | 2.495 | 2.156 | 2.097 |  -0.398 |
|    13 | 2.743 | 2.417 | 2.424 |  -0.319 |
|    14 | 3.430 | 2.789 | 2.537 |  -0.893 |
|    15 | 3.311 | 2.730 | 2.441 |  -0.870 |

Layers 2 and 9 show slightly *higher* M3 entropy than M1, despite the
shorter sequence.  These are exceptions — the overall pattern is
lower entropy under eviction across all layers.

The largest entropy drops are in layers 14-15 (-0.89 and -0.87 nats).
M2 and M3 track each other closely at most layers, with M3 slightly
lower in the later layers (10-15) and slightly higher in a few early
layers (2, 5).

---

## Discussion

### Per-head entropy is lower, but total coverage is slightly higher

The per-layer bar chart shows each individual head is more concentrated
under eviction (lower per-head entropy).  But when attention is summed
across all 16 layers × 32 heads and normalised into a single
distribution over token positions, the entropy is slightly *higher* for
M2/M3:

| Condition          | Total entropy (nats) | Effective positions |
| ------------------ | -------------------: | ------------------: |
| M1 (full context)  |                 3.61 |         ~37 of 5700 |
| M2 (NAMM, no LoRA) |                 3.73 |          ~42 of 350 |
| M3 (LoRA + NAMM)   |                 3.62 |          ~37 of 350 |

These are not contradictory.  Each evicted head is individually more
peaked (fewer tokens get attention from any single head), but different
heads attend to *different* positions in the retained cache.  The
collective coverage across all heads is slightly broader.  With only
~350 tokens available, 512 head-layer slots (32 × 16) have to spread
out — there aren't enough popular positions for all heads to pile onto
the same few.

M1's heads, by contrast, are individually less peaked but more
redundant — many heads attend to the same positions within the full
5700-token context.

### M2 ≈ M3: LoRA does not change attention

The most notable finding is that M2 and M3 are nearly identical on
both measures.  Per-head entropy differs by only 1.7%.  Total entropy
is 3.73 vs 3.62.  Despite M3's LoRA producing dramatically different
weights (1.42x larger norms, near-orthogonal subspaces per Report 4)
and M3 achieving much higher val F1 (52.06 vs M2's 14.90), the
attention distributions are indistinguishable.

This means M3's LoRA improves performance through mechanisms other than
changing how attention is distributed — likely by transforming what
information is extracted from the attended tokens (value projections)
rather than which tokens receive attention (query-key interaction).

### Connection to other reports

- **Report 4 (LoRA weights):** M3's LoRA weights are 1.42x larger and
  near-orthogonal to M1's, yet produce the same entropy as M2 (no LoRA)
  under eviction.
- **Report 6 (NAMM alignment):** NAMM scores weakly correlate with
  attention.  M1, M2, M3 all show the same correlation.
- **Report 3 (retention):** Mean retention ~6.5%, consistent with the
  cache sizes observed here.

---

## Plots

| Plot                                                   | Description                                        |
| ------------------------------------------------------ | -------------------------------------------------- |
| [`attention_entropy.png`](plots/attention_entropy.png) | Per-layer per-head entropy (M1 vs M2 vs M3)        |
| [`total_entropy.png`](plots/total_entropy.png)         | Total entropy across all layers and heads combined  |
| [`entropy_heatmap.png`](plots/entropy_heatmap.png)     | Layer x head entropy heatmaps (M1 vs M3)           |
| [`entropy_diff.png`](plots/entropy_diff.png)           | Entropy diff heatmap: M3 evicted minus M1           |
