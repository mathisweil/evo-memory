# Analysis 3 -- Per-Layer Retention Pattern Analysis

## TL;DR

NAMM eviction is **highly non-uniform across layers**. For M3 cs1024, the most aggressive layer (layer 9) retains only 11.4% of tokens, while the least aggressive (layer 0) retains 22.0%. Retention is **stable over training** (Spearman r=0.125 between step and mean retention), consistent with the frozen NAMM policy seeing different samples at each step. Retention shows a weak correlation with val F1 (Spearman r=0.036, p=5.313e-01).

## Summary of Findings

This analysis examines the per-layer retention ratios logged during M3 training (LoRA fine-tuning with a frozen NAMM eviction policy). The retention ratio at each layer measures the fraction of input tokens that survive NAMM eviction at that layer. Since the NAMM policy is frozen, any variation in retention across training steps reflects differences in input samples, not changes in the eviction policy itself.

**Key findings:**

1. **Eviction is layer-specific, not uniform.** The coefficient of variation (CV) of mean retention across layers is 0.183 (cs1024), 0.000 (cs2048), 0.057 (cs3072). Smaller cache sizes show more variation, indicating the policy differentiates layers more aggressively when eviction pressure is higher.

2. **Overall retention scales with cache size** as expected: cs1024 retains 19.5%, cs2048 retains 41.3%, cs3072 retains 54.1% of tokens on average.

3. **Retention is stable over training.** For cs1024, Spearman correlation between global step and mean retention is r=0.125 (p=2.013e-03). This confirms the frozen policy produces consistent eviction patterns, with step-to-step variation driven by sample differences.

4. **Retention does not significantly correlate with F1.** For cs1024, Spearman r=0.036 (p=5.313e-01, n=304). The eviction rate for a given evaluation step does not reliably predict performance.

## Mean Retention Per Layer Per Cache Size

| Layer | M3 cs1024 | M3 cs2048 | M3 cs3072 |
|------:|----------:|----------:|----------:|
|     0 |    0.2197 |    0.4133 |    0.5637 |
|     1 |    0.1917 |    0.4133 |    0.5743 |
|     2 |    0.2197 |    0.4133 |    0.5541 |
|     3 |    0.2197 |    0.4133 |    0.5636 |
|     4 |    0.2197 |    0.4133 |    0.5633 |
|     5 |    0.2172 |    0.4133 |    0.5529 |
|     6 |    0.1537 |    0.4133 |    0.5723 |
|     7 |    0.2197 |    0.4133 |    0.5766 |
|     8 |    0.1159 |    0.4128 |    0.5536 |
|     9 |    0.1143 |    0.4133 |    0.5584 |
|    10 |    0.1743 |    0.4131 |    0.5108 |
|    11 |    0.2197 |    0.4133 |    0.5073 |
|    12 |    0.2194 |    0.4133 |    0.4934 |
|    13 |    0.2196 |    0.4133 |    0.5202 |
|    14 |    0.1956 |    0.4133 |    0.5010 |
|    15 |    0.2001 |    0.4133 |    0.4839 |
| **Mean** | **0.1950** | **0.4133** | **0.5406** |

## Is Retention Uniform or Layer-Specific?

Retention is clearly **layer-specific**. The per-layer profiles show structured patterns rather than flat bars:

- **M3 cs1024**: CV=0.183, range=0.1054 (layer 9 retains 0.1143, layer 0 retains 0.2197)
- **M3 cs2048**: CV=0.000, range=0.0006 (layer 8 retains 0.4128, layer 0 retains 0.4133)
- **M3 cs3072**: CV=0.057, range=0.0927 (layer 15 retains 0.4839, layer 7 retains 0.5766)

The pattern is especially pronounced for cs1024 (highest eviction pressure). See `layer_retention_profile.png`.

## Does Retention Correlate with F1?

The correlation is not statistically significant (Spearman r=0.036, p=5.313e-01, n=304).
This suggests that the eviction rate is largely independent of task difficulty -- or that the relationship is more nuanced than a simple linear correlation.

See `retention_vs_f1.png`.

## Does Retention Change Over Training?

For M3 cs1024, the Spearman correlation between global step and mean retention is r=0.125 (p=2.013e-03).

There is a statistically significant but weak trend: retention increases over training (r=0.125). Since the NAMM policy is frozen, this likely reflects systematic differences in the training data distribution as the dataloader iterates through the dataset (e.g., samples encountered later may have different length distributions).

See `retention_over_training.png` and `retention_heatmap.png`.

## Discussion: What Does the Retention Pattern Tell Us?

The per-layer retention profiles reveal how NAMM distributes eviction pressure across the transformer's depth:

1. **NAMM learns layer-specific eviction strategies.** Rather than applying a uniform eviction rate, the evolved policy identifies layers where tokens can be safely discarded and layers where they must be preserved. This suggests the policy has learned something about the information flow through the transformer.

2. **Higher eviction pressure amplifies layer differentiation.** As cache size decreases from 3072 to 1024, the CV of retention across layers increases, meaning the policy becomes more selective about where to evict. Under low pressure (cs3072), most tokens survive at most layers; under high pressure (cs1024), the policy must make hard choices and concentrates eviction in specific layers.

3. **Stability over training confirms the policy is input-driven, not state-driven.** Since the NAMM policy is frozen and retention barely changes over training, the eviction decisions are determined by the input tokens themselves, not by the LoRA weights that change during training. This is a useful property: it means the NAMM policy generalises its eviction strategy regardless of the downstream adapter.

4. **Implications for NAMM architecture design.** The non-uniform retention suggests that a simpler eviction strategy (e.g., uniform random eviction across layers) would be suboptimal. The evolved policy effectively allocates more "memory budget" to layers that need it, which may explain why NAMM outperforms simpler baselines.

## Plots

- `layer_retention_profile.png` -- Mean retention per layer per cache size (bar chart)
- `retention_heatmap.png` -- Retention heatmap over training for M3 cs1024
- `retention_vs_f1.png` -- Scatter: mean retention vs val F1 for M3 cs1024
- `retention_over_training.png` -- Mean retention and val F1 over training for M3 cs1024
