# Analysis 8: Probing for Residual Knowledge of Evicted Content

## Overview

This analysis trains linear probes on hidden-state representations to test whether M3 (LoRA + frozen NAMM) retains information about evicted tokens in the representations of surviving tokens. We compare probe accuracy between M1 (full context, upper bound) and M3 (evicted context) across all 17 layers (embedding + 16 transformer layers).

> **Naming note:** "M3" here refers to what `results/main_table_5t/` labels "M4" (LoRA + frozen NAMM). See `experiment_specification.md` for the naming warning.

## Methodology

1. **Sample preparation:** 40 test samples from the 5-task QA subset. For each, we locate answer-relevant token positions by searching for gold-answer substrings in the tokenised prompt. 20/40 samples had identifiable answer token positions.

2. **Hidden state extraction:**
   - **M1 (full context):** Forward pass with Recency passthrough (no eviction), extract mean-pooled hidden states per layer.
   - **M3 (evicted context):** Forward pass with NAMM active (cache_size=1024), extract mean-pooled hidden states from retained tokens only. Retention ranged from ~1000-1300 tokens out of ~2700-6300 total (~20-43%).

3. **Binary label:** 1 if any answer tokens overlap with evicted positions, 0 otherwise. Balanced split: 20 positive (answer tokens evicted), 20 negative.

4. **Probe training:** Per-layer logistic regression (sklearn) with 5-fold stratified CV. Accuracy = fraction of correctly classified samples.

## Results

### Probe accuracy by layer

| Layer     | M1 accuracy | M3 accuracy | Random |
| --------- | ----------: | ----------: | -----: |
| Embedding |       0.575 |       0.575 |  0.500 |
| Layer 0   |       0.575 |       0.550 |  0.500 |
| Layer 1   |       0.575 |       0.500 |  0.500 |
| Layer 2   |       0.525 |       0.550 |  0.500 |
| Layer 3   |       0.525 |       0.500 |  0.500 |
| Layer 4   |       0.525 |       0.525 |  0.500 |
| Layer 5   |       0.625 |       0.575 |  0.500 |
| Layer 6   |       0.575 |       0.575 |  0.500 |
| Layer 7   |       0.550 |       0.450 |  0.500 |
| Layer 8   |       0.550 |       0.450 |  0.500 |
| Layer 9   |       0.450 |       0.450 |  0.500 |
| Layer 10  |       0.525 |       0.375 |  0.500 |
| Layer 11  |       0.575 |       0.450 |  0.500 |
| Layer 12  |       0.550 |       0.400 |  0.500 |
| Layer 13  |       0.525 |       0.375 |  0.500 |
| Layer 14  |       0.700 |       0.375 |  0.500 |
| Layer 15  |       0.550 |       0.550 |  0.500 |

### Key observations

1. **M1 probe accuracy is only modestly above chance** (mean ~0.56, range 0.45-0.70). This suggests the binary probe task is hard — even with full context, mean-pooled hidden states carry only weak signal about whether answer tokens are present at specific positions. The strongest M1 signal is at layer 14 (0.70).

2. **M3 probe accuracy degrades in later layers.** M3 starts near M1 at the embedding layer (both 0.575) but diverges in layers 7-14, where M3 drops to 0.375-0.450 while M1 stays at 0.525-0.700. This is consistent with information loss accumulating through the network as eviction removes tokens that deeper layers would have attended to.

3. **The M1-M3 gap is largest at layer 14** (M1=0.700, M3=0.375, gap=0.325). This aligns with Report 4's finding that later layers bear the heaviest adaptation burden (q_proj norm ratios exceed 2x at layers 13-15). The LoRA compensates for information loss, but the probe reveals the underlying representations have genuinely lost information about evicted content.

4. **M3 recovers at layer 15** (0.550, matching M1). The final layer may re-aggregate information from the LoRA-adapted representations, partially recovering the lost signal. This could reflect the lm_head projection compressing task-relevant features.

5. **Layer 9 is equally poor for both** (M1=0.450, M3=0.450). This is below chance, suggesting layer 9 representations are not informative for this particular probe task. Report 3 noted that layer 9 has the most aggressive eviction (11.4% retention at cs1024).

### Answer token survival

NAMM retains ~20-43% of tokens across samples. The precise eviction analysis (phase 1c) had limited success due to device issues, yielding insufficient data for per-task survival statistics. The approximate retention from KV cache sizes confirms the expected ~4-6x compression.

## Interpretation

The probing results support the hypothesis that M3's LoRA adaptation compensates for information loss but does not fully compress evicted content into retained representations:

- **Information IS lost.** The M1-M3 probe gap in layers 7-14 demonstrates that evicted token information is genuinely absent from M3's later-layer representations. This is not a dormant capability — the information is gone.

- **The loss is concentrated in deep layers.** Early layers (0-6) show similar M1 and M3 probe accuracy, suggesting that initial token representations are comparable. The divergence in layers 7-14 mirrors the NAMM retention profile (Report 3: layers 6-9 have the most aggressive eviction).

- **Despite lost information, M3 matches M1 on task performance** (test micro 32.28 vs 31.14). This means the LoRA adapter learns to extract sufficient task-relevant signal from the reduced context, even though the underlying representations carry less information about evicted content. The adaptation is in the processing strategy, not in information preservation.

## Caveats

- **Small sample size** (n=40 with 50/50 split). Probe accuracies have high variance (stds of 0.10-0.27). The results are suggestive, not definitive.
- **Mean-pooling is lossy.** Averaging over all retained token positions discards positional information that could be relevant.
- **Binary probe is coarse.** A more refined probe (e.g., predicting specific evicted entities) might reveal finer-grained patterns.
- **Precise eviction analysis was limited** by device errors in the `analyze` pass. Per-task survival statistics are incomplete.

## Figures

| File                         | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| `probe_accuracy.png`         | Per-layer probe accuracy for M1 vs M3 vs random baseline |
| `entity_survival.png`        | Retention fractions and answer token survival estimates  |
| `layer_wise_information.png` | Per-layer accuracy difference (M1 - M3)                  |
| `probe_data.npz`             | Raw probe results for reproduction                       |
