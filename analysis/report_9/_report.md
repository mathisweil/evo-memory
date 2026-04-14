# Analysis 9 -- Gradient Flow and Loss Attribution Under Eviction

> Checkpoints: M2 NAMM from WandB `z5bo4n8k`, M3 LoRA from `h0bzg6on` step 260.

> **TL;DR:** NAMM eviction retains ~4% of tokens and increases CE loss by
> ~79% relative to full context. Gradient directions under eviction are
> weakly correlated with full-context gradients (cosine similarity ~0.21),
> indicating partial but not full alignment. The eviction regime creates a
> harder but not unrecognisable optimisation landscape.

## Methodology

Instrumented forward+backward passes over training data, computing
per-token CE loss on answer tokens and recording LoRA gradient norms under
evicted (cache_size=1024) and full-context conditions.

- **Samples:** 255 (51 per task, balanced across all 5 tasks)
- **Checkpoint:** M3 (`h0bzg6on` step 260, val F1 52.06) with M2 NAMM (`z5bo4n8k`)
- **Cache size:** 1024 tokens

## Results

### Loss Under Eviction

| Metric                      |             Value |
| --------------------------- | ----------------: |
| Mean evicted loss           |  2.291 +/- 1.687  |
| Mean full-context loss      |  1.283 +/- 1.219  |
| Absolute loss gap           |             1.008 |
| Loss increase from eviction |              79%  |
| Mean retention ratio        | 0.041 +/- 0.005  |

Eviction increases CE loss by 79% (from 1.28 to 2.29). The NAMM retains
approximately 4.1% of tokens per layer. The loss increase is moderate —
the model can still predict answer tokens reasonably well even with ~96%
of context evicted.

### Gradient Direction Consistency

Per-layer cosine similarity between gradient directions computed under
evicted vs full-context conditions.

The overall mean cosine similarity is approximately **0.21**. This is
weakly positive — gradient directions under eviction partially align with
full-context gradients, meaning the optimisation signal from evicted
training is not random relative to full-context training. The model
receives a noisy but directionally informative gradient signal.

### Per-Layer Gradient Norms

Gradient norms under eviction are higher in early layers (0-5) and
comparable in later layers (12-15). Eviction amplifies gradients in
early layers where the model first processes the truncated context.

## Interpretation

### Eviction Creates a Harder but Related Task

The 79% loss increase and 0.21 cosine similarity paint a different
picture from the old 40-sample analysis (which reported +641% loss and
~0.01 cosine). With balanced sampling across all 5 tasks and 255
samples, the eviction regime is substantially less severe than previously
reported. The gradient signal under eviction points in a similar (though
noisier) direction as full-context gradients, meaning the LoRA receives
meaningful training signal even under eviction.

### Why the Old Numbers Were Wrong

The previous analysis used 40 samples drawn unevenly from a few tasks,
biasing toward harder samples with longer sequences and more eviction.
The balanced 255-sample analysis shows that across the full task
distribution, eviction is less catastrophic than the skewed subset
suggested.

### M3 Learns Effectively Under Eviction

M3 achieves val F1 52.06 (surpassing M1's 45.48 by 14.5%) while
training under this gradient regime. The partial gradient alignment
(cos ~0.21) explains how this is possible — the training signal is noisy
but not random, allowing the LoRA to gradually improve.

## Connection to Other Reports

| Report          | Finding                              | Implication for Rpt 9                      |
| --------------- | ------------------------------------ | ------------------------------------------ |
| 1 (Sensitivity) | M3 surpasses M1 by 14.5%            | Effective learning under moderate distortion|
| 3 (Retention)   | Retention ~4%                        | Consistent with retention observed here     |
| 5 (Entropy)     | M2 ≈ M3 attention entropy            | Gradients change magnitudes, not attention  |
| 6 (Alignment)   | rho ≈ 0 (no NAMM-attention trend)    | NAMM scoring independent of model state     |

## Figures

| File                                   | Description                                            |
| -------------------------------------- | ------------------------------------------------------ |
| `plots/loss_stratified.png`            | Box plot of CE loss: full context vs evicted            |
| `plots/grad_norms.png`                 | Per-layer LoRA gradient L2 norms under eviction vs full |
| `plots/grad_direction_consistency.png` | Per-layer cosine similarity of gradient directions      |
