# Analysis 7 — Representation Similarity (CKA)

> **TL;DR:** Linear CKA between M1 and M3 cs1024 is **very high (0.979–1.0) but not identical** across all layers when both models process full-context inputs. The embedding layer is 1.0 (shared tokenizer), but transformer layers show a characteristic dip, with the minimum at **layer 3 (CKA=0.979)**. This aligns with Report 4's finding that M3's LoRA norms are 1.5-2.6× larger in orthogonal subspaces — the weight-space divergence produces a small but measurable representation-space divergence, even on full context. The cross-layer heatmap reveals an asymmetric block structure with a transition around layers 6-7.

## Methodology

We loaded M1 and M3 cs1024 checkpoints (LoRA merged into base weights), ran inference on 10 test prompts (1024 tokens each), and extracted mean-pooled hidden states at all 17 layers (embedding + 16 transformer layers).

**Linear CKA** measures representation similarity between two models:
```
CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
```
CKA = 1.0 means identical representations (up to linear transform); CKA = 0.0 means unrelated.

We computed:
1. **Layer-wise CKA:** M1 layer i vs M3 layer i for all 17 layers
2. **Cross-layer CKA heatmap:** M1 layer i vs M3 layer j for all (i, j) pairs

## Findings

### Layer-wise CKA

| Layer | CKA |
|-------|-----|
| Embedding | 1.000 |
| Layer 0 | 1.000 |
| Layer 1 | 1.000 |
| Layer 2 | 0.996 |
| **Layer 3** | **0.979** |
| Layer 4 | 0.988 |
| Layer 5 | 0.990 |
| Layer 6 | 0.993 |
| Layer 7 | 0.993 |
| Layer 8 | 0.993 |
| Layer 9 | 0.991 |
| Layer 10 | 0.989 |
| Layer 11 | 0.991 |
| Layer 12 | 0.990 |
| Layer 13 | 0.990 |
| Layer 14 | 0.995 |
| Layer 15 | 0.994 |

See `cka_by_layer.png`. All layers show CKA > 0.97, with a dip at layer 3.

**Mean CKA across all layers: 0.992**

### Cross-layer CKA

See `cka_heatmap.png`. Key observations:
- The **diagonal** is near-1.0 throughout, confirming high same-layer similarity
- **Block structure:** Early layers (Emb–6) and late layers (7–15) form two similarity blocks with a visible transition around layer 6-7
- **Early–late asymmetry:** Embedding representations have lower CKA with deep layers (~0.6-0.7) than deep layers have with each other (~0.9+)
- The block structure is consistent across both models, reflecting the base model's architecture

## Interpretation

### CKA is high but not 1.0

The LoRA update modifies the model's weights by a rank-8 perturbation to q_proj and v_proj. For LLaMA 3.2-1B, these projections have dimension 2048. A rank-8 update changes at most 8/2048 = 0.4% of the weight space per layer. On full-context inputs, the base model's 1.2B parameters largely dominate, but the LoRA perturbation is large enough to produce a detectable CKA difference of ~1-2%.

### Layer 3 as the point of maximum divergence

The CKA minimum at layer 3 (0.979) is notable. This is in the early-middle layers where:
- Attention transitions from positional/syntactic to semantic processing (Report 5 shows low entropy in layers 1-3)
- Report 4 showed M3 LoRA norms grow with layer depth, but the largest *relative* impact may occur in early layers where base representations are less entrenched
- The q_proj modifications at this layer may alter how the model routes information, with effects that partially recover in later layers

### Connection to other reports

| Report | What it measures | M1 vs M3 on full context |
|--------|-----------------|--------------------------|
| 4 (LoRA weights) | Weight-space difference | **Orthogonal subspaces**, M3 norms 1.5-2.6× larger |
| 5 (Attention) | Function-space difference (attention) | **Measurably different**: M3 higher entropy (+4%), lower sinks |
| 7 (CKA) | Function-space difference (representations) | **Very similar but not identical**: CKA 0.979-1.0 |

This is a coherent picture: M1 and M3 differ in weight space (Report 4), and this translates to small but measurable differences in both attention patterns (Report 5) and hidden representations (Report 7) even on full context. The adaptation is not fully dormant — it produces a functional signature that may serve as a form of pre-adaptation, readying the model for the distribution shift caused by eviction.

### Implications

1. **M3's adaptation has a small cost on full context.** The models are not functionally identical — M3's LoRA weights produce slightly different representations. However, Report 1 shows M3 matches M1 on aggregate F1 (val: 45.59 vs 45.48; test micro: 32.28 vs 31.14), so the representational difference does not hurt task performance — M3 reaches similar answers via a different computation path.
2. **CKA is sensitive enough to detect rank-8 LoRA perturbations.** CKA can resolve ~1-2% representation differences from low-rank fine-tuning, making it a viable tool for comparing LoRA-adapted models.
3. **Layer 3 may be a critical adaptation point.** The CKA dip at layer 3 suggests this is where M3's LoRA makes the largest representational change, potentially redirecting information flow for eviction robustness.

### Implications for the paper

Combined with Reports 4 and 5, the CKA results support a **"pre-emptive hedging"** interpretation of M3's adaptation rather than a "dormant adaptation" story:

1. **M3 is a genuinely different model, not a dormant variant of M1.** The orthogonal LoRA subspaces (Report 4), the attention entropy shift (Report 5, +4%), and the CKA divergence (this report, min 0.979) all show that M3 occupies a different region of function space. The adaptation is always active, not switched on by eviction.

2. **Different path, same destination.** M3 produces different representations and attention patterns but achieves the same task performance on full context (Report 1). This means eviction-aware training finds an alternative solution that is robust to both full-context and evicted-context conditions — a stronger claim than "it doesn't change anything without eviction."

3. **The adaptation is concentrated, not diffuse.** The CKA dip at layer 3 and the head-specific entropy shifts (Report 5) show that M3's changes are localised to specific layers and heads, consistent with LoRA's low-rank structure selectively modifying particular attention circuits rather than globally perturbing the model.

## Figures

- `cka_by_layer.png` — Layer-wise CKA (0.979–1.0 range)
- `cka_heatmap.png` — Cross-layer CKA heatmap (M1 layer i vs M3 layer j)

## References

- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." *ICML 2019*.
- Raghu et al. (2017). "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability." *NeurIPS 2017*.
- Neyshabur et al. (2020). "What is being transferred in transfer learning?" *NeurIPS 2020*.
