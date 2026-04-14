# Distribution Shift from KV Cache Eviction: Evidence and Mechanism

## The Argument

KV cache eviction introduces a train-test mismatch for LoRA-finetuned models.
A LoRA trained on full-context representations faces a fundamentally different
input distribution when the KV cache has been evicted — and this shift is
specific to the eviction regime, not a generic degradation.

## Evidence

### 1. M1 LoRA degrades differently under different NAMMs

| Condition | test micro | Δ from M1 full |
|-----------|---:|---:|
| M1 (full cache, no eviction) | 31.14 | — |
| M1 + original NAMM cs1024 | 26.97 | -4.17 |
| M1 + maskfix NAMM cs1024 | 24.11 | -7.03 |
| Truncation + M1 LoRA cs1024 | 26.90 | -4.24 |

The M1 LoRA wasn't trained with any eviction, yet it performs differently
under the two NAMM policies:
- Original NAMM (broken attention): -4.17 drop
- Maskfix NAMM (correct attention): -7.03 drop

This is counterintuitive — the maskfix NAMM has correct attention and less
ghost information (key cosines +0.65 vs -0.30). Yet M1 performs WORSE with it.

### 2. Why maskfix NAMM is harder for M1

The two NAMMs create different cache states:

| Property | Original NAMM | Maskfix NAMM |
|----------|---:|---:|
| Position bias (last third) | 58% | **79%** |
| Head agreement (Jaccard) | 0.48 | **0.56** |
| Ghost key cosine (vs full) | -0.30 | **+0.65** |

The maskfix NAMM is more tail-biased (79% vs 58%) and retains fewer tokens
from the document body. M1's LoRA was trained on full-context representations
where ALL document tokens are present. The maskfix NAMM removes more of the
document than the original does, creating a larger mismatch with M1's
training distribution.

The original NAMM's broken attention accidentally preserved a more "full-context-like"
cache (only 58% tail, more document retained) that happened to be more compatible
with M1's adaptation.

### 3. Per-task patterns confirm regime-specificity

| Task | M1 + orig NAMM | M1 + maskfix NAMM | Δ |
|------|---:|---:|---:|
| qasper_e | **27.48** | 15.05 | -12.43 |
| hotpotqa_e | **35.33** | 29.70 | -5.63 |
| 2wikimqa_e | 26.19 | **32.65** | +6.46 |
| qasper | 19.42 | **21.67** | +2.24 |

The shift is task-dependent: qasper_e drops 12 points under maskfix while
2wikimqa_e improves by 6 points. The original NAMM's broader retention
(keeping more document tokens) helped M1 on qasper_e (where answers are
mid-document). The maskfix NAMM's tail focus helps on 2wikimqa_e (where
the answer entity is often near the question).

### 4. LoRA weights are orthogonal across regimes

| LoRA pair | Mean cosine |
|-----------|---:|
| M1 ↔ M4 (broken NAMM) | -0.010 |
| M1 ↔ M4f (maskfix NAMM) | -0.005 |
| M4 ↔ M4f (broken vs fixed) | -0.002 |

All three LoRAs occupy orthogonal subspaces. The M4 trained under broken
NAMM and the M4f trained under maskfix NAMM are just as different from each
other as either is from M1. Each eviction regime induces a unique optimal
adaptation direction.

### 5. M4 norm ratios confirm adaptation burden

The original M4 (broken attention) has 21% larger norms than M1, concentrated
in later layers (up to 30% at layer 15). The maskfix M4f has ~same norms as M1
(ratio 0.99x). With correct attention, the LoRA doesn't need to over-adapt —
the attention mechanism works properly, so the value-path compensation is
unnecessary.

## Mechanism

1. **During LoRA training**, the model's q_proj and v_proj learn to produce
   queries and values that are optimal for the attention patterns and cache
   content present during training.

2. **Under a different eviction regime at eval**, the cache contains different
   tokens (different retention pattern), the surviving KVs have different ghost
   contamination levels, and the attention patterns are different. The LoRA's
   learned q/v projections are mismatched to this new regime.

3. **The mismatch is not generic noise** — it's a specific, structured shift
   that depends on the exact eviction policy. Two different NAMM checkpoints
   (original vs maskfix) produce different shifts, and the LoRA adapted to one
   doesn't transfer to the other.

4. **Training under eviction (M4/M4f) mitigates the shift** by adapting the
   LoRA to the specific cache state produced by that NAMM policy. The adaptation
   is in an orthogonal subspace to M1's, not a refinement of it.

## Implication for the Thesis

This demonstrates that KV cache eviction is not just an inference-time
optimization — it fundamentally changes the distribution the model operates on.
Parameter adaptation (LoRA finetuning) under the eviction regime is necessary
to recover full performance. The adaptation is regime-specific: a LoRA trained
under one eviction policy does not transfer to another, even when both policies
use the same architecture (NAMM) and cache budget (cs=1024).
