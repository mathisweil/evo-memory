# ES + NAMM Experiment Notes

## Cache Size Efficiency Frontier

Key finding: NAMM with small cache (15%, i.e. 1024/6500 tokens) is **slower** than no eviction (~500s/iter vs ~240s/iter) because the scoring network runs on every eviction and eviction triggers frequently. At 40-60% cache (2600-3900 tokens), eviction is infrequent enough that scoring overhead is minimal (~165s/iter), making it **faster** than full-context processing.

**TODO**: Plot eviction rate vs inference speed across cache sizes (15%, 25%, 40%, 60%, 80%, 100%) to find the optimal efficiency point.

## KV Cache Capping During Batch Size Detection

The auto batch-size detector (`_detect_batch_size` in `evaluator.py`) bypasses NAMM eviction to avoid crashes on uninitialised scoring-network state. Without capping, the KV cache grows unboundedly during detection chunks, causing the detector to overestimate GPU memory usage and select overly conservative batch sizes.

Fix: after each chunk in detection, cap the KV cache to `max_memory_length` (= `cache_size`). This simulates the effect of NAMM eviction on memory without requiring the scoring network, giving tighter batch size estimates that better match actual runtime memory usage.

## cache_size Override Has No Effect (BinarySelection Bug) — RESOLVED

**Status:** Fixed by commit `a0bdee4` (TPU cache validity masking, 2026-03-11). Only affected the GPU code path.

The 40% and 60% cache experiments on **GPU** produced bit-for-bit identical results to 15%. Root cause: `BinarySelection.select_new_tokens()` uses a two-stage eviction:

1. **Threshold at score >= 0** (`threshold_score_idxs` in `base_dynamic.py`): keeps only tokens where the scoring network output is non-negative.
2. **TopK cap at cache_size**: only activates if tokens surviving step 1 still exceed `cache_size`.

On GPU, the threshold eviction **compounds across eviction steps** — evicted tokens are physically removed, shrinking the cache. After ~15 eviction steps on a 4000-token input, only ~191 tokens survive, well below any `cache_size` cap.

**Fix (TPU path):** The TPU implementation uses fixed-size tensors with a `cache_validity_mask`. Evicted tokens are zeroed out and masked as `-inf` but not physically removed. The cache stays at `cache_size` entries, so the topK cap is always the binding constraint. This prevents compounding and makes `cache_size` overrides effective.

**Experiment_2** ran on TPU and shows distinct results per cache size (baseline F1: 11.65 / 13.12 / 14.43 at c1024 / c3072 / c5120), confirming the fix works.
