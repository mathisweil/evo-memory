# NAMM Efficiency Audit Checklist — Assessment

Date: 2026-03-29
Branch: `namm_threshold_training`

## 3.1 STFT / Spectrogram Extraction

- [x] **STFT vectorized over tokens** — `deep_embedding_spectogram.py:187-203`: attention weights are flattened with `.flatten(start_dim=0, end_dim=-2)` then a single `torch.stft()` call processes all tokens. No Python loop.
- [x] **Hann window pre-computed** — `deep_embedding_spectogram.py:101-109`: window is computed once in `store_stft_params()` and stored as `self.stft_window` registered buffer.
- [x] **Float32 for STFT is necessary** — cuFFT does not support bfloat16. Code correctly casts `.float()` at line 193 and casts back with `.to(orig_dtype)` at line 212.
- [x] **Attention matrix not copied** — `.transpose(dim0=-2, dim1=-1)` at line 154 creates a view, not a copy.

## 3.2 EMA State

- [x] **EMA reduction vectorized** — `base_deep_components.py:76-132` `reduce_ema_values()` uses discount vectors, broadcasting, and `.sum()`. No per-token loop.
- [x] **EMA pruning uses gather** — `base_deep_components.py:580-583`: `torch.gather(input=ema_output, dim=2, index=retained_idxs)`.
- [x] **No unnecessary .contiguous()** — EMA buffers are not explicitly made contiguous; gather handles layout.

## 3.3 BAM Self-Attention

- [x] **BAM uses `F.scaled_dot_product_attention`** — `modules/attention.py:305-312` (multi-head) and `410-416` (mono-head). Dispatches to FlashAttention when available.
- [x] **Causal mask pre-computed** — `modules/attention.py:200-216`: `_init_causal_mask()` builds full mask at init time, stored as `self._causal_mask` registered buffer. `_get_causal_mask()` slices at runtime.
- [x] **Linear applied as batch** — `modules/base.py:139`: `torch.baddbmm(input=bias, batch1=input, batch2=weight)` for batched matmul.
- [ ] **Residual addition uses `+` not `add_`** — `deep_scoring.py:506`: `current_output = current_output + next_output`. Allocates new tensor. Impact is minimal for the small scoring network.

## 3.4 Eviction / KV Cache Reindexing

- [x] **KV cache uses `torch.gather`** — `deep.py:623-626`: `torch.gather(key_cache, dim=-2, index=exp_retained_idxs)`. Fast single-op reindexing.
- [ ] **Eviction per-layer in Python loop** — `base.py:151-208`: `for i, layer_past_key_values in enumerate(past_key_values):` processes each layer sequentially. This is architecturally required since scoring depends on per-layer attention patterns.
- [x] **No defensive .clone()** — No `.clone()` on KV cache before or after eviction.

## 3.5 Python Overhead and Dispatch Cost

- [ ] **Python loop over layers** — Same as 3.4. Each layer's attention produces unique scores, so the loop is required. GPU is partially idle during Python dispatch between layers.
- [x] **`torch.no_grad()` applied consistently** — `trainer.py:967` (`_train_step()`), `trainer.py:1169` (`train()`), `evaluator.py:230` all use `@torch.no_grad()` decorator or context manager. CMA-ES does not backpropagate.
- [x] **`torch.compile` applied to scoring network** — **FIXED**: Added `compile_scoring: bool` parameter to `MLPScoring` and `GeneralizedScoring` in `deep_scoring.py`. When enabled, compiles MLP/attention forward methods with `mode='reduce-overhead', dynamic=True`.

## 3.6 Population-Level Batching

- [x] **Population members batched** — `policy/base.py:750-762`: `set_params_batch_idxs()` creates index tensors; `get_layer_params()` uses `torch.gather` to select per-batch-element params. Multiple pop members evaluated in same forward pass.
- [ ] **LLM prefill repeated per pop member** — Each pop member's sequence is duplicated and processed independently through the LLM. Prefix KV-cache reuse not implemented. Gated `cache_prefix_reuse` flag planned but not yet implemented (high risk).
- [x] **Same sequences reused across pop members** — `evaluator.py:612`: `dataset_samples = dataset_samples*pop_reps`.

## 3.7 Memory Layout and Dtype

- [x] **NAMM params cast to model dtype once** — **FIXED**: `policy/base.py:set_params()` now casts `pop_params` and `pop_shared_params` to `self._model_dtype` after copying float32 CMA-ES params. Set via `trainer.py` calling `model.set_memory_dtype(self.ptdtype)`.
- [x] **Attention matrix detached** — Outer `@torch.no_grad()` on all training/eval paths prevents autograd graph retention.
- [x] **Positional embeddings pre-computed** — `modules/attention.py:42-54`: `_cos_cached` and `_sin_cached` stored as registered buffers.
- [x] **Float32 decorator removed** — **FIXED**: Removed `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` from `update_cache()` and `buffer_cache()` in `policy/base.py`. The STFT path already handles its own float32 requirement internally.

## Summary

| Category | Items | Already OK | Fixed | Not Applicable / Deferred |
|----------|-------|-----------|-------|---------------------------|
| 3.1 STFT | 4 | 4 | 0 | 0 |
| 3.2 EMA | 3 | 3 | 0 | 0 |
| 3.3 BAM | 4 | 3 | 0 | 1 (residual `+` vs `add_` — minimal impact) |
| 3.4 Eviction | 3 | 2 | 0 | 1 (per-layer loop — architecturally required) |
| 3.5 Python | 3 | 1 | 1 | 1 (per-layer loop — architecturally required) |
| 3.6 Population | 3 | 2 | 0 | 1 (prefix reuse — high risk, deferred) |
| 3.7 Dtype/Layout | 4 | 2 | 2 | 0 |
| **Total** | **24** | **17** | **3** | **4** |
