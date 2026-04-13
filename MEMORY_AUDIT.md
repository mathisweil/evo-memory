# VRAM Memory Audit -- LoRA Fine-Tuning Pipeline

## 1. Memory Lifecycle Timeline

The following traces VRAM allocation from model load through steady-state
training for the M1 (LoRA-only) and M3 (LoRA + frozen NAMM) conditions on a
single GPU with 8 GB usable VRAM.

### 1.1 Model Load (`run_lora.py`)

| Step | What happens | Approx VRAM |
|---|---|---|
| `make_eval_model()` | HF loads Llama-3.2-1B-Instruct in default dtype (fp32), wraps it in `WrappedLlamaForCausalLM` which deep-copies weights into `LlamaMemoryModel`, deletes the original, and calls `empty_gpu_cache()`. Net: one copy of the model on GPU (still fp32 at this point). | ~4.96 GB |
| `memory_model.to(bf16, device)` | Casts all parameters to bf16 on GPU. | ~2.48 GB |
| `gradient_checkpointing_enable()` (M1 only) | Registers checkpoint hooks on each `LlamaMemoryDecoderLayer`. No new tensors; just hooks. | +0 |
| `apply_lora_adapters()` | Injects LoRA A/B matrices via PEFT. Freezes base weights. Casts LoRA params to fp32. | +4 MB |
| AdamW construction | Allocates momentum + variance buffers (2x LoRA params). | +8 MB |
| Cosine LR scheduler | Python state only. | +0 |
| **Post-init steady state** | | **~2.50 GB** |

### 1.2 First Training Step (`_train_step`)

**M1 (non-NAMM, gradient checkpointing ON)**

| Sub-step | Tensors | VRAM delta |
|---|---|---|
| `input_ids.to(device)` + `labels.to(device)` | [1, 7000] int64 x2 | +112 KB |
| Forward pass through 16 layers | With gradient checkpointing only segment-boundary activations are saved. Per-layer attention matrix [1, 32, q, k] is computed and freed within each recomputed segment. | +200-400 MB transient |
| `output_hidden_states=True` (before fix) | Returns tuple of 17 hidden-state tensors [1, 7000, 2048] bf16. With grad ckpt these are *extra* copies that defeat checkpointing. | **+~460 MB** |
| `shift_hidden = hidden_states[:,:-1,:].contiguous()` | New contiguous copy [1, 6999, 2048] bf16 | +28.7 MB |
| Chunked cross-entropy (512-tok chunks) | `lm_head(chunk)` produces [1, 512, 128256] fp32 logits, immediately deleted. Peak per-chunk: | +250 MB transient |
| `loss.backward()` | Recomputes forward (gradient checkpointing). Gradient tensors for LoRA A/B only. | +4 MB grads |
| **Per-step peak (before fix)** | | **~3.4 GB** |

**M3 (NAMM active, no gradient checkpointing)**

| Sub-step | Tensors | VRAM delta |
|---|---|---|
| Phase 1 (no_grad): context tokens up to `context_end` | KV cache built incrementally in 64-token chunks via split processing. After eviction: 16 layers x 2 x [1, 8, 1024, 64] bf16. | +33 MB |
| `del ctx_outputs; empty_cache()` | Frees intermediate outputs from phase 1. | -small |
| Phase 2 (with grad): ~64 answer tokens | Forward through answer tokens with past KV. Attention: [1, 32, 64, 1088] bf16 per layer. All 16 layers retain activations (no gradient checkpointing). | +100-200 MB |
| Chunked CE + backward | Same pattern as M1 but on ~64 tokens. Much smaller. | +small |
| **Per-step peak** | | **~2.9 GB** |

### 1.3 Periodic Evaluation (every `eval_interval` steps)

| Sub-step | Tensors | VRAM delta |
|---|---|---|
| `model.eval()` | Switches BN/dropout mode. No new tensors. | +0 |
| `empty_cache()` | Returns freed blocks to allocator. | -variable |
| `_evaluate_f1('val')` | Iterates over ~64 val samples, calling `evaluate_lb()` per task. | |
| -- per sample: `tok_batch_encode` | [1, ~6500] int64 input_ids + attn_mask on GPU | +104 KB |
| -- per sample: `model.generate()` | Context processed in 64-token chunks (split processing). KV cache grows to full prompt length (M1) or cache_size=1024 (M3). Attention per chunk: [1, 32, 64, kv_len] fp32 softmax. | +200-500 MB transient |
| -- per sample: decode + score | String decode, F1 scoring on CPU. | +0 |
| `empty_cache()` after each task | Frees generation KV caches. | -variable |
| `_evaluate_f1('train', num_samples=val_n)` | Same as val. | same |
| `_debug_generate(n=3)` | 3 samples through generate(). | same |
| `model.train(); _freeze_base_weights()` | Restores training mode. | +0 |
| **Eval peak (M1, seq_len=6500)** | Attention [1,32,64,6500] fp32 during context split | **~3.0-3.5 GB** |

### 1.4 Checkpoint Save

| Step | VRAM delta |
|---|---|
| `p.data.clone()` for LoRA params (before fix) | +4 MB GPU temporary |
| `torch.save()` | Serializes to disk, creates CPU copies. | +4 MB host |
| Net after save completes | +0 (temporaries freed) |

### 1.5 Steady-State Oscillation

Training steady state: ~2.5-3.0 GB.
Eval peak: ~3.0-3.5 GB.
The 11 GB -> 7 GB oscillation reported in the issue context includes PyTorch's
internal allocator overhead and CUDA context (~1.2 GB).

---

## 2. Itemised VRAM Budget

All estimates for Llama-3.2-1B-Instruct, bf16 base weights, fp32 LoRA,
batch_size=1, seq_len=7000 (training) / 6500 (eval).

### 2.1 Static Allocations

| Component | Size | Notes |
|---|---|---|
| Base model weights (frozen, bf16) | 2.48 GB | ~1.24B params x 2 bytes |
| LoRA A/B matrices (fp32) | 4 MB | r=8, q_proj+v_proj, 16 layers: 1,048,576 params x 4B |
| AdamW momentum (fp32) | 4 MB | Same shape as LoRA params |
| AdamW variance (fp32) | 4 MB | Same shape as LoRA params |
| Gradient tensors (fp32) | 4 MB | Same shape as LoRA params |
| CUDA context + PyTorch allocator | ~1.0-1.2 GB | Fixed overhead |
| **Total static** | **~3.5 GB** | |

### 2.2 Training Dynamic Allocations (per step)

| Component | M1 (no NAMM) | M3 (NAMM active) |
|---|---|---|
| Input tensors [1,7000] int64 x2 | 112 KB | 112 KB |
| Forward activations (with grad ckpt) | ~200 MB | N/A (no grad ckpt) |
| Forward activations (without grad ckpt) | N/A | ~200 MB (phase 2 only, 64 tok) |
| `output_hidden_states=True` tuple (before fix) | **460 MB** (17 x [1,7000,2048] bf16) | 4.5 MB (17 x [1,64,2048] bf16) |
| `shift_hidden` contiguous copy | 28.7 MB | 0.3 MB |
| Chunked CE logits (peak per chunk) | 250 MB ([1,512,128256] fp32) | 250 MB |
| KV cache (M3 phase 1) | N/A | 33 MB (cache_size=1024) |
| Attention matrix (M3 phase 2) | N/A | <1 MB per layer |

### 2.3 Eval Dynamic Allocations (per sample)

| Component | M1 (no NAMM) | M3 (NAMM, cs=1024) |
|---|---|---|
| Input tokens [1,6500] | 52 KB | 52 KB |
| KV cache (full prompt) | 212 MB (16L x 2 x [1,8,6500,64] bf16) | 33 MB (capped at 1024) |
| Attention per 64-tok chunk (fp32 softmax) | [1,32,64,kv_len] -- up to 53 MB | [1,32,64,1088] -- 0.9 MB |
| `lm_head` logits during generation | [1,1,128256] fp32 = 0.5 MB | Same |

### 2.4 Other Buffers

| Component | Size | Notes |
|---|---|---|
| Loss accumulators (`total_loss`, `n_tokens`) | <1 KB | Scalar tensors on GPU |
| Metric tensors in `get_score()` | 0 | Computed on CPU (Python floats) |
| `_last_retention_dict` | 0 | Python dict of floats |
| WandB logging dicts | 0 | Python dicts |
| SFTDataset samples (CPU) | ~50 MB host | ~306 samples x [~7000] int64 tensors |

---

## 3. Leak and Fragmentation Inventory

### 3.1 Confirmed Issues

**L1. `output_hidden_states=True` defeats gradient checkpointing (M1)**
- Location: `trainer.py:451`, `llama.py:598`
- When `output_hidden_states=True`, the inner `LlamaModel.forward()` accumulates
  all 17 layer hidden-state tensors in a Python tuple. With gradient checkpointing,
  autograd does NOT hold references to intermediate activations -- that is the whole
  point of checkpointing. But the `hidden_states` tuple pins them in memory,
  defeating the savings.
- The caller (`_train_step`) only uses `outputs.hidden_states[-1]` (the last
  layer). The other 16 tensors (~460 MB for seq_len=7000) are pure waste.
- The wrapper (`WrappedLlamaForCausalLM.forward()`) propagates the full tuple
  to its return value, keeping all 17 tensors alive until `_train_step` returns.

**L2. `outputs` object retained through backward pass**
- Location: `trainer.py:486-504`
- After the chunked CE loop and `loss.backward()`, the `outputs` variable still
  holds references to `past_key_values`, `attentions`, `query_states`, etc.
  These are accessed for ANLYS-01 logging (line 486) and the return value
  (line 504). The entire output object survives until function return, preventing
  garbage collection of attention weights and other intermediates.

**L3. `shift_hidden` and `shift_labels` retained through backward**
- Location: `trainer.py:435-478`
- After the chunked CE loop completes, `shift_hidden` ([1, 6999, 2048] bf16,
  ~29 MB) and `shift_labels` ([1, 6999] int64, ~56 KB) are no longer needed
  by user code (autograd holds its own internal references). Explicit deletion
  allows Python reference counting to release them sooner.

**L4. Checkpoint save creates GPU clone before CPU copy**
- Location: `trainer.py:529-533`
- `p.data.clone()` allocates a GPU copy of each LoRA parameter. `torch.save()`
  then creates a CPU copy for serialization. The GPU clone is a ~4 MB temporary
  that is unnecessary.

**L5. No `gc.collect()` before `torch.cuda.empty_cache()`**
- Location: `trainer.py:868,1003,1007,1014` and `tasks.py:726`
- `torch.cuda.empty_cache()` only returns memory that Python has already freed.
  Without `gc.collect()` first, cyclic references may prevent Python objects
  (holding CUDA tensor refs) from being freed, making `empty_cache()` less
  effective.

**L6. `torch.no_grad()` used where `torch.inference_mode()` is better**
- Location: `trainer.py:870,879,1005,1091,1095`
- `torch.no_grad()` disables gradient computation but still tracks tensor
  versions and maintains autograd metadata. `torch.inference_mode()` is strictly
  more efficient: it disables version counting and autograd metadata entirely.
  Since eval results are Python floats (not fed back into autograd), inference
  mode is safe and reduces per-tensor overhead.

### 3.2 Fragmentation Sources

**F1. Training-to-eval mode transition**
- After training fills VRAM with activations of one shape (seq_len=7000), eval
  frees them and allocates KV caches of a different shape (seq_len=6500, growing
  incrementally). The PyTorch allocator's block splitting produces internal
  fragmentation.
- Mitigation: `gc.collect()` + `empty_cache()` at every transition.

**F2. Eval generation incremental KV growth**
- `model.generate()` grows the KV cache one token at a time. Each growth
  may trigger a new CUDA allocation rather than extending the existing block.
- Mitigation: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.1+).

**F3. Chunked CE logits allocation/deallocation**
- The 512-token chunked CE loop allocates [1, 512, 128256] fp32 (~250 MB)
  and immediately frees it, 14 times per step (for seq_len=7000). The
  allocator reuses the same block each iteration, so fragmentation is minimal.
  No action needed.

---

## 4. Peak Analysis

### 4.1 M1 Training Peak (before optimisations)

The highest VRAM peak for M1 occurs during the **first chunked CE logit
computation** within `_train_step`:

```
Base model (bf16)                  2,480 MB
CUDA context + allocator           1,200 MB
LoRA + optimizer + gradients          16 MB
output_hidden_states tuple (L1)      460 MB  <-- largest waste
shift_hidden contiguous copy          29 MB
Forward activations (grad ckpt)      200 MB  (recomputed segments)
Chunked CE logits (one chunk)        250 MB
-------------------------------------------
Estimated peak:                   ~4,635 MB
```

After fixing L1, the peak drops by ~460 MB to ~4,175 MB.

### 4.2 M1 Eval Peak

The highest eval VRAM peak occurs during **context processing in generate()**
for the longest prompt (~6500 tokens):

```
Base model (bf16)                  2,480 MB
CUDA context + allocator           1,200 MB
LoRA + optimizer (still allocated)    16 MB
KV cache (6500 tokens, 16 layers)    212 MB
Attention per 64-tok chunk (fp32)     53 MB
lm_head logits                         1 MB
-------------------------------------------
Estimated peak:                   ~3,962 MB
```

### 4.3 M3 Training Peak

M3 training is split into two phases. The peak is lower than M1 because
phase 2 processes only ~64 answer tokens:

```
Base model (bf16)                  2,480 MB
CUDA context + allocator           1,200 MB
LoRA + optimizer + gradients          16 MB
Phase-1 KV cache (cs=1024)           33 MB
Phase-2 activations (64 tok)         100 MB
Chunked CE logits (one chunk)        250 MB
-------------------------------------------
Estimated peak:                   ~4,079 MB
```

---

## 5. Optimisation Plan

### Constraints

All changes below are purely memory-management improvements with zero impact
on training numerics. LoRA rank/alpha/lr, dataset, model, epoch count,
effective batch size, and eval metrics are unchanged.

### O1. Return only last hidden state when `skip_lm_head=True`

- **File:** `namm/llms/llama.py` -- `WrappedLlamaForCausalLM.forward()`
- **What:** When `skip_lm_head=True`, set `hidden_states=(hidden_states,)`
  in the return value instead of propagating the full `outputs.hidden_states`
  tuple from the inner model. The caller always accesses `[-1]`, so the API
  is unchanged.
- **Savings:** ~460 MB for M1 (16 intermediate hidden-state tensors freed at
  wrapper return instead of surviving until `_train_step` return). Negligible
  for M3 (phase 2 is only ~64 tokens).
- **Risk:** None. Only affects `skip_lm_head=True` callers, which is
  exclusively `_train_step`.

### O2. Delete `outputs` early in `_train_step`

- **File:** `grad_lora_finetuning/trainer.py` -- `_train_step()`
- **What:** After extracting `hidden_states` and `past_key_values` from
  `outputs`, delete `outputs` immediately. Update ANLYS-01 logging and return
  value to use the extracted `_past_kv` variable.
- **Savings:** Frees attention weights, query states, and any other output
  fields. ~0-50 MB depending on whether attentions are materialized.
- **Risk:** None.

### O3. Delete intermediate tensors after use in `_train_step`

- **File:** `grad_lora_finetuning/trainer.py` -- `_train_step()`
- **What:** `del shift_hidden, shift_labels` after the chunked CE loop.
  These Python references are the last non-autograd references; deleting them
  allows earlier GC after `backward()` completes.
- **Savings:** ~29 MB (shift_hidden) freed slightly earlier.
- **Risk:** None.

### O4. Use `torch.inference_mode()` for eval forward passes

- **File:** `grad_lora_finetuning/trainer.py` -- `train()`
- **What:** Replace `with torch.no_grad():` with `with torch.inference_mode():`
  at all eval call sites (baseline eval, periodic eval, final eval, debug gen).
- **Savings:** Reduced per-tensor autograd overhead during eval. ~5-10%
  reduction in eval-time transient allocations.
- **Risk:** None. Eval results are Python floats, never fed back to autograd.

### O5. Add `gc.collect()` before `torch.cuda.empty_cache()`

- **Files:** `grad_lora_finetuning/trainer.py`
- **What:** Pair every `torch.cuda.empty_cache()` with a preceding
  `gc.collect()` to ensure cyclic Python references holding CUDA tensors
  are freed before the allocator reclaims blocks.
- **Savings:** Variable. Most impactful at training-to-eval transitions
  where large tensor shapes change.
- **Risk:** None. `gc.collect()` adds ~10-20 ms per call.

### O6. Avoid GPU clone in checkpoint save

- **File:** `grad_lora_finetuning/trainer.py` -- `_save_checkpoint()`
- **What:** Change `p.data.clone()` to `p.data.detach().cpu().clone()`
  so the state dict is built directly on CPU, avoiding a temporary GPU copy.
- **Savings:** ~4 MB temporary GPU allocation during checkpoint save.
- **Risk:** None.

### O7. Enable `pin_memory` for DataLoader

- **File:** `grad_lora_finetuning/trainer.py` -- `__init__()`
- **What:** Set `pin_memory=True` in the `DataLoader` constructor (when CUDA
  is available) and use `non_blocking=True` on `.to(device)` calls in
  `_train_step`.
- **Savings:** Faster async host-to-device transfers. No VRAM savings, but
  reduces synchronization stalls that can delay memory release.
- **Risk:** None. Slightly higher host RAM usage for pinned pages.

### O8. Add cleanup between eval chunks in `evaluate_lb`

- **File:** `namm/evaluation/evaluator.py` -- `evaluate_lb()`
- **What:** Call `empty_gpu_cache()` after each successful generation chunk
  when `self.force_clear_cache` is True (default). This ensures KV caches
  and generation intermediates are returned to the allocator between samples.
- **Savings:** Reduces peak fragmentation during eval. Most impactful for
  M1 where KV caches grow to 6500 tokens.
- **Risk:** None. Adds ~20 ms per eval sample.

### O9. Recommended environment variable

- **Type:** Config / documentation only
- **What:** Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the
  launch environment. This tells PyTorch's allocator to use expandable
  segments, reducing fragmentation from incremental KV cache growth during
  generation.
- **Savings:** Reduces external fragmentation. Exact savings depend on
  allocation patterns.
- **Risk:** None (PyTorch 2.1+ only; silently ignored on older versions).

### Summary Table

| ID | File | Estimated Saving | Risk | Type |
|---|---|---|---|---|
| O1 | `llama.py` | ~460 MB (M1) | None | Code |
| O2 | `trainer.py` | ~0-50 MB | None | Code |
| O3 | `trainer.py` | ~29 MB earlier release | None | Code |
| O4 | `trainer.py` | ~5-10% eval overhead | None | Code |
| O5 | `trainer.py` | Variable (fragmentation) | None | Code |
| O6 | `trainer.py` | ~4 MB temporary | None | Code |
| O7 | `trainer.py` | Perf (async transfer) | None | Code |
| O8 | `evaluator.py` | Fragmentation reduction | None | Code |
| O9 | Environment | Fragmentation reduction | None | Config |
