# Claude Code Task: NAMM Implementation Efficiency Audit

## Background

The `namm/` folder contains the implementation of Neural Attention Memory Models (Cetin et al., ICLR 2025). The NAMM scores every token in the KV cache using a learned neural network (`mφ`) and evicts those with score `s_i < 0`. It runs once every `nup` tokens (the **fixed delay**, default 256–512). The paper authors explicitly state:

> *"the main objective of our work was to provide performance benefits — we did not particularly optimize our code for memory efficiency or speed"*

The implementation is therefore almost certainly sub-optimal in ways that compound hard during training, where the NAMM is called `pop_size × samples_batch_size × (max_conditioning_length / nup)` times per CMA-ES generation. At our settings (pop=8, batch=16, context=4086, delay=256) that is **~2048 NAMM forward passes per generation**, across 300 generations. Any inefficiency in the inner loop is multiplied by ~614,400.

The goal of this task is to **audit the implementation, measure where time actually goes, and apply targeted optimisations** — without changing any public API or altering model behaviour.

---

## Step 1 — Read and map the codebase (do this before profiling or changing anything)

1. List all files under `namm/` and understand the module structure.
2. Read the following in order:
   - `namm/policy.py` (or equivalent) — the `ParamMemoryPolicy`, `SelectionCriteria`, and any `BAM`/`MLP` network classes.
   - `namm/features.py` (or equivalent) — the STFT spectrogram extraction and EMA reduction logic.
   - `namm/llms.py` (or equivalent) — `MemoryModelWrapper`, and specifically how the NAMM is invoked during the LLM forward pass.
   - `namm/evaluation.py` — `MemoryHFEvaluator` and how it orchestrates evaluation.
   - `namm/evolution.py` — `MemoryEvolution`, i.e. the CMA-ES loop.
3. For each file, record:
   - What tensors are allocated inside the inner loop (per chunk, per token, per layer).
   - Where Python loops iterate over things that could be batched.
   - Where `.cpu()`, `.numpy()`, or explicit host-device transfers occur.
   - Where `torch.no_grad()` is and is not used.
   - Whether `torch.compile` has been applied anywhere.

Produce a **written map** of the full execution pipeline before writing any code.

---

## Step 2 — Profile: find the actual bottlenecks

Do not guess. Measure.

### 2a. Wall-clock breakdown

Instrument the code with `torch.cuda.synchronize()` + `time.perf_counter()` to measure wall-clock time for each of these stages across a single NAMM call (one eviction step, one layer, one sequence):

| Stage | What to measure |
|---|---|
| Feature extraction | STFT computation per token column |
| EMA reduction | Reducing the time axis of the spectrogram |
| Positional embedding concat | Building the final feature vector `ω_i` |
| BAM / MLP forward | `mφ(ω_i)` — the scoring network |
| Score thresholding / topk | `selection_criteria` — eviction decision |
| KV cache reindexing | Physically removing evicted tokens from cache |
| EMA state update | Writing `ω→_i ← ω_i` for retained tokens |

Print a summary table. Then repeat across realistic cache sizes (256, 512, 1024, 2048 tokens) to check how each stage scales.

### 2b. PyTorch profiler

Run `torch.profiler.profile` with `record_shapes=True` and `with_stack=True` for a single full sequence (prefill + 1 NAMM eviction step). Export the trace and report the top-10 operators by self-CUDA time. This is the ground truth — follow it, not intuition.

### 2c. Memory allocation trace

Use `torch.cuda.memory_allocated()` before and after each stage to find stages that allocate large intermediates that are immediately discarded. Identify any allocations inside Python loops that should be pre-allocated once.

---

## Step 3 — Efficiency audit checklist

Work through each item below. For each one: determine whether the issue exists in the current code, estimate its impact, and note whether it is safe to fix.

### 3.1 STFT / spectrogram extraction

- [ ] **Is the STFT vectorized over tokens?** The paper processes each attention column `A[:, i]` independently. If this is a Python loop over `i`, it should be replaced with a single batched `torch.stft` call over the entire attention matrix at once — `torch.stft` accepts a batch dimension.
- [ ] **Is the Hann window pre-computed and reused?** If `torch.hann_window(nw)` is called inside the loop, move it to module initialisation.
- [ ] **Is the magnitude computed in float32 unnecessarily?** The complex STFT output can be converted to magnitude with `torch.abs()` in the same dtype as the computation. If there are explicit `.float()` casts on the intermediate, check if they are needed.
- [ ] **Is the attention matrix sliced or copied unnecessarily?** If `A[:, i]` creates a new tensor rather than a view, consider using `A.T` or `A.contiguous()` once before the loop.

### 3.2 EMA state

- [ ] **Is the EMA update in-place?** `ω_i ← Σ γ^t ω^t_i + γ^nT ω→_i` should be computed in-place (`mul_` + `add_`) to avoid allocating a new tensor per token per step.
- [ ] **Is the EMA state stored for evicted tokens?** After eviction, the EMA state for evicted tokens must be pruned to match the surviving KV cache. Check whether this pruning is done with a gather/index_select (fast) or a Python loop constructing a new list (slow).
- [ ] **Is the EMA state stored contiguously in memory?** After repeated evictions and appends, the EMA buffer may become non-contiguous. Check if `.contiguous()` is called unnecessarily often, or never called when it should be.

### 3.3 BAM self-attention (the scoring network)

- [ ] **Is the BAM self-attention (`attention_M̃(KΩ, VΩ, QΩ)`) batched across all tokens in one call?** It should be — the backward (counter-causal) mask is static and can be pre-built once for `max_memory_length` tokens, then sliced to the current cache size `[N, N]`. If it is rebuilt every call, that is wasted work.
- [ ] **Is the mask created on CPU and then moved to GPU?** Move it to GPU at initialisation.
- [ ] **Is the BAM's linear layer applied per-token in a loop?** It should be a single `nn.Linear` applied to the stacked feature matrix `[N, D]`.
- [ ] **Can the BAM self-attention use `F.scaled_dot_product_attention`?** This dispatches to FlashAttention when available and is significantly faster than a manual `(Q @ K.T / sqrt(d)) @ V` implementation. Check if the backward mask structure is compatible (it is a standard causal mask, just reversed — PyTorch's `is_causal` flag won't work but a custom `attn_mask` will).
- [ ] **Is residual addition done with `+` (allocating a new tensor) or `add_`?** For small networks running millions of times, this matters.

### 3.4 Eviction / KV cache reindexing

- [ ] **How are evicted tokens removed from the KV cache?** The correct approach is a single `index_select` (or fancy indexing) on the cache tensor along the sequence dimension using the boolean mask or surviving indices. If it uses a Python list comprehension or a loop over layers, that is the most expensive possible implementation.
- [ ] **Is the eviction mask computed once and applied to all layers simultaneously?** The NAMM runs on one layer's attention matrix to produce a score, but the same eviction mask should be applied to all layers' KV caches in a single batched operation. Check whether this is done, or whether there is a separate eviction call per layer.
- [ ] **Is there an unnecessary `.clone()` of the KV cache before eviction?** If the KV cache is cloned defensively before indexing, remove it.

### 3.5 Python overhead and dispatch cost

- [ ] **Are there Python loops over layers?** The NAMM is applied to each layer's attention separately. If the feature extraction or eviction is called in a Python `for layer in layers:` loop, the GPU is idle during the Python dispatch between iterations. Measure the ratio of Python overhead to GPU time using the profiler. If Python overhead is >20% of total time, consider fusing the per-layer calls.
- [ ] **Is `torch.no_grad()` applied consistently throughout the NAMM forward pass?** Since the NAMM is evolved via CMA-ES (not backpropagated through), gradients are never needed. Confirm `@torch.no_grad()` or `with torch.no_grad():` wraps every NAMM forward call and is not accidentally absent in any code path triggered during training.
- [ ] **Is `torch.compile` applicable?** The BAM and MLP scoring networks are small, static-shaped networks — ideal candidates for `torch.compile(mode='reduce-overhead')`. The eviction step (dynamic shape due to variable cache size) is not a good candidate. Check if compile has been tried, and if not, add it to the scoring network only.

### 3.6 Population-level batching (training-specific)

- [ ] **Are population members evaluated sequentially?** During CMA-ES training, `pop_size` candidate NAMM parameter sets are each evaluated on `samples_batch_size` sequences. If evaluation is sequential (one candidate at a time), there is no opportunity for GPU parallelism across candidates. Check if the `MemoryEvolution` or `MemoryTrainer` evaluates multiple candidates simultaneously by swapping NAMM weights and batching their LLM forward passes.
- [ ] **Is the LLM forward pass re-run in full for every population member?** The LLM (frozen during NAMM training) produces the same hidden states regardless of NAMM parameters, up to the first eviction step. Consider whether prefix KV-cache states (before the first eviction) can be cached and reused across all pop members for the same sequence — this would eliminate `pop_size - 1` full prefill passes per sequence per generation.
- [ ] **Are the same task sequences resampled or reused across population members within a generation?** If each population member sees a different random batch, the fitness signal is noisier. More importantly, it prevents the prefix-caching optimisation above.

### 3.7 Memory layout and dtype

- [ ] **Are all NAMM parameters and buffers in the same dtype as the LLM?** Unnecessary dtype casting (e.g., the NAMM running in float32 while the LLM is in bfloat16) causes implicit conversions on every call.
- [ ] **Is the attention matrix extracted with `.detach()`?** The STFT input is the attention matrix from the LLM. If it is not detached, the autograd graph is unnecessarily retained.
- [ ] **Is positional embedding stored as a buffer or recomputed every call?** Sinusoidal positional embeddings are a function of position index only and should be pre-computed up to `max_memory_length` and stored as a `register_buffer`.

---

## Step 4 — Apply optimisations

For each issue identified as real (i.e., confirmed by measurement to be costly, not just theoretically wasteful), implement the fix. Prioritise in this order:

1. **Python loop → batched tensor op** (typically 10–100× speedup for the affected stage).
2. **Pre-allocation / buffer reuse** (eliminates GC pressure and fragmentation).
3. **FlashAttention dispatch for BAM** (2–4× speedup on the scoring network if not already used).
4. **`torch.compile` on the scoring network** (typically 1.5–2× for small MLPs).
5. **Prefix KV-cache reuse across pop members** (large training speedup if feasible — validate correctness carefully).

For each change:
- Add a comment explaining what was changed and why.
- Run the profiler again to confirm the speedup is real.
- Run a correctness check: confirm the output scores `s_i` are identical (within float tolerance) before and after the change for a fixed input.

---

## Step 5 — Benchmark and report

After applying all safe optimisations, produce a **benchmark report** in the following format:

```
NAMM Efficiency Audit — Results
================================

Hardware: <GPU model>
Model: <LLM name>
Cache size: 1024 tokens
Sequence length: 4086 tokens (nup=256, → 16 NAMM calls per sequence)

Stage-level breakdown (per NAMM call):
---------------------------------------
Stage                    | Before (ms) | After (ms) | Speedup
-------------------------|-------------|------------|--------
STFT feature extraction  |             |            |
EMA reduction            |             |            |
Positional embed concat  |             |            |
BAM / MLP forward        |             |            |
Score thresholding       |             |            |
KV cache reindexing      |             |            |
EMA state update         |             |            |
TOTAL per NAMM call      |             |            |

End-to-end training estimate (300 generations):
-------------------------------------------------
Before: X hours
After:  Y hours
Saving: Z hours (N%)

Key changes applied:
1. ...
2. ...

Changes NOT applied and why:
1. ...
```

---

## Constraints

- **Do not change the NAMM architecture** (BAM structure, feature dimensionality, EMA formula, eviction rule). Efficiency gains must come from implementation, not from architectural changes.
- **Do not change any public API** — function signatures in `namm/` must remain the same so that `run_namm.py`, `run_eval.py`, and other scripts continue to work without modification.
- **Do not change checkpoint format** — any NAMM checkpoint saved before this change must still load correctly.
- **Validate correctness for every change** — confirm that output scores are numerically identical (or within `atol=1e-5` for float16 operations) before and after each optimisation.
- **Prefix KV-cache reuse** (Step 3.6, item 2) is a high-risk change — only implement it if correctness can be rigorously verified, and gate it behind a config flag (`cache_prefix_reuse: false` by default).

---

## Definition of Done

- [ ] Profiling output produced for baseline implementation (top-10 ops by CUDA time).
- [ ] All 3.1–3.7 checklist items assessed (exists / does not exist / not applicable).
- [ ] At least items 3.1 (STFT batching), 3.3 (BAM mask pre-build), 3.4 (eviction reindexing), and 3.5 (no_grad coverage) are either fixed or confirmed to already be correct.
- [ ] Correctness validated for all changes with a fixed-seed forward pass comparison.
- [ ] Benchmark report produced showing before/after wall-clock time per stage.
- [ ] No public API signatures changed.
- [ ] All existing tests (if any under `tests/`) still pass.
