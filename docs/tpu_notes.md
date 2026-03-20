# TPU Notes (v6e-8, torch_xla 2.9.0)

## Hardware
- **TPU v6e-8** (Trillium): 8 chips, 32 GB HBM each = 256 GB total
- Region: europe-west4-a (64-chip spot quota)
- Biggest single-host config for v6e

## XLA Compilation

### The Core Problem
XLA compiles a **new graph for every unique tensor shape**. Each compilation takes ~2-5 minutes for full model forward passes. Sources of unique shapes:

1. **NAMM threshold eviction** — `torch.min(first_above_thresh)` produces data-dependent variable-length slices. Every eviction step = new shape = new compilation.
2. **KV cache growth phase** — with `fixed_delay=256` and `cache_size=1024`, cache grows through 256 → 512 → 768 → 1024. Each is a different shape.
3. **Partial last batch** — if dataset size isn't divisible by batch_size, the last batch has a different batch dimension.
4. **Negative indexing overflow** — `tensor[..., -N:]` where N > tensor size is valid on GPU (returns full tensor) but crashes on XLA.

### Fixes Applied

**NAMM fixed-size eviction (Option B):**
- Cache stays at fixed `cache_size` after eviction
- Evicted entries are zeroed out instead of removed
- `cache_validity_mask` propagated to attention (-inf masking) and scoring (min_value masking)
- TPU-only, gated behind `is_tpu()` / `_IS_TPU` checks
- Files: `base_dynamic.py`, `deep.py`, `deep_selection.py`, `llama.py`

**Recency cache padding:**
- On TPU, cache is padded to `cache_size` during growth phase
- Eliminates 3 of 4 growth-phase shapes (only initial no-cache vs steady-state)
- Validity mask marks padded entries as invalid
- Files: `base.py` (`Recency`, `AttnRequiringRecency`)

**Negative index clamping:**
- `base.py`: `min(self.cache_size, key_cache.shape[-2])` before slicing
- `llama.py`: `min(max_cache_length + input_ids.shape[1], attention_mask.shape[1])` in `prepare_inputs_for_generation`
- `deep_selection.py`: `min(self.cache_size, attn_mask.shape[-1])` before slicing

**Empty DynamicCache guard:**
- `llama.py` line 305: `len(past_key_values) > 0` before indexing

### Batch Size
- Must be **fixed** (not "auto") so XLA graphs are deterministic
- Dataset size must be divisible by batch_size to avoid partial batch shapes
- Qasper: 180 samples → batch_size=18 (180/18=10 exact batches)
- `mini_batch_size` must equal `batch_size`

## Persistent XLA Cache

### Status: NOT WORKING
- `XLA_PERSISTENT_CACHE_PATH` writes graphs to disk but **cannot reload them**
- Error: `"Failed to deserialize executable: UNIMPLEMENTED: Deserializing serialized executable not supported."`
- Root cause: libtpu 0.0.21 has `DeserializeExecutable` internally but it's not wired through the PJRT C API plugin interface that torch_xla uses
- JAX's persistent cache works (different code path to libtpu)
- torch_xla 2.9.0 pins libtpu==0.0.21; newer libtpu requires Python 3.11+
- **No fix available** — each new Python process recompiles from scratch

### Implication
- Compilation cost is per-process, not persistent
- With fixed-size caches and batch padding, total unique shapes are bounded (~10-20 per config)
- First evaluation pass is slow (compilation), subsequent passes reuse in-memory cache

## Environment Setup
```bash
source setup/activate_tpu.sh   # sets PJRT_DEVICE=TPU, XLA env vars
```

Key env vars:
- `PJRT_DEVICE=TPU`
- `XLA_PERSISTENT_CACHE_PATH` — local dir for graph cache (writes but can't read)
- `XLA_USE_BF16=0` — disabled, using explicit dtype control

## Sequence Length Bucketing
Input sequences padded to nearest bucket: `[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]`
Limits sequence-length-related recompilations to ~9 possible shapes.
Implemented in `namm/evaluator.py`.

## Warmup Script
`scripts/warmup_xla_cache.sh` — runs 1 ES iteration per config to validate all shapes compile:
- Base model (full cache, no eviction)
- Recency × 6 cache sizes (1024-6144)
- NAMM × 6 cache sizes (1024-6144)
- 13 total runs

## GPU Compatibility
All TPU changes are gated behind `_IS_TPU = os.environ.get("PJRT_DEVICE") == "TPU"`.
GPU code paths are completely unchanged.

## Not Bit-for-Bit Identical to GPU
- NAMM: EMA/scoring buffers see zeroed-but-present evicted entries for one step before they get replaced by topk. Eviction decisions are identical since evicted entries score `min_value`.
- Recency: padded entries are masked in attention and evicted first (they're oldest = leftmost). Functionally identical.
