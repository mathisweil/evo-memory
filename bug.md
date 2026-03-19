## Bug 1: NAMM bfloat16 dtype mismatches on GPU — FIXED

### What it was

When `model.generate()` runs with the full NAMM policy active on GPU in bfloat16, two dtype mismatches crashed the forward pass:

1. `torch.stft()` in `deep_embedding_spectogram.py` — cuFFT doesn't support bfloat16
2. `torch.baddbmm()` in `namm/modules/base.py` — CMA-ES params (float32) mixed with bfloat16 embeddings

### Root cause

The tpu branch casts the model to bfloat16 via `model.to(dtype=torch.bfloat16)`. This works on TPU (XLA promotes dtypes transparently) but fails on GPU where CUDA ops are strict about matching dtypes. Two specific issues:

- **STFT**: cuFFT has no bfloat16 kernel.
- **CMA-ES params**: Stored as plain tensor attributes (not registered nn.Parameters), so `model.to(bfloat16)` doesn't cast them. When the STFT output is cast back to bfloat16 but the scoring network weights are still float32, `baddbmm` fails.

### Fix applied (commit f8ecf6d)

Two surgical changes:

1. **`deep_embedding_spectogram.py`**: Cast STFT input/window to float32, cast output back to `orig_dtype` after `.abs()` (or `.view_as_real()`). Only the single STFT op runs in float32.

2. **`namm/modules/base.py`**: Cast `weight`/`bias` to match `input.dtype` before `baddbmm`. This aligns the unregistered CMA-ES params to the model's running dtype. Direction matters — casting params to match input (not vice versa) keeps the model dtype consistent through the rest of the forward pass.

### Failed approaches

- Casting STFT output to `orig_dtype` on the complex tensor (before `.abs()`) discarded the imaginary part.
- Wrapping `generate()` in `autocast(float32)` promoted the entire LLM forward to float32, doubling memory (22GB vs 11GB) and halving throughput.
- `autocast(bfloat16)` conflicted with the existing `@custom_fwd(cast_inputs=float32)` decorator on `update_cache`, which disables autocast inside the function.

---

## Bug 2: NAMM eval runs unnecessarily during non-NAMM LoRA training — FIXED

### What it was

When `run_lora.py` runs with `--namm_active false`, the LoRA training loop correctly skips NAMM eviction. But the F1 evaluation path (`_evaluate_f1` → `task_sampler.evaluate` → `evaluator.evaluate_lb` → `model.generate`) still ran the full NAMM policy during text generation, because the evaluator used whatever `memory_policy` was loaded by `make_eval_model` from the Hydra config.

This made baseline and periodic eval extremely slow — each val sample went through NAMM's STFT + scoring network at every generation step, even though NAMM wasn't being trained.

### Root cause

`make_eval_model()` always instantiates the full NAMM policy from the Hydra config (e.g. `namm_bam_i1_llama32_1b` → Deep policy with STFT + BAM scoring + binary selection), regardless of whether `--namm_active` is true or false. The evaluator and model were constructed with this heavy policy attached, so every `model.generate()` call during F1 eval ran the full scoring pipeline — with no loaded checkpoint, triggering `WARNING: unable to normalize due to empty buffers` at every layer for every token.

### Fix applied (scripts/run_lora.py)

After `make_eval_model()` constructs the model, if `namm_active=False`, three things are fixed:

**1. Swap policy to Recency (passthrough):**

```python
if not args.namm_active:
    from namm.policy.base import Recency
    recency_policy = Recency(cache_size=args.cache_size)
    memory_evaluator.swap_memory_policy(recency_policy)
    memory_policy = recency_policy
```

`swap_memory_policy()` on the evaluator propagates to the inner model, so one call handles both. `Recency.requires_attn_scores` is `False`, so all STFT/scoring branches in the forward path are skipped. When `cache_size=None` (the default), Recency is a pure identity — KV cache passes through unchanged.

When `namm_active=True`, nothing changes — the full Deep policy stays in place.

**2. Fix `max_memory_length` mismatch:**

The Hydra config (`namm_bam_i1_llama32_1b`) sets `max_memory_length=1024`, tuned for NAMM's small KV cache. Without NAMM eviction, the KV cache grows to the full context length (~6500 tokens). The auto batch-size detector calibrates against `max_memory_length=1024`, picks a batch size that's too large, then actual generation with ~6500-token contexts OOMs or thrashes memory.

```python
    memory_evaluator.max_memory_length = memory_evaluator.max_conditioning_length
```

**3. Disable auto batch-size detection:**

`batch_size="auto"` runs a binary search from 64 down on every `evaluate_lb()` call, which is slow and (after fix #2) unnecessary. Set to 1 for predictable, safe evaluation:

```python
    if memory_evaluator.batch_size == "auto":
        memory_evaluator.batch_size = 1
```

Users can override with `--batch_size_eval N` if they want larger batches.

### Why this approach over the alternatives

- **"Add a flag to `model.generate()`"** — would thread a new parameter through `evaluate_lb` → `generate` → the forward loop. Invasive and fragile.
- **"Construct evaluator with `apply_memory_policy=False`"** — only addresses the evaluator, not the model. If anything else touches `model.generate()`, the problem resurfaces.
- **Swap at config level** — one call, fixes it at the root. The evaluator, the model, and any future code paths all get the Recency passthrough. The `swap_memory_policy()` infrastructure already existed.

### Verification

All 24 tests pass (`pytest tests/ -v`):

- `test_recency_swap.py` (13 tests) — Recency passthrough behaviour, swap_memory_policy correctness, run_lora.py swap logic (policy swap, batch_size override, max_memory_length fix)
- `test_lora_seam.py` (5 tests) — LoRA API correctness (module count, base weight stability, float32, round-trip)
- `test_lora_grad_trainer.py` (6 tests) — gradient flow, checkpoint resume, 10-step loss decrease

End-to-end smoke test confirmed: baseline eval on 17 qasper val samples completes successfully with `lb/avg_f1: 8.72`, followed by training with checkpoint saves and periodic eval.

---

## Full test summary

All tests run on Quadro RTX 6000 (24GB), CUDA 13.1, Python 3.9, LLaMA 3.2-1B-Instruct.

### `tests/test_recency_swap.py` — 13 tests (no GPU required)

Bug 2 fix verification. Runs without CUDA or HuggingFace auth.

| Test | What it verifies |
|------|-----------------|
| `test_recency_none_cache_is_passthrough` | `Recency(None)` returns KV cache unchanged, `requires_attn_scores=False` |
| `test_recency_with_cache_size_truncates` | `Recency(50)` keeps only last 50 tokens via slicing |
| `test_recency_has_no_parameters` | Recency has zero trainable params (no scoring network) |
| `test_swap_replaces_deep_with_recency` | `swap_memory_policy()` replaces Deep on both evaluator and model |
| `test_swap_with_cache_size_updates_max_memory_length` | `Recency(2048)` caps `evaluator.max_memory_length` |
| `test_swap_none_cache_preserves_max_memory_length` | `Recency(None)` doesn't cap `max_memory_length` |
| `test_recency_registers_with_model_config` | `finalize_registration()` creates `rotary_offset` buffer |
| `test_swap_fires_when_namm_inactive` | Swap executes when `namm_active=False` |
| `test_swap_does_not_fire_when_namm_active` | Swap skipped when `namm_active=True`, Deep policy stays |
| `test_swap_with_explicit_cache_size` | `--cache_size 2048` propagates through swap correctly |
| `test_swap_sets_batch_size_to_1` | Auto batch-size detection disabled (set to 1) |
| `test_swap_preserves_explicit_batch_size` | User-set `--batch_size_eval 4` is not overridden |
| `test_swap_updates_max_memory_length_to_conditioning` | `max_memory_length` corrected from 1024 to 6500 |

### `tests/test_lora_seam.py` — 5 tests (GPU + HF auth required)

LoRA API correctness on `WrappedLlamaForCausalLM` (LORA-04 invariants from merge plan Phase 1).

| Test | What it verifies |
|------|-----------------|
| `test_lora_module_count` | 32 LoRA modules injected (16 layers × q_proj + v_proj) |
| `test_base_weight_stability` | Base weights bit-for-bit identical after `set_lora_params()` |
| `test_lora_weights_float32_in_checkpoint` | LoRA params stored as float32 (not bfloat16) |
| `test_round_trip_injection` | `get_lora_params_flat()` → `set_lora_params()` → `get_lora_params_flat()` error = 0.0 |
| `test_set_lora_params_size_mismatch_raises` | `ValueError` on flat vector size mismatch |

### `tests/test_lora_grad_trainer.py` — 6 tests (GPU + HF auth required)

Gradient flow and trainer correctness (TRAIN-04/05/06 from merge plan Phase 4).

| Test | What it verifies |
|------|-----------------|
| `test_loss_requires_grad` | `loss.requires_grad=True` after forward (PEFT hook active) |
| `test_lora_grads_nonzero` | All LoRA params have non-None, non-zero grad after backward |
| `test_base_grads_none` | All frozen base params have `grad=None` after backward |
| `test_lora_float32` | LoRA parameter tensors are float32 |
| `test_checkpoint_resume` | Checkpoint save/load round-trips optimizer state |
| `test_loss_decreases_10_steps` | Net loss decrease over 10 gradient updates (smoke test) |

### End-to-end smoke test (`scripts/run_lora.py`)

Full pipeline run with `--namm_active false --num_epochs 1 --eval_interval 3 --max_seq_len 512`.

| Check | Result |
|-------|--------|
| Recency swap message printed | `Swapped to Recency policy (namm_active=False, cache_size=None, eval_batch_size=1)` |
| Zero NAMM warnings | No `unable to normalize due to empty buffers` in output |
| GPU memory (healthy) | 8.4 GB (vs 21 GB with broken auto-detection) |
| Baseline eval completes | `lb/qasper: 8.72`, `lb/avg_f1: 8.72` |
| Debug generation works | 3 val samples generated with predictions |
| Training starts | `=== Epoch 1/1 ===`, checkpoint saved at step 3 |
