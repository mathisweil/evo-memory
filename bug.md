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

## Bug 2: NAMM eval runs unnecessarily during non-NAMM LoRA training — OPEN

### What it is

When `run_lora.py` runs with `--namm_active false`, the LoRA training loop correctly skips NAMM eviction. But the F1 evaluation path (`_evaluate_f1` → `task_sampler.evaluate` → `evaluator.evaluate_lb` → `model.generate`) still runs the full NAMM policy during text generation, because the evaluator uses whatever `memory_policy` was loaded by `make_eval_model` from the Hydra config.

This makes baseline and periodic eval extremely slow — each val sample goes through NAMM's STFT + scoring network at every generation step, even though NAMM isn't being trained.

### Confirmed behaviour

The full `run_lora.py` end-to-end test (with `--namm_active false`, `--eval_interval 2`) ran for 20+ minutes without crashing (dtype fixes are working) but never completed baseline eval. The output was an endless stream of `WARNING: unable to normalize due to empty buffers` — the NAMM policy was running with no loaded checkpoint, so normalization buffers were empty and every generation step triggered the warning at every layer.

The process was killed manually. It was not hung — just extremely slow because every generated token passes through the full NAMM forward path (STFT + scoring network + selection) at every layer, even though NAMM is not being trained.

### Impact

A baseline eval on 17 val samples takes 20+ minutes instead of ~1-2 minutes. This makes the `eval_interval` setting impractical for iterative development. The training loop itself runs fine and fast.

### Likely fix

Either:
- Swap the memory policy to `Recency` (no-op passthrough) before eval when `namm_active=False`
- Add a flag to `model.generate()` to skip the memory policy
- Have `run_lora.py` construct the evaluator with `apply_memory_policy=False`

This is a design decision for the LoRA trainer, not a dtype/correctness bug.
