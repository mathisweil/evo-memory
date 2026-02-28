# Changelog — evo-memory modifications

All changes made on top of the original SakanaAI evo-memory repo, on the
`tiny_llama_implementation` branch.

---

## Commit `84b650e` — Profiling, Qasper bench, context-length filtering

### New files

- **`profile_log.py`** — Lightweight profiling logger. When `PROFILE_TIME=1` is
  set, all `[PROFILE]` lines are written to both stdout and a timestamped file
  under `profiling/` (configurable via `PROFILE_LOG_DIR`). File is opened lazily
  on first write.

- **`check_qasper.py`** — Quick script to inspect the Qasper dataset structure.

- **`cfgs/task/qasper.yaml`** — Task config for the Qasper QA benchmark.

- **`cfgs/run/tinyllama_qasper_namm_1024cs_trial.yaml`** — Trial run config
  (short `max_iters`) for timing experiments without polluting real runs.

- **`cfgs/run/tinyllama_qasper_*.yaml`** — Run configs for TinyLlama on Qasper
  with various memory policies (full cache, recency, NAMM 1024cs, NAMM no-cs,
  sweep).

- **`cfgs/run/namm_bam_eval_baseline_tinyllama.yaml`** — Baseline eval config.

- **`cfgs/policy/passthrough.yaml`** — Passthrough (no-op) memory policy config.

### Modified files

#### `memory_evaluator.py`
- Added `PROFILE_TIME=1` gated per-prompt profiling: tokenization time,
  generation time, ms/token, context length, batch size.
- Added `DISABLE_TQDM=1` env var to suppress the eval progress bar.
- Fixed crash when `RANK` env var is unset (defaulting to 0).

#### `memory_trainer.py`
- Added `PROFILE_TIME=1` gated profiling for: `pop_sampling`, `pop_evaluate`
  (per accumulation step with total/per-prompt timing), and `evolution_update`.
- **Bug fix**: added guard when checkpoint `iter_num` exceeds `max_iters` —
  resets `start_iter` to 0 with a warning instead of producing an empty
  `range()`.
- **Bug fix**: removed stray `iter_num += 1` after the training loop that
  crashed with `UnboundLocalError` when the loop body never executed.

#### `memory_policy/base.py`
- Added `import time` and `from profile_log import plog` (used by profiling in
  previous iteration; per-eviction logging was later removed to reduce noise).

#### `memory_policy/deep.py`
- Added `import time` and `from profile_log import plog`.
- Removed verbose per-layer NAMM profiling (embed/score/select/gather per
  layer per eviction event) to keep output condensed.

#### `task_sampler.py`
- Added `max_conditioning_length` parameter to `TaskSampler.__init__()`.
- Added context-length filtering in `init_tasks()`: examples whose word count
  exceeds `max_conditioning_length / 1.3` are dropped, with a summary printed
  (e.g., `Length filter lb/qasper: 200 -> 181 examples`).

#### `cfgs/run/base_memory_policy/deep/bam/base_bam.yaml`
- Minor config adjustment.

#### `cfgs/run/namm_bam_eval.yaml`
- Minor config adjustment.

#### `cfgs/task/base_sampler.yaml`
- Added `max_conditioning_length` field.

---

## Uncommitted changes — Batch parallelism + GPU memory logging

### `stateless_parallel_modules/attention.py`

**Enables `batch_size > 1` for the NAMM scoring network**, allowing multiple
prompts to be evaluated simultaneously in a single forward pass.

**Problem**: With `batch_size=1`, the 32 forward passes per training step
(8 pop members × 4 prompts) run sequentially. Setting `batch_size > 1` crashed
because the NAMM scoring network's `StatelessAttention` flattens batch dims
(`[bs, n_heads]` → `[bs*n_heads]`) but the `attn_mask` was not expanded to
match.

**Root cause**: `attn_mask` has shape `[bs, 1, T]` (one mask per sequence,
singleton head dim). After the QKV states flatten `[bs, n_heads]` →
`[total_batch]`, the mask's batch dim `bs` no longer matches `total_batch =
bs * n_heads`. The `logical_or` between the flattened mask and the causal mask
`[T, T]` then fails on dimension mismatch.

**Fix** (applied to both `StatelessAttention.forward()` and
`MonoHeadStatelessAttention.forward()`):

```python
if attn_mask is not None and len(batch_dims) > 1:
    total_batch = qkv_states.shape[0]
    mask_batch = attn_mask.shape[0]
    if mask_batch > 1 and mask_batch != total_batch:
        repeat = total_batch // mask_batch
        # [bs, 1, T] -> [bs, repeat, T] -> [total_batch, 1, 1, T]
        attn_mask = (attn_mask.expand(-1, repeat, -1)
                     .reshape(total_batch, 1, attn_mask.shape[-1])
                     .unsqueeze(-2))
```

The 4D output `[total_batch, 1, 1, T]` ensures that `logical_or` with
`causal_mask [T, T]` produces `[total_batch, 1, T, T]`, which matches sdpa's
expected mask shape `[batch, heads, query_len, key_len]`.

The `mask_batch > 1` guard leaves `batch_size=1` untouched (PyTorch's native
broadcasting handles that case).

**Usage**: set `batch_size=4` (or 8, 16, ...) on the command line:
```bash
python3 main.py ... batch_size=4
```

### `memory_evaluator.py`

Added GPU memory reporting to the per-prompt profiling line:

```python
_mem_mb = torch.cuda.memory_allocated() / 1024**2
_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
```

Output now includes `gpu=XXXXmb peak=XXXXmb` to help determine the maximum
safe `batch_size` for a given GPU.

---

## Environment notes

- **`transformers==4.41.2`** is required — 4.45+ introduces breaking
  `DynamicCache` API changes.
- **`peft==0.10.0`** is required — newer versions depend on
  `EncoderDecoderCache` (only in transformers 4.45+).
- **`PROFILE_TIME=1`** — enables all profiling instrumentation.
- **`DISABLE_TQDM=1`** — suppresses eval progress bars.
- **`HF_HOME=...`** — redirects HuggingFace model cache to avoid home dir
  quota issues.
