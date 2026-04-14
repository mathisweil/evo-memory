# NAMM Memory Investigation — M3/M4 OOM at batch_size=2

**Date:** 2026-04-14

## 1. Symptom

At an earlier working-point, M1 (`m1_lora_5t.yaml`) trained at
~11 GB VRAM on a 24 GB 3090 Ti with `batch_size=2,
gradient_accumulation_steps=8`. M3 (`m3_lora_frozen_namm_5t.yaml`) OOM'd at
the same config — the only deltas were `namm_active: true` and
`cache_size: 1024`. M4 (`m4_joint_lora_5t.yaml`) already pinned
`lora_batch_size=1, gradient_accumulation_steps=16`, matching what the
training rules require.

## 2. Root cause — phase-2 span blowup at `batch_size > 1`

`grad_lora_finetuning/trainer.py` implements a two-phase forward to
bound NAMM-active memory:

- **Phase 1** processes context tokens up to a `chunk_align`-aligned
  answer boundary under `torch.no_grad()`. The pre-eviction KV cache is
  not stored for backward.
- **Phase 2** processes answer tokens with gradients, attending to the
  post-eviction cache from phase 1.

The dominant OOM cause at `batch_size=2` was in how the phase boundary
was chosen. The pre-fix `_train_step` computed:

```python
answer_mask = (labels[0] != -100)
context_end = (answer_start // chunk_align) * chunk_align
```

`labels[0]` uses **only sample 0**. `pad_collate_fn`
(`grad_lora_finetuning/datasets.py:301-349`) right-pads to the batch
max with per-sample `label_start`, so sample 0's `answer_start` is
independent of sample 1's. When sample 1's answer begins much later
than sample 0's — routine with heterogeneous prompts — phase 2 ends up
running over a 2000+-token span with gradients, for a two-sample
batch. That activation tape (hidden states × layers × bs + eager
attention matrices of shape `[bs, heads, q_len, kv_len]`, forced by
`output_attentions=True`) is what pushes past 24 GB.

At `batch_size=1` the split is correct by construction
(`labels[0]` is the only sample), so the prior working point held.

### Smaller contributors (still real, not dominant)

- Phase 2 used to call `memory_policy.update_cache(...)` without a
  `torch.no_grad()` wrapper. The frozen BAM scoring head's inputs
  derive from `outputs.attentions` and `past_key_values`, both
  `requires_grad=True` in phase 2; PyTorch retained the scoring
  activation tape until backward even though the NAMM weights are
  frozen and `torch.topk` is non-differentiable. Fix A (below) removes
  this overhead. It is real but was not the dominant term at
  `batch_size=2`.
- `scripts/run_lora.py:280-284` disables gradient checkpointing when
  `namm_active=True`, because the memory-policy path needs
  `use_cache=True`, which HF gradient checkpointing is incompatible
  with. M3 therefore pays activation-storage cost that M1 amortises
  away.
- `namm/llms/llama.py` forces `output_attentions=True`, which forces
  eager attention and materialises the full attention matrix per layer
  per step.

## 3. Fixes applied

### Fix C — per-sample phase-split loop in `_train_step_namm`

`grad_lora_finetuning/trainer.py` now splits the NAMM-active training
step into a per-sample loop. For each sample in the batch:

1. NAMM buffers (`memory_policy.initialize_buffers`,
   `set_params_batch_idxs(np.zeros([1]))`) are reset.
2. `answer_start` is computed from that sample's own `labels` (not
   `labels[0]`).
3. Phase 1 runs under `torch.no_grad()` up to the sample's
   `context_end`; phase 2 runs with gradients over only that sample's
   answer span.
4. The chunked CE loss is computed and `(sample_loss /
   gradient_accumulation_steps).backward()` accumulates gradients into
   `.grad`.

**Why the gradient math is preserved.** The original normalisation was
`loss = (Σ_i sample_total_loss_i) / n_tokens_total / grad_accum`, with a
single `backward()`. The per-sample version backwards
`sample_total_loss_i / n_tokens_total / grad_accum` for each sample,
using `n_tokens_total = (labels_full != -100).sum()` computed once over
the full batch. Since `.backward()` accumulates into `.grad`, the two
are identical up to floating-point summation order.

**Why this is the right fix at `batch_size=2`.** The non-NAMM path
(M1) keeps a single forward over the full batch — its activation
memory was never the problem because `labels[0]`'s phase split was
irrelevant in that branch (no phase split happens at all). The NAMM
path (M3, M4 LoRA stages) is the one where a heterogeneous batch
silently inflates phase 2 to the worst-case sample's span; looping
per-sample bounds peak phase-2 memory at the `batch_size=1`
footprint, regardless of YAML `batch_size`.

### Fix A — wrap NAMM scoring in `torch.no_grad()`

A source-level change in `namm/llms/llama.py:566-594` wraps both the
`update_cache` and `buffer_cache` branches inside `if
apply_memory_policy:` in `with torch.no_grad():`.

**Why it is safe for every training path.** The `self.model(...)`
call above (`llama.py:494-506`) has already returned
`outputs.hidden_states` and `outputs.attentions`, and those are what
feed the LoRA loss (either directly via `labels` at
`llama.py:537-547`, or via the `hidden_states[-1]` the trainer pulls
out for its chunked CE). The loss autograd graph for the current step
is complete **before** `update_cache` runs. The output it produces —
the post-eviction `past_key_values` — is only consumed cross-step or
by the retention logging, which reads only tensor shapes. Neither
path needs gradients.

**Why it is safe for NAMM training itself.** `.claude/rules/namm.md`
forbids gradient flow through the eviction step. NAMM is always
either frozen (M3) or trained by CMA-ES (M2, M4 NAMM stages) — never
by autograd. Fix A strictly decreases peak memory without changing
any gradient.

Fix A removes a real, but non-dominant, memory term (the frozen BAM
scoring activation tape). It stays in regardless of Fix C.

### Fix B (reverted) — `batch_size=1, grad_accum=16` in the YAMLs

An earlier edit pushed `m3_lora_frozen_namm_5t.yaml` and
`m1_lora_5t.yaml` to `batch_size=1, gradient_accumulation_steps=16` on
the grounds that `.claude/rules/training.md` mandates it for FAIR-01.
The user reverted M3 back to `batch_size=2, grad_accum=8` (matching
the current on-disk YAML). Fix C above is what makes that config
actually run at `batch_size=2` on a 24 GB 3090 Ti. The fairness
question — whether M1 and M3 must share per-step `batch_size` or only
effective batch — is a training-rules question, not a memory
question, and is out of scope for this investigation. The on-disk
YAMLs are the source of truth.

## 4. Memory budget — per-step footprint after Fix C

Peak activation memory in the NAMM path is now bounded by the
single-sample phase-2 span regardless of YAML `batch_size`. The
`batch_size` knob controls how many sequential per-sample backwards
are accumulated before the optimizer step, not how much lives in the
autograd graph at once.

At `cache_size=1024`, bf16, 16 layers, 8 kv-heads, head_dim=64, one
sample in the graph:

| Component | Size | Notes |
|---|---|---|
| Llama-3.2-1B weights | ~2.5 GB | bf16 |
| LoRA A/B (r=8) | ~25 MB | q_proj + v_proj only |
| Optimizer state (AdamW) | ~100 MB | 2 moments × fp32 over LoRA params |
| Phase-1 post-eviction KV cache | ~34 MB | 1024 × 16 × 2 × 8 × 64 × 2 B |
| Phase-2 graph (single sample, ~64 tokens) | <100 MB | hidden states + eager attention |
| Chunked CE, chunk=512 | ~250 MB | one chunk's logits in fp32 (128k vocab) |

Peak ~4 GB. The pre-fix failure mode was sample 1 having an
`answer_start` much later than sample 0, so phase 2 ran with gradients
over a ~2000-token span for both samples simultaneously — the eager
attention tensor alone
(`[bs=2, heads=8, q_len=2000, kv_len=3000]` in bf16) was ~6 GB,
plus hidden states and BAM scoring activations.

## 5. Impact on existing results

None. This is a training-memory issue, not a correctness issue.

- Any M1 / M3 run that completed at `batch_size=2, grad_accum=8` used
  effective batch 16, the same as `batch_size=1, grad_accum=16`.
  Gradients are mathematically identical up to floating-point
  summation order.
- No existing evaluation in `results/main_table_5t/` needs to be
  re-run. The headline F1 numbers stand.

## 6. Decision summary

- **M1 YAML**: on-disk config is the source of truth.
- **M3 YAML**: on-disk config is the source of truth. Current state
  `batch_size=2, grad_accum=8` now runs without OOM on a 24 GB 3090
  Ti because the NAMM path loops per sample.
- **M4 YAML**: `lora_batch_size=2, gradient_accumulation_steps=8` —
  picks up the same fix automatically via the shared trainer.
- `grad_lora_finetuning/trainer.py` — NAMM path now in
  `_train_step_namm`, per-sample phase-split loop. Non-NAMM path
  unchanged.
- `namm/llms/llama.py:566-594` — `update_cache` / `buffer_cache`
  inside `apply_memory_policy` run under `torch.no_grad()`.
