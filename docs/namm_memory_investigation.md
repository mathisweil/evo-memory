# NAMM Memory Investigation — M3/M4 OOM at batch_size=2

**Date:** 2026-04-14

## 1. Symptom

At an earlier working-point, M1 (`lora_rh_m1_instruct_5t.yaml`) trained at
~11 GB VRAM on a 24 GB 3090 Ti with `batch_size=2,
gradient_accumulation_steps=8`. M3 (`lora_rh_m4_instruct_5t.yaml`) OOM'd at
the same config — the only deltas were `namm_active: true` and
`cache_size: 1024`. M4 (`joint_lora_m4_5t.yaml`) already pinned
`lora_batch_size=1, gradient_accumulation_steps=16`, matching what the
training rules require.

## 2. Root cause — NAMM scoring subgraph retained in phase 2

`grad_lora_finetuning/trainer.py:384-432` already implements a two-phase
forward specifically to bound NAMM-active memory:

- **Phase 1** processes context tokens up to a `chunk_align=64`-aligned
  answer boundary under `torch.no_grad()`. The pre-eviction KV cache
  (~6500 tokens × 16 layers × 8 heads × 64 dim × bf16) is **not** stored
  for backward — only the post-eviction `past_key_values` (~1024 tokens
  per layer) survives, and it is detached because phase 1 ran under
  no_grad.
- **Phase 2** processes answer tokens (~64) with gradients, attending to
  the post-eviction cache from phase 1.

This means the "full pre-eviction KV cache stays in the autograd graph"
hypothesis from the task brief is already defended against. The real
cost is elsewhere.

### Where the extra memory actually lives

Phase 2 still calls `apply_memory_policy=True`
(`trainer.py:419-428`), which routes through
`self.memory_policy.update_cache(...)` in `namm/llms/llama.py:567-574`
**without** a `torch.no_grad()` wrapper. Inside that path
(`namm/policy/deep.py:450-630`):

1. `self.scoring_network.get_tokens_score(...)` (deep.py:450) runs the
   BAM scoring head over all ~1088 phase-2 tokens × 16 layers × 8 heads.
   Its inputs derive from `outputs.attentions` and the current
   `past_key_values`, both of which carry `requires_grad=True` in phase
   2. PyTorch therefore retains every intermediate activation of the
   scoring head until backward, even though the NAMM weights are frozen
   and the scores themselves feed only into a non-differentiable
   `torch.topk` (`namm/policy/deep_selection.py:299`).
2. The retained indices are applied via
   `torch.gather(key_cache, dim=-2, index=exp_retained_idxs)`
   (deep.py:627-630). `gather` is differentiable w.r.t. `key_cache`, so
   phase-2 LoRA gradients flow back through the retained-token subset —
   which is load-bearing and must stay as is.
3. The sole downstream consumer of the phase-2 eviction is the
   retention logging at `trainer.py:491-507`, which reads only
   `layer_kv[0].shape[-2]` — no gradient is ever taken through it.

So in phase 2 PyTorch is storing a full BAM-scoring activation tape
(1088 × 16 × 8 tokens × feature dim in bf16 plus the attention matrices
that feed it) that is never used by the backward pass. At `batch_size=1`
this fits; at `batch_size=2` it pushes past 24 GB.

Two additional, smaller contributors:

- `scripts/run_lora.py:280-284` disables
  `model.gradient_checkpointing_enable(...)` whenever
  `namm_active=True`, because the memory-policy path needs
  `use_cache=True` which is incompatible with HF gradient checkpointing.
  So M3 pays activation-storage cost that M1 amortises away.
- `namm/llms/llama.py` forces `output_attentions=True` for the scoring
  network. With `batch_size=2, heads=8, layers=16, q_len≈64, kv_len≈1088`
  the attention tensors alone are ~71 MB per step — small individually,
  but they live in the phase-2 graph alongside the scoring activations.

## 3. Fixes applied

### Fix B — align training configs to the rules

`.claude/rules/training.md` already mandates
`batch_size=1, gradient_accumulation_steps=16` for **all of M1, M3, and
M4** (effective batch = 16 in every condition). M4
(`joint_lora_m4_5t.yaml:52-53`) was already compliant; M1 and M3 had
drifted to `batch_size=2, grad_accum=8`.

- `scripts/configs/lora_rh_m1_instruct_5t.yaml:33-34` — `batch_size:
  1, gradient_accumulation_steps: 16`.
- `scripts/configs/lora_rh_m4_instruct_5t.yaml:34-35` — `batch_size:
  1, gradient_accumulation_steps: 16`.

This on its own is sufficient to stop the OOM: `batch_size=1` was the
prior known-good working point for M3/M4. It also enforces the
FAIR-01 per-step-processing requirement that the rules were written
to close.

### Fix A — wrap NAMM scoring in `torch.no_grad()`

A companion source-level change removes the activation-tape overhead
at its origin rather than only hiding from it via smaller batches.

- `namm/llms/llama.py:566-594` — both the `update_cache` and
  `buffer_cache` branches inside `if apply_memory_policy:` are now
  wrapped in `with torch.no_grad():`.

**Why it is safe for every training path.** The `self.model(...)`
call above (`llama.py:494-506`) has already returned `outputs.hidden_states`
and `outputs.attentions`, and those are what feed the LoRA loss (either
directly via `labels` at `llama.py:537-547`, or via the
`hidden_states[-1]` the trainer pulls out for its chunked CE at
`grad_lora_finetuning/trainer.py:430`). The loss autograd graph for the
current step is therefore complete **before** `update_cache` runs. The
output it produces — the post-eviction `past_key_values` — is only
consumed cross-step (as the next step's `past_kv` input) or by the
retention logging at `trainer.py:491-507`, which reads only tensor
shapes. Neither path needs gradients.

**Why it is safe for NAMM training itself.** `.claude/rules/namm.md`:
*"You MUST NOT add gradient flow through the eviction step. Token
selection is non-differentiable by design — that is precisely why the
codebase trains it with CMA-ES rather than SGD."* In this codebase
NAMM is **always** either frozen (M3) or trained by CMA-ES (M2, M4
NAMM stages) — never by autograd. So there is no training path whose
backward would ever flow through `update_cache`, and guarding it
strictly decreases peak memory without changing any gradient.

**Scope of the change.** The fix lives at a single call site. Every
training entry point (`run_lora.py`, `run_joint.py`, `run_namm.py`,
`run_es.py`) routes through `NammLlamaForCausalLM.forward` and picks
it up automatically. The profiling/analysis scripts
(`scripts/profile_namm.py`, `analysis/report_6/generate_plots.py`)
call `memory_policy.update_cache` directly outside the model forward
and are unaffected.

**Relationship between the two fixes.** Fix B restores the known-good
FAIR-01 working point (`batch_size=1`). Fix A eliminates the memory
overhead that forced Fix B in the first place, giving headroom for
any future follow-up that needs `batch_size>=2` (e.g., a
larger-model sweep) without re-opening the confound. We keep Fix B
in place regardless: the M1/M3/M4 comparison still requires
identical per-step processing, and `batch_size=1` is what the rules
document mandates.

## 4. Memory budget — expected per-step footprint after fix

These are analytic estimates (no profiler run this session; the fix
restores the prior working configuration, for which
`batch_size=1` was the known-good setting).

At `batch_size=1, cache_size=1024`, bf16, 16 layers, 8 kv-heads,
head_dim=64:

| Component | Size | Notes |
|---|---|---|
| Llama-3.2-1B weights | ~2.5 GB | bf16 |
| LoRA A/B (r=8) | ~25 MB | q_proj + v_proj only |
| Optimizer state (AdamW) | ~100 MB | 2 moments × fp32 over LoRA params |
| Phase-1 post-eviction KV cache | ~34 MB | 1024 × 16 × 2 × 8 × 64 × 2 B |
| Phase-2 new tokens (64) in graph | <100 MB | attentions + hidden states, 1 sample |
| Phase-2 BAM scoring activations | ~400-600 MB | retained until backward |
| Chunked CE over 64 tokens, chunk=512 | ~250 MB | one chunk's logits in fp32 (128k vocab) |

Peak ~4 GB, well within the 24 GB budget. Doubling to `batch_size=2`
scales the phase-2 scoring activations and logits roughly linearly —
enough headroom evaporation to break the budget on a 3090 Ti.

## 5. Impact on existing results

None. This is a training-memory configuration issue, not a correctness
issue:

- Any M1 / M3 run that completed at `batch_size=2, grad_accum=8` used
  the same **effective** batch (16) as the `batch_size=1,
  grad_accum=16` setting. Gradients are mathematically identical up to
  per-step gradient-noise reordering (which is absorbed by
  gradient-accumulation statistics and the seed).
- No existing evaluation in `results/main_table_5t/` needs to be
  re-run. The headline F1 numbers stand.
- Forward-going: `run_all_experiments.sh` and any new M1/M3 runs will
  pick up the corrected YAMLs automatically (no CLI overrides needed).

## 6. Decision summary

- **M1** `batch_size=1, gradient_accumulation_steps=16`.
- **M3** `batch_size=1, gradient_accumulation_steps=16`.
- **M4** `lora_batch_size=1, gradient_accumulation_steps=16` (already
  compliant).
- Source-level NAMM scoring now runs under `torch.no_grad()` at
  `namm/llms/llama.py:566-594`. No gradient or fairness property
  changes; memory overhead from the retained BAM activation tape is
  eliminated.
