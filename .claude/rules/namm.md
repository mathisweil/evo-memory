---
description: Rules for NAMM training, eviction policies, and Hydra config plumbing
paths:
  - scripts/run_namm.py
  - scripts/check_eviction_stats.py
  - scripts/profile_namm.py
  - namm/**
  - config/policy/**
  - config/run/namm_*.yaml
  - config/run/full_cache_*.yaml
  - config/run/recency_*.yaml
  - config/evolution/**
  - config/config.yaml
---

# NAMM rules (eviction policy training)

## Hydra invocation

- `run_namm.py` is Hydra-driven. You MUST use the `run@_global_=<preset>` override syntax — plain `run=<preset>` raises a Hydra error because the run config is mounted at `@_global_` scope in `config/config.yaml`.
  - Correct: `python scripts/run_namm.py 'run@_global_=namm_bam_i1_llama32_1b_5t'`
  - Incorrect: `python scripts/run_namm.py run=namm_bam_i1_llama32_1b_5t`
- The 5-task QA preset is `namm_bam_i1_llama32_1b_5t`. The non-`_5t` variant uses a different task mix and MUST NOT be substituted in M-condition runs.
- Hydra writes to `outputs/{date}/{time}/`. You MUST NOT redirect this to `experiments/...` from inside the script — `experiment_utils.py` reorganises outputs after the fact.

## Eviction modes

NAMM supports two eviction rules. They are not interchangeable.

- **Top-k (default, `threshold_only=false`)**: keeps the `cache_size` highest-scoring tokens — hard budget enforced every step. This is what M1–M4 use.
- **Threshold-only (`threshold_only=true`)**: evicts tokens with score `s_i < 0`; cache size varies per step. Matches the original Cetin et al. NAMM paper but is NOT the FAIR-01 setting.

If you set `threshold_only=true`, you MUST also set `scoring_initializer=2`.

- **Why:** With the default `scoring_initializer=0`, the CMA-ES mean starts at the eviction boundary (score=0). The first perturbation pushes every token below zero and the policy collapses to evict-everything before learning anything.
- `scoring_initializer=2` lifts every token above threshold so CMA-ES has room to learn selective eviction before the threshold is first crossed.

## Buffer vs split-filter sizing

`max_conditioning_length` and `split_max_conditioning_length` look like duplicates but they are not.

- `max_conditioning_length` sets the model's KV buffer capacity.
- `split_max_conditioning_length` controls which prompts the train/val/test splitter considers eligible.

If you override only `max_conditioning_length`, the splitter falls back to the same value and silently drops every long prompt — turning a 306-train-prompt dataset into an empty one. You MUST override both independently when shrinking the buffer.

## CMA-ES hyperparameters (M2 / M4 NAMM stages)

These come from the `namm_bam_i1_llama32_1b_5t` preset and have already been tuned. You MUST NOT change them unless the user explicitly asks.

- `pop_size=8`
- `elite_ratio=0.5`
- `init_sigma=0.065`
- `memory_policy_fixed_delay=256`
- `max_memory_length=1024` (4× compression vs the 4096-token min context)
- `samples_batch_size=8` prompts per task per CMA-ES step
- `batch_size=4` sequences per GPU forward pass
- `max_iters=200` generations
- `seed=1337` (paired with LoRA `seed=42` for FAIR-01)

## What MUST NOT change in NAMM modules

- You MUST NOT modify `namm/policy/` scoring heads to add bias terms, gating, or "stability" tricks unless the user explicitly asks. The `BAM-i1` architecture is the experimental baseline; ad-hoc changes break comparison with the M2 GCS checkpoints.
- You MUST NOT add gradient flow through the eviction step. Token selection is non-differentiable by design — that is precisely why the codebase trains it with CMA-ES rather than SGD. If you find yourself wanting `torch.where` with gradients, stop and re-read the NAMM paper.
- You MUST NOT shorten `max_iters` for "faster iteration" — the M2 checkpoints converge late (best val F1 around iter 105 for cs1024). Truncating loses the actual best policy.
- You MUST NOT change `save_checkpoint_every` from `null` (save-every-iter) without warning the user. The NAMM trainer relies on `latest.pt` reflecting the most recent generation for resume-after-preemption on TPU spot VMs.

## Diagnostics

When eviction looks broken, use `scripts/check_eviction_stats.py` (pass `--cache_size 0` for threshold-mode checkpoints) before editing policy code. The first question is always "is the policy actually evicting anything", not "is the policy scoring correctly".
