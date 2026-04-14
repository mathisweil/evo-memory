# Claude Code Task: Investigate & Fix OOM When Training LoRA with Frozen NAMM

## Problem

M1 (LoRA only, no NAMM) trains at ~11GB VRAM with `batch_size: 2, gradient_accumulation_steps: 8`.
M3 (LoRA + frozen NAMM) OOMs with the **identical** config except `namm_active: true, cache_size: 1024`.

The NAMM is frozen — its weights don't receive gradients. So why does it blow up memory?

---

## Step 1: Identify the Memory Overhead Source

Read the following files and trace the exact code path for a single training step with NAMM active:

1. **`scripts/run_lora.py`** — how does it set up the model when `namm_active=True`? Does it load the NAMM as a separate module? How is it attached to the model?

2. **`grad_lora_finetuning/trainer.py`** — the training loop. Find the forward pass. When NAMM is active:
   - Is the NAMM scoring network called inside the same `forward()` that computes the LoRA loss?
   - Are the NAMM scoring computations inside or outside `torch.no_grad()`?
   - Does the model do a full forward pass first (building the complete KV cache), then NAMM scores and evicts, then another forward pass with the evicted cache? Or is it a single pass with eviction happening layer-by-layer?

3. **`namm/` module** — specifically:
   - `namm/policy/` — the scoring/eviction policy. How does it compute scores? Does it process attention weights, hidden states, or key/value tensors?
   - `namm/llms/` — the LLM wrapper. How does it integrate NAMM into the forward pass?
   - Look for any `torch.no_grad()` or `.detach()` calls around the NAMM scoring.

4. **The NAMM paper's approach** (from `2410.13166v4.pdf` if needed): NAMM uses a small neural network (BAM) that scores each token per layer per head. It processes features derived from attention patterns or key/value states.

### Most Likely Cause: NAMM scoring inside the autograd graph

If the NAMM scoring happens inside the computational graph (no `torch.no_grad()` wrapper), then even though the NAMM weights are frozen (requires_grad=False), **PyTorch still stores all intermediate activations needed for backprop through the NAMM's inputs**. The LoRA gradients flow backward through:

```
loss → logits → transformer layers → KV cache (post-eviction) → NAMM scoring → KV cache (pre-eviction) → earlier layers
```

This means PyTorch stores:
- The full pre-eviction KV cache at every layer (not just the evicted 1024 tokens)
- All intermediate tensors from NAMM scoring (attention features, score computations)
- The eviction mask / index tensors

With 16 layers, ~6500 tokens pre-eviction, and bfloat16, the pre-eviction KV cache alone is:
`16 layers × 2 (K+V) × 8 heads × 6500 tokens × 64 dim × 2 bytes ≈ 2.1 GB per sample`

With batch_size=2, that's ~4.2 GB of extra activation storage just for the pre-eviction cache — before counting NAMM scoring intermediates. This easily pushes past GPU memory.

### Other Possible Causes

- **Double KV cache:** The model maintains both the full KV cache and the evicted KV cache simultaneously during the forward pass, doubling memory.
- **NAMM scoring network size:** Unlikely to be significant (it's a small network), but check.
- **Per-layer scoring:** If NAMM scores tokens at every layer independently, it may store 16× the scoring intermediates.
- **Gradient checkpointing not enabled:** M1 might survive without it, but M3 needs it due to the extra overhead.

---

## Step 2: Plan the Fix

Based on what you find, propose fixes in order of preference:

### Fix A: Detach NAMM inputs from the autograd graph (BEST)

If the NAMM scoring is inside the autograd graph, the fix is to detach the tensors going into the NAMM:

```python
# Before NAMM scoring:
with torch.no_grad():
    scores = namm_policy(keys.detach(), values.detach(), attention_weights.detach())
# Use scores to select which tokens to keep, then continue forward pass
```

Or equivalently, wrap the entire NAMM scoring call in `torch.no_grad()`.

**Important:** This only works because the NAMM is frozen during M3 training. We don't need gradients through the NAMM. The LoRA gradients flow through the post-eviction KV cache, which is fine — the eviction decision itself doesn't need to be differentiable.

Check: does `run_joint.py` (M4) need gradients through the NAMM? If M4 uses CMA-ES for NAMM (not gradient-based), then NAMM scoring never needs to be in the autograd graph, and this fix applies to M4 too.

### Fix B: Reduce batch_size

If Fix A isn't feasible (e.g., the code architecture makes it hard to detach), reduce batch_size for M3:

```yaml
# M3 config
batch_size: 1
gradient_accumulation_steps: 16  # keeps effective batch = 16
```

This halves the activation memory per step. The gradients are mathematically identical (accumulated over 16 steps of 1 sample vs 8 steps of 2 samples). The only cost is slower wall-clock time (less GPU parallelism).

**This was the original config before it was changed to batch_size=2.** It was batch_size=1 for a reason — NAMM memory overhead.

### Fix C: Enable gradient checkpointing

If the model supports gradient checkpointing (most HuggingFace models do):

```python
model.gradient_checkpointing_enable()
```

This trades compute for memory by recomputing activations during backward pass instead of storing them. Typically reduces activation memory by ~50-70% at the cost of ~30% slower training.

Check if `run_lora.py` already supports a `gradient_checkpointing` config key. If not, it's straightforward to add.

### Fix D: Combination

If NAMM overhead is inherently high even with detaching, combine A + B:
- Detach NAMM scoring from autograd (Fix A)
- Use batch_size=1 for M3/M4 (Fix B)
- Keep batch_size=2 for M1 only if we add a fairness note explaining the difference

But note: if M1 and M3 use different batch_sizes, document why. The effective batch is the same (16), but per-step gradient noise differs slightly.

---

## Step 3: Memory Profiling

Before and after the fix, add a temporary memory measurement to confirm the fix works:

```python
import torch

# Add at the start of the training loop (after model is loaded, before first step):
print(f"Model loaded: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")

# Add after the first forward pass:
print(f"After forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")

# Add after the first backward pass:
print(f"After backward: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")
torch.cuda.reset_peak_memory_stats()
```

Run a 2-step smoke test for both M1 and M3 and report the memory numbers. This tells us exactly where the overhead is (forward pass vs backward pass vs optimizer step).

---

## Step 4: Apply the Fix and Update Configs

After implementing the fix:

1. **Run the smoke test again** to confirm M3 fits in memory with batch_size=2 (if Fix A) or batch_size=1 (if Fix B).

2. **Update `lora_rh_m4_instruct_5t.yaml`** if batch_size needs to change:
   ```yaml
   batch_size: 1                # NAMM active requires batch_size=1 for memory
   gradient_accumulation_steps: 16  # effective batch = 16
   ```

3. **If M3 must use batch_size=1 but M1 uses batch_size=2**, decide:
   - **Option 1 (preferred for fairness):** Set M1 to batch_size=1 too. Slower but controlled.
   - **Option 2:** Keep M1 at batch_size=2 and document that effective batch is identical (16). Acknowledge the minor per-step gradient noise difference.
   
   If you change M1's batch_size, update `lora_rh_m1_instruct_5t.yaml` to match.

4. **Update `joint_lora_m4_5t.yaml`** — M4 also runs LoRA with NAMM active. It currently has `lora_batch_size: 1`. If the fix (A) allows batch_size=2 for NAMM-active training, update M4 too.

5. **If you modified any source files** (e.g., added `torch.no_grad()` around NAMM scoring), document the change and why it's safe (frozen NAMM doesn't need gradients; CMA-ES doesn't use autograd).

---

## Step 5: Write Findings

Write `docs/namm_memory_investigation.md`:

1. **Root cause** — what exactly causes the OOM (with code references)
2. **Memory breakdown** — model size, KV cache, NAMM overhead, activation storage (numbers from profiling)
3. **Fix applied** — what was changed and why it's safe
4. **Batch size decision** — final batch_size for M1, M3, M4 and the justification
5. **Impact on existing results** — do any completed runs need to be re-evaluated? (Probably not — this is a training memory issue, not a correctness issue)

---

## Output Summary

```
docs/namm_memory_investigation.md    # Root cause, fix, memory numbers

# If code was modified:
namm/llms/<file>.py                  # or wherever NAMM scoring happens
  — Added torch.no_grad() around NAMM scoring during LoRA training

# If configs were modified:
scripts/configs/lora_rh_m4_instruct_5t.yaml   # batch_size if changed
scripts/configs/lora_rh_m1_instruct_5t.yaml   # batch_size if changed for fairness
scripts/configs/joint_lora_m4_5t.yaml         # lora_batch_size if changed
```
