# Pitfalls Research

**Domain:** Adding gradient-based LoRA finetuning to existing gradient-free NAMM CMA-ES system (LLaMA 3.2-1B)
**Researched:** 2026-03-03
**Confidence:** HIGH — based on direct codebase inspection of all modified files + PEFT v0.11.1 source + PyTorch autograd mechanics. ES-specific pitfalls from v1.0 PITFALLS.md retained where still applicable.

---

## Critical Pitfalls

### Pitfall 1: `@torch.no_grad()` Blanket Decorator Kills Gradient Training

**What goes wrong:**
`MemoryTrainer.__init__`, `_train_step`, `train`, and essentially every method in `memory_trainer.py` is decorated with `@torch.no_grad()` (confirmed at lines 88, 350, 361, 439, 506, 527, 540, 591, 669, 678, 843, 858, 874, 883, 912, 920, 972, 1185). The existing system is entirely gradient-free by design. A new gradient-based LoRA training loop running inside or through any of these decorated methods will have its computational graph silently destroyed. `loss.backward()` will raise `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` — or worse, succeed with zero gradients if autograd enabled downstream accidentally re-enables gradient tracking.

**Why it happens:**
The NAMM CMA-ES trainer was written gradient-free from day one; `@torch.no_grad()` is applied at the class method level throughout. Any new LoRA training loop naively added inside the existing training methods inherits the no-grad context. The error is not obvious because PyTorch's `no_grad` context is inherited by all ops in the decorated scope regardless of where they are called.

**How to avoid:**
The LoRA gradient training loop must live in a completely separate trainer class or method that is not decorated with `@torch.no_grad()`. Do not add the LoRA training loop inside `MemoryTrainer._train_step`. Either: (a) write a new `LoRATrainer` class that does not use `@torch.no_grad()`, or (b) add a new method `_train_lora_step` that explicitly uses `torch.enable_grad()` as a context manager at the entry point. For m3 (LoRA first) this means LoRA training runs entirely before touching `MemoryTrainer`. For m4 (LoRA with NAMM active) the LoRA training pass must use `torch.enable_grad()` to re-enable gradients even if it calls methods that touch NAMM internals.

**Warning signs:**
- `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` on `loss.backward()`
- Gradients are zero for all LoRA parameters after `optimizer.step()` (check `[p.grad for p in lora_params if p.grad is not None]`)
- `loss.requires_grad` is `False` before calling `.backward()`

**Phase to address:**
Phase implementing gradient LoRA training loop (first LoRA training phase, pre-m1/m3/m4 runs). This is the single most dangerous pitfall in the codebase — it will prevent all gradient training from working if not caught first.

---

### Pitfall 2: NAMM's KV-Cache Eviction is Non-Differentiable — Gradients Do Not Flow Through It

**What goes wrong:**
NAMM's eviction mechanism (`threshold_score_idxs` in `memory_policy/base_dynamic.py`, and `BinarySelection.select_new_tokens` in `memory_policy/deep_selection.py`) uses `torch.topk()` and integer index selection (`retained_idxs`) to decide which KV-cache entries to keep. Both operations are fundamentally non-differentiable: `torch.topk()` returns indices (integers), and the subsequent `gather()` on those indices has zero gradient with respect to the selection decision itself. In m4 (LoRA training with NAMM active), gradients from the NTP loss will flow back through the LLaMA forward pass, reach the KV-cache eviction point, and stop dead. The gradient of which tokens were selected is zero — there is no signal pushing NAMM toward better eviction decisions through backprop.

**Why it happens:**
This is intentional and correct by design: NAMM is trained by CMA-ES (gradient-free), not by gradients. The non-differentiability is a fundamental architectural choice. The pitfall is not that this breaks NAMM training (it doesn't, NAMM still uses CMA-ES) but that LoRA gradients in m4 are computed on a truncated computational graph. The NTP loss gradient will be: `d_loss/d_LoRA_params` correctly computed through the attention layers that used the cached tokens, but will not include any signal through "which tokens were in the cache." This means: LoRA training in m4 is valid and will work, but it trains the model conditional on whatever NAMM decided to keep — if NAMM evicts a crucial token, LoRA cannot compensate via gradients.

**How to avoid:**
This is not a bug to fix but a design fact to document in the experimental writeup. For m4 training: verify that `loss.backward()` completes without error (it will — NAMM eviction happens before the autograd graph is constructed for the retained tokens), confirm that LoRA parameters receive non-zero gradients after backward, and add an assertion that `sum(p.grad.norm() for p in lora_params) > 0`. Do NOT attempt to make eviction differentiable (Straight-Through Estimator, Gumbel-Softmax, etc.) — this would change NAMM fundamentally and is out of scope.

**Warning signs:**
- Confusion about why gradient training "breaks" with NAMM active — it does not break, but the gradient graph is truncated at the eviction point
- Mistakenly trying to backpropagate through `topk` → will get zero gradient but no error
- Misinterpreting "gradient does not flow through eviction" as "gradient does not flow at all" — LoRA params do get gradients through the tokens that were retained

**Phase to address:**
m4 design phase. Add a comment in the LoRA training loop explicitly noting `# NAMM eviction is non-differentiable — gradients flow only through retained tokens.` Run a gradient-norm assertion test before the first m4 training run.

---

### Pitfall 3: PEFT Injection Before `WrappedLlamaForCausalLM` Construction Silently Produces Zero LoRA Adapters

**What goes wrong:**
`WrappedLlamaForCausalLM.__init__()` calls `load_partial_state_dict()` which captures `base_model_param_keys` — the set of expected base-model parameter names — at construction time. If PEFT is applied before wrapper construction, PEFT renames parameters (`q_proj.weight` → `base_layer.weight`, adds `lora_A.default.weight`, `lora_B.default.weight`). The subsequent `load_partial_state_dict()` sees unfamiliar names and silently skips them, leaving LoRA modules at PEFT's initialization (B=0, A=kaiming_uniform) but with mismatched or absent target modules. Additionally, PEFT's auto-detection (`TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`) does not contain `WrappedLlamaForCausalLM` — without explicit `target_modules`, PEFT injects zero modules.

**Why it happens:**
Two independent failure modes that can both produce the symptom "LoRA adapter injects successfully according to PEFT API but does nothing." First: injection ordering. Second: missing `target_modules` in `LoraConfig`. Both produce no error, no warning. Training proceeds but LoRA parameters have zero effect on output.

**How to avoid:**
Always inject PEFT **after** `WrappedLlamaForCausalLM` construction. Always specify `target_modules=["q_proj", "v_proj"]` (or the intended targets) explicitly in `LoraConfig` — never rely on PEFT auto-detection. After injection, assert: `lora_count = sum(1 for n, _ in model.named_modules() if 'lora_A' in n); assert lora_count == num_layers * len(target_modules)`. For LLaMA 3.2-1B with 16 layers and `["q_proj", "v_proj"]`, expected count is 32.

**Warning signs:**
- `sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)` returns 0
- LoRA L2 norm stays at initialization (near-zero) throughout training
- Loss curve in m1/m3 identical to base model forward pass

**Phase to address:**
Phase 2 (LoRA seam, already implemented and tested in v1.0). For gradient training: re-verify injection order in the new LoRA trainer init since it instantiates the model differently than the ES trainer.

---

### Pitfall 4: `model.generate()` in NAMM Evaluation Blocks Gradient Flow for m4

**What goes wrong:**
In m4, LoRA is trained while NAMM is active. The training objective is NTP loss (next-token prediction), which requires a forward pass, not `model.generate()`. However, if the LoRA trainer accidentally reuses the `MemoryHFEvaluator` evaluation path (which calls `model.generate()`) to compute training loss — rather than implementing a dedicated NTP forward pass — there will be no gradient. `model.generate()` uses KV-cache in autoregressive decoding mode; the computational graph for the generated tokens is not connected to the loss computation in any useful way for LoRA gradients.

**Why it happens:**
The existing codebase only has `model.generate()` as its LLM interface. There is no existing NTP forward pass (teacher-forcing mode). A developer might be tempted to adapt the evaluator's generation infrastructure for training loss, which will silently produce zero-gradient training.

**How to avoid:**
Implement a clean NTP forward pass: `logits = model(input_ids)[0]`; `loss = cross_entropy(logits[:, :-1], input_ids[:, 1:])`. Do not use `model.generate()` for computing LoRA training loss. The two paths (generation for NAMM eval, teacher-forcing for LoRA training) must remain separate. Test by checking `loss.requires_grad == True` after the NTP forward pass.

**Warning signs:**
- LoRA training loss does not decrease over epochs
- `loss.grad_fn is None` after the forward pass

**Phase to address:**
LoRA training loop implementation phase (first phase of v2.0 gradient work). This is likely the highest-impact implementation error for anyone porting the existing eval harness into a training loop.

---

### Pitfall 5: `@torch.no_grad()` on `MemoryTrainer.__init__` Freezes NAMM State During m4 Active Inference

**What goes wrong:**
In m4, the intent is that NAMM is active (performing KV-cache eviction) during LoRA training. If the NAMM model state (loaded from a checkpoint) is accessed through paths decorated with `@torch.no_grad()`, NAMM's internal attention-score computation for scoring tokens runs without gradient tracking. This is fine for NAMM itself (gradient-free), but if LoRA is being trained while NAMM is active, any tensor produced by NAMM's scoring network and used as input to the LLaMA forward pass will have `requires_grad=False`, potentially detaching parts of the gradient graph.

**Why it happens:**
NAMM's `ParamMemoryPolicy.forward()` is called inside attention layers. If this call happens inside a `no_grad` scope (directly or via inherited context), the outputs of NAMM's scoring network become leaves with `requires_grad=False`. Since these outputs influence which KV tokens are retained, and the retained tokens influence the NTP loss, some gradient paths can get cut.

**How to avoid:**
In m4 training forward pass: wrap NAMM's forward call with `torch.no_grad()` explicitly (to prevent PyTorch from attempting gradients through NAMM), but ensure the LLaMA attention forward pass — specifically the computation over retained tokens — runs under `torch.enable_grad()`. The cleanest solution: run NAMM's eviction step under `torch.no_grad()`, collect the retained token indices, then perform the LoRA attention computation on those indices under `torch.enable_grad()`. In practice, since NAMM eviction produces integer indices and the gather operation has zero gradient w.r.t. eviction decision anyway (Pitfall 2), this may be a non-issue in practice — but it must be verified with a gradient-flow test.

**Warning signs:**
- LoRA gradients are smaller in m4 than in m1 (expected) vs zero (not expected)
- `model.train()` correctly called but gradients still missing

**Phase to address:**
m4 implementation phase. Add `assert all(p.grad is not None for p in lora_params if p.requires_grad)` after the first backward pass in m4.

---

### Pitfall 6: Checkpoint Handoff Between m3 and m4 Fails Silently on LoRA Config Mismatch

**What goes wrong:**
m3 produces a LoRA-finetuned checkpoint. m4 loads a NAMM checkpoint and also needs to load LoRA weights from the m3 checkpoint (or start LoRA from scratch). If the LoRA rank or target_modules in the m4 config do not exactly match what was used in m3, `_load_ckpt` raises a `ValueError` (caught at line 1156-1163 in `memory_trainer.py`) — but only if `lora_config` is checked. If the check is bypassed or the new LoRA trainer uses a different config-loading path, parameter shapes will silently mismatch, causing a corrupted model rather than an error.

A secondary handoff failure: m4 may load from the best NAMM checkpoint (`ckpt.pt` from v1.0, which has no `lora_state_dict`). In that case, `_load_ckpt` falls back gracefully and LoRA initializes from PEFT defaults (B=0). This is correct for m4 when LoRA starts fresh, but is a bug if the intent was to continue m3 LoRA training.

**Why it happens:**
Multi-stage pipelines require careful choreography of which checkpoint provides which component. With two independent state dicts (NAMM evolution state and LoRA state), the four combinations (no NAMM + no LoRA, NAMM + no LoRA, no NAMM + LoRA, NAMM + LoRA) must all be explicitly handled.

**How to avoid:**
Define a `StageConfig` or `CheckpointManifest` that explicitly declares: `namm_ckpt_path`, `lora_ckpt_path`, `expected_lora_rank`, `expected_lora_targets`. Validate at load time that the checkpoint found at `lora_ckpt_path` has matching config. Add a checkpoint compatibility check script that verifies both components before starting a long training run. Use `checkpoint['joint_es_mode']` (already saved in v1.0) to assert the loaded checkpoint came from the expected stage.

**Warning signs:**
- m4 starts with LoRA at zero when the intent was to load m3 weights
- Cryptic `RuntimeError: size mismatch` at load time if ranks differ
- Eval scores in m4 match m2 (NAMM-only) rather than improving over m3

**Phase to address:**
Multi-stage pipeline config phase. Write a checkpoint inspector utility before the first m3→m4 handoff.

---

### Pitfall 7: Adam Optimizer State is Lost on Checkpoint Resume for LoRA Training

**What goes wrong:**
The existing `_save_ckpt` saves NAMM evolution state and LoRA weights, but does NOT save the Adam optimizer state for gradient training. If a LoRA training run is interrupted and resumed from a checkpoint, Adam's first and second moment estimates (`exp_avg`, `exp_avg_sq`) are reset to zero. This causes a sudden effective learning rate spike on resume (Adam's moment estimates underestimate the gradient magnitude at start), often producing a loss spike or instability at the resume point.

**Why it happens:**
CMA-ES does not have an optimizer state in the Adam sense — `evolution_state` covers CMA-ES's sigma, mean, and covariance. The gradient training world has an additional stateful component (Adam moments) that the existing checkpoint infrastructure has no slot for.

**How to avoid:**
Extend `_save_ckpt` (or the new LoRA trainer's checkpoint method) to save `optimizer.state_dict()`. On resume, load and restore optimizer state before the first backward pass. Alternatively, if runs are short enough to complete without interruption (200-iter LoRA training ≈ 2-3h), skip optimizer state saving but document that resume will cause a loss spike.

**Warning signs:**
- Loss spike immediately after checkpoint resume
- Training instability in the first 10-20 steps after resume
- Final validation score after resume is worse than expected based on pre-resume training curve

**Phase to address:**
LoRA training loop implementation phase. Include `optimizer.state_dict()` in the checkpoint spec from the start.

---

### Pitfall 8: Attention Hooks for Entropy Computation Interfere with PEFT's Forward Hooks

**What goes wrong:**
The project plans to log attention entropy as a live analysis metric during training. PEFT registers its own forward hooks on linear layers to implement the LoRA perturbation (`forward_hook` in `peft/tuners/lora/layer.py`). If the entropy computation also registers `register_forward_hook` on the same attention layers (or on `q_proj`/`v_proj` directly), hook ordering and output capture can interfere. Specifically: (a) if the entropy hook captures the raw linear output before PEFT's LoRA hook has added the LoRA perturbation, entropy will be computed on the base model activations, not the LoRA-modified ones; (b) if a hook modifies the tensor in-place, it can corrupt PEFT's LoRA computation.

**Why it happens:**
PyTorch's `register_forward_hook` fires in registration order. PEFT registers hooks during `get_peft_model()`. Any hook registered after PEFT will see the LoRA-modified outputs; any hook registered before will see pre-LoRA outputs. With `WrappedLlamaForCausalLM`'s custom attention, the hook target is the custom attention module — not the PEFT-wrapped linear — so this may work differently than expected. Additionally, hooks on attention layers during training add per-step overhead that can slow training by 10-30% depending on implementation.

**How to avoid:**
Register entropy hooks AFTER `get_peft_model()` is called, so they see the final (LoRA-modified) outputs. Hook on the `LlamaMemoryAttention` module level (captures the full attention output after LoRA + NAMM), not on `q_proj`/`v_proj` directly. Make hooks detached and `torch.no_grad()` internally — they should not affect the gradient graph. Use `register_forward_hook` with `with_kwargs=True` to verify the hook is receiving the expected tensor. Benchmark overhead before enabling for full training runs.

**Warning signs:**
- Attention entropy values during m1 (no NAMM) match entropy values during m4 (NAMM active) — likely means entropy is being measured before NAMM eviction rather than after
- Training step time increases >20% when hooks are active
- `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` — a hook is modifying tensors in-place

**Phase to address:**
Attention entropy logging phase. Test hook ordering with a single forward pass before enabling during training.

---

### Pitfall 9: Mixed Precision — bfloat16 Forward Pass + float32 LoRA Weights + Adam in float32 Requires Careful Cast Management

**What goes wrong:**
LLaMA 3.2-1B runs in bfloat16 (model activations, base weights). LoRA A and B matrices are stored in float32 (v1.0 design decision, necessary to prevent underflow at low rank). During the forward pass, when the LoRA output `lora_B @ lora_A @ x` is added to the base `q_proj` output, there is an implicit dtype promotion: `bfloat16 + float32 → float32`. This is fine for the computation, but: (a) the loss is computed in float32 from the float32-promoted logits; (b) if AMP (`torch.cuda.amp.autocast`) is used, bfloat16 autocast may cast LoRA weights to bfloat16 mid-forward, losing the float32 precision benefit; (c) Adam moment estimates for float32 LoRA params consume 2x the memory of what bfloat16 would.

**Why it happens:**
PEFT's `LoraLinear.forward()` performs `lora_B(lora_A(dropout(x))) * scaling` and adds to base output. The dtype of this addition depends on `x`'s dtype (bfloat16 in this setup). With AMP enabled, autocast can downcast float32 parameters to bfloat16 in the cast region, overriding the explicit float32 dtype set during injection.

**How to avoid:**
Do not use `torch.cuda.amp.autocast` with the bfloat16+float32 mixed LoRA setup — it was designed for fp16 training, not bfloat16, and can interfere with explicitly float32 LoRA weights. Run the LoRA forward pass in bfloat16 for the base model and let PEFT handle the LoRA addition without autocast interference. Since bfloat16 does not need loss scaling (same exponent range as float32), do not use `GradScaler` either. Verify after each training step: `assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if 'lora_' in n)`.

**Warning signs:**
- LoRA weights silently cast to bfloat16 mid-training (check dtypes after first backward)
- Loss oscillates or is `nan`/`inf` — often a dtype promotion issue
- `GradScaler` causes `inf` gradient scaling issues that prevent optimizer steps

**Phase to address:**
LoRA training loop implementation phase. Establish the dtype contract (no AMP, float32 LoRA params) in training loop documentation before implementation.

---

### Pitfall 10: DataLoader for Long-Context Documents Causes OOM During Gradient Training

**What goes wrong:**
The existing NAMM eval uses `model.generate()` with KV-cache — tokens are generated one at a time, and only the KV-cache for retained tokens is held in VRAM. During gradient LoRA training (teacher-forcing NTP loss), the entire sequence must be processed in one forward pass with the full attention graph retained for backpropagation. For long-context documents (QASPER averages ~3000-4000 tokens), storing activations for all layers × all tokens × batch_size for backprop requires dramatically more VRAM than generation mode.

For LLaMA 3.2-1B (16 layers), a 4096-token sequence with batch_size=1 and bfloat16 activations requires approximately: `16 × 4096 × 2048 × 2 bytes ≈ 268 MB` for intermediate activations alone, before LoRA gradient buffers. With batch_size=4, this is ~1 GB — workable. But if the DataLoader pads to the maximum sequence length in the batch, a single long document (8k tokens) forces all batch members to be padded to 8k, multiplying VRAM by 2x unexpectedly.

**Why it happens:**
The existing `TaskSampler` was designed for generation-mode evaluation, not gradient training. It may not truncate sequences to a training-compatible length, and the existing `eval_samples_batch_size=4` assumes generation (low peak VRAM). Training with teacher-forcing at the same batch size will OOM.

**How to avoid:**
Truncate all training sequences to `max_training_len` (e.g., 2048 tokens) for gradient training. Use `max_length=max_training_len` in the tokenizer call. Implement a custom collate function that pads to the batch maximum (not a global maximum). Use gradient accumulation if batch_size=1 is needed for memory. Profile peak VRAM with `torch.cuda.max_memory_allocated()` on a single-batch forward+backward before scaling up. For m4 specifically: if NAMM is active and cache_size=128, the KV-cache for retained tokens is small, but the full-sequence NTP forward pass still requires the full activation graph.

**Warning signs:**
- `torch.cuda.OutOfMemoryError` on the first backward pass but not on the first forward pass
- OOM occurs only with batch_size>1, not batch_size=1
- Training step VRAM is 3-5x higher than eval step VRAM (expected ratio for training vs generation)

**Phase to address:**
LoRA DataLoader/training loop setup phase. Establish `max_training_len` and padding strategy before first training run.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reusing `MemoryHFEvaluator` for LoRA training loss | No new data pipeline code | `model.generate()` path produces zero-gradient NTP loss; must be replaced entirely | Never — implement a dedicated NTP trainer |
| Not saving Adam optimizer state in checkpoints | Simpler checkpoint format | Loss spike + instability on every resume; training curves are not reproducible across interruptions | Only if all training runs are guaranteed to complete in one shot (<3h on 4070Ti) |
| Using same `max_new_tokens` batch size for training as for eval | One less config knob | OOM during backpropagation; gradient training needs 3-5x lower effective batch size | Never — always profile separately |
| Registering entropy hooks before `get_peft_model()` | Simpler initialization order | Entropy computed on pre-LoRA activations — scientifically incorrect for m1/m3/m4 comparison | Never if entropy is a primary metric |
| Running m4 without `torch.enable_grad()` wrapper | No code change needed | All LoRA gradients are zero; optimizer never updates | Never |
| Loading m4 LoRA init from NAMM ckpt's zero-initialized LoRA | Convenient — one checkpoint load | If intent was to continue m3 LoRA weights, starts from scratch silently | Acceptable only when m4 is explicitly designed to start LoRA from zero |

---

## Integration Gotchas

Common mistakes when connecting the gradient trainer to the existing NAMM system.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PEFT + `WrappedLlamaForCausalLM` | Call `model.train()` on the PEFT-wrapped model — this may enable training mode for base model weights, allowing them to accumulate gradients | Call `model.train()` then immediately call `model.base_model.model.requires_grad_(False)` to re-freeze base weights; verify with `sum(p.requires_grad for p in model.base_model.parameters() if 'lora_' not in n)` |
| NAMM KV-cache state across training steps | Not clearing the KV-cache between NTP forward passes — NAMM's internal cache state from the previous forward pass contaminates the next | Call `model.reset_memory()` or equivalent before each NTP forward pass in training |
| m4 NAMM + LoRA joint forward | Calling `model.training_mode()` on MemoryTrainer (which enables NAMM's training behavior) while also doing gradient LoRA training — NAMM in training mode may enable auxiliary losses that interfere | In m4, run NAMM in eval mode for eviction decisions, run LoRA in train mode for gradient updates; these are separate concerns |
| Checkpoint load ordering in m3→m4 | Loading NAMM checkpoint first, then attempting to inject PEFT LoRA after load — `_load_ckpt` may fail if LoRA adapters are not yet present when the checkpoint attempts to restore them | Inject PEFT LoRA adapters before loading any checkpoint; the checkpoint load then restores LoRA weights into the already-injected adapter |
| NTP loss with padding tokens | Computing cross-entropy loss including padding token positions — inflates loss, masks signal, gradients on padding positions are meaningless | Apply attention mask to loss: `loss = cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=pad_token_id)` |

---

## Performance Traps

Patterns that work but fail to scale or are substantially slower than expected.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full-sequence NTP forward pass at eval sequence lengths (3000-4000 tokens) | OOM or >2min/step | Truncate training sequences to 2048 tokens; use gradient checkpointing for sequences >2048 | First training step on QASPER with batch_size>1 |
| Gradient accumulation without zeroing NAMM KV-cache between accumulation steps | Corrupted KV-cache state in later accumulation steps; loss diverges | Reset NAMM state at each accumulation step, not just each optimizer step | Immediately — accumulation with stateful cache breaks from step 2 |
| Entropy hooks on every forward pass during training | >20% step slowdown; train/eval speed diverges | Log entropy every N steps (N=10 or 100), not every step; compute once per step not per layer | When hooks are enabled and batch_size>1 |
| Adam with `lr=1e-4` for LoRA at rank 4 (standard LoRA default) | Initial loss spike; instability | Start with `lr=2e-5` for task-specific NTP on a pretrained model; cosine LR schedule | First 10-20 training steps on QASPER |
| Saving full model checkpoint (base + LoRA) instead of adapter-only | 2.5 GB checkpoint instead of ~7 MB | Save `model.save_pretrained()` for adapter-only; or save only `lora_state_dict` as done in v1.0 `_save_ckpt` | When storage or checkpoint loading becomes slow |

---

## "Looks Done But Isn't" Checklist

Things that appear to be working but have a critical missing piece.

- [ ] **LoRA training loop:** Training loss decreasing but LoRA weights not actually updating — verify `param.grad is not None` for all LoRA params after first backward. Check that `optimizer.param_groups` contains the LoRA params, not an empty list.
- [ ] **m4 active NAMM:** LoRA training appears to work but NAMM is not actually doing eviction — verify `model.memory_policy.cache_size < full_context_length` during training forward pass; log `num_retained_tokens` per step.
- [ ] **Checkpoint handoff m3→m4:** m4 loads checkpoint but LoRA weights are at PEFT initialization (zero B matrix) — verify by comparing `lora_B.weight.norm()` before and after loading; a correctly loaded m3 checkpoint will have non-zero B.
- [ ] **Attention entropy:** Values are logged every step but appear constant or near-maximum — likely the entropy hook is capturing pre-eviction attention (all tokens present), not post-eviction. Verify the hook fires after NAMM's `select_new_tokens` is called.
- [ ] **Stage isolation in m3:** m3 is "LoRA finetuning without NAMM" — verify that `policy=none` or `cache_size=context_length` is set; if NAMM is accidentally active, m3 is silently equivalent to m4.
- [ ] **Frozen base weights in gradient training:** After calling `model.train()`, base model weights may have `requires_grad=True` — check `model.base_model.model.layers[0].self_attn.q_proj.base_layer.weight.requires_grad` — must be `False`.
- [ ] **bfloat16 LoRA dtype:** LoRA adapters appear to work but precision degraded — check `model.named_parameters()` for `lora_A.weight.dtype` — must be `torch.float32`, not `torch.bfloat16`.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| `@no_grad()` kills LoRA gradients | LOW | Add `torch.enable_grad()` context manager at LoRA training loop entry point; no code restructuring needed |
| LoRA checkpoint mismatch (rank/targets differ) | MEDIUM | Re-run m3 training with matching config, OR convert checkpoint with a migration script that zero-pads LoRA A/B to new rank; second option is experimental |
| Adam optimizer state lost on resume | LOW | Accept loss spike for first 20 steps post-resume; add learning rate warmup (10 steps) to mitigate the effective LR spike |
| OOM during first training backward | LOW | Reduce `batch_size` by half; add `max_length=2048` truncation; enable gradient checkpointing in PEFT with `model.enable_input_require_grads()` before wrapping |
| Entropy hooks capturing wrong activations | LOW | Remove and re-register hooks after `get_peft_model()` call; add one-step verification that hook output changes when LoRA weights are perturbed |
| PEFT base weights unfrozen after `model.train()` | LOW | Call `model.base_model.model.requires_grad_(False)` immediately after `model.train()`; add assertion in training loop entry |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: `@no_grad()` decorator kills LoRA gradients | Phase implementing LoRA training loop (first gradient phase) | Unit test: `loss.requires_grad == True` after NTP forward pass in new trainer class |
| P2: NAMM eviction non-differentiable | m4 design phase — document, not fix | Assertion: `lora_params[0].grad is not None` after backward in m4 confirms gradients flow through retained tokens |
| P3: PEFT injection ordering / zero adapters | Phase 2 (v1.0, done) — re-verify in gradient trainer init | Module count assertion: `lora_count == 32` for default config |
| P4: `model.generate()` in training path blocks gradients | Phase implementing NTP trainer | `loss.grad_fn is not None` after NTP forward pass; never import `MemoryHFEvaluator` in LoRA training path |
| P5: NAMM state interferes with gradient graph | m4 implementation phase | Gradient norm assertion per LoRA parameter after m4 backward |
| P6: Checkpoint handoff m3→m4 fails silently | Multi-stage pipeline config phase | Checkpoint inspector utility; `lora_B.weight.norm() > 0` after m4 load when m3 LoRA is expected |
| P7: Adam state lost on resume | LoRA training loop implementation | `optimizer.state_dict()` in checkpoint; loss-spike test on intentional resume |
| P8: Entropy hooks interfere with PEFT hooks | Attention entropy logging phase | Hook ordering test: entropy differs between base model and LoRA-injected model on same input |
| P9: Mixed precision bfloat16+float32 dtype issues | LoRA training loop implementation | Post-training dtype assertion; no AMP GradScaler in training config |
| P10: DataLoader OOM during gradient training | DataLoader/training loop setup phase | Profile VRAM with `torch.cuda.max_memory_allocated()` before scaling batch size |

---

## Retained Pitfalls from v1.0 (ES-Based LoRA) — Still Applicable

These pitfalls from the original NAMM + LoRA-ES research remain relevant for the gradient-based system.

### V1-P5: PEFT merge_adapter() Corrupts Base Weights

During m4, if any code path calls `model.merge_adapter()` while NAMM is performing population evaluation (CMA-ES runs), base weights get permanently modified. Still relevant because NAMM CMA-ES training still runs (via CMA-ES, gradient-free), and if any eval path accidentally triggers merge, NAMM's fitness evaluation runs on a corrupted base model.

**Prevention:** Never call `merge_adapter()` anywhere in the codebase. Use `disable_adapter_layers()` / `enable_adapter_layers()` for ablation. Assert base weight hash before and after any eval pass.

**Phase:** Any phase where NAMM CMA-ES and PEFT coexist (m2 re-run, m4).

---

### V1-P4: Credit Attribution — NAMM Already Converged, LoRA May Not Find Signal

For the gradient-based system, this pitfall transforms: in m4, NAMM's eviction decisions are frozen (NAMM is not being trained in m4, it is a fixed policy). LoRA gradients only receive signal through the tokens that NAMM chose to retain. If NAMM (trained on QASPER) retains highly task-specific tokens, LoRA may overfit to a narrow retrieval pattern rather than learning general representation improvements.

**Prevention:** Run m3 (LoRA without NAMM) first to establish a LoRA baseline uncontaminated by NAMM's token selection. Compare m3 vs m4 LoRA weight norms and attention entropy to determine if NAMM's presence during LoRA training is helping or constraining.

**Phase:** Experimental design phase — m3 must run before m4 for valid scientific comparison.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection:
  - `memory_trainer.py`: `@torch.no_grad()` at lines 88, 350, 361, 439, 506, 527, 540, 591, 669, 678, 843, 858, 874, 883, 912, 920, 972, 1185
  - `memory_policy/deep_selection.py`: `BinarySelection.select_new_tokens()` uses `topk`/index-select — confirmed non-differentiable
  - `memory_policy/base_dynamic.py`: `threshold_score_idxs()` uses `torch.topk()` + integer indices — non-differentiable
  - `memory_llms/base.py:108-163`: LoRA injection and flat-vector interface
  - `memory_trainer.py:1055-1183`: `_save_ckpt` / `_load_ckpt` — no Adam optimizer state saved
- PEFT v0.11.1 source: hook registration order in `tuners/lora/layer.py`
- PyTorch autograd mechanics: `@torch.no_grad()` inheritance behavior confirmed via PyTorch docs

### Secondary (MEDIUM confidence)
- PyTorch community issue: [Combining no_grad() decorator causes gradient re-enabling](https://discuss.pytorch.org/t/combining-no-grad-decorator-and-with-torch-no-grad-operator-causes-gradients-to-be-enabled/39203) — confirmed that `no_grad` as decorator has specific edge cases
- PEFT GitHub issues: [gradient checkpointing + LoRA](https://github.com/huggingface/peft/issues/1142), [requires_grad=False after checkpoint load](https://github.com/huggingface/peft/issues/245) — PEFT hook registration ordering behavior
- HuggingFace Transformers issues: [no_grad after PeftModel creation breaks requires_grad](https://github.com/huggingface/transformers/issues/26334) — `model.train()` does not automatically restore frozen base weights
- Mixed precision training: bfloat16 + float32 LoRA interaction from PyTorch AMP docs — no loss scaling needed for bfloat16
- NAMM checkpoint inspect: confirmed v1.0 `_save_ckpt` saves no Adam optimizer state

### Tertiary (LOW confidence)
- Attention entropy hook interference — inferred from PEFT hook registration documentation; not empirically tested in this codebase
- m4 NAMM token-selection bias on LoRA representation learning — theoretical inference; empirical validation required via m3 vs m4 comparison

---

*Pitfalls research for: Adding gradient-based LoRA finetuning to gradient-free NAMM CMA-ES system*
*Researched: 2026-03-03*
*Supersedes v1.0 PITFALLS.md (ES-based LoRA) for the gradient training milestone*
