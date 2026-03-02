# Domain Pitfalls: Joint NAMM + LoRA-ES Training

**Domain:** Gradient-free joint optimization of memory policy (NAMM) and LoRA adapters via Evolution Strategies on a single-GPU LLaMA 3.2-1B
**Researched:** 2026-03-02
**Confidence:** MEDIUM — codebase analysis + established ES/LoRA literature

---

## Critical Pitfalls

### Pitfall 1: CMA-ES Covariance Matrix Explodes VRAM at LoRA Scale

**What goes wrong:**
`cma_es.py:161` allocates `self.C = nn.Parameter(data=torch.eye(param_size))` — a full N×N matrix.
For NAMM alone (~300-3000 params) this is negligible. Mode A concatenating LoRA weights is catastrophic:
- r=2, q+v only, 16 layers: ~213K params → C = 213K×213K×4 bytes = **~181 GB**

The `clipped_param_size = min(param_size, 40000)` at line 36 affects learning rate scaling only; it does NOT cap C allocation.

**Prevention:** Mode A is only theoretically possible at r=1, q_proj only, and even then ~65K params exceeds CMA-ES feasibility. Add hard assertion: `assert joint_param_size < 5000` before Mode A CMA-ES instantiation. **Mode B is required for any real LoRA config.**

**Detection:** OOM at process start before data loads. `torch.eye(param_size)` triggers CUDA allocation error.

**Phase:** Phase 1 — resolve before any joint training attempt.

---

### Pitfall 2: PEFT Injection Ordering vs. NAMM's Model Wrapping

**What goes wrong:**
`WrappedLlamaForCausalLM.__init__()` reconstructs the model and calls `load_partial_state_dict()` which captures `base_model_param_keys` before PEFT injection. Injecting PEFT before wrapping means PEFT's new parameter names (`lora_A.default.weight`, `base_layer.weight`) are silently skipped in the load loop. Injecting after wrapping should work, but NAMM's custom attention forward pass must be verified to still call through the LoRA wrapper.

**Prevention:** Apply PEFT **after** `WrappedLlamaForCausalLM` construction. Always specify `target_modules` explicitly. Assert after injection: `assert any('lora_A' in n for n, _ in model.named_parameters())`.

**Detection:** LoRA A/B norms zero after resume; training curve identical to NAMM-only.

**Phase:** Phase 1 — unit test required before ES training.

---

### Pitfall 3: Mixed-Scale Parameter Vector Breaks CMA-ES (Mode A)

**What goes wrong:**
NAMM params after 200 iters: O(0.1-1.0), `init_sigma=0.065`. LoRA B (PEFT default): exactly 0.0. LoRA A: O(0.01-0.1) from `kaiming_uniform`. A single CMA-ES sigma for both is wrong by orders of magnitude. CMA-ES needs O(N) evaluations to learn relative scales — far beyond a 200-iter budget.

**Prevention:** Use Mode B (separate step sizes). If Mode A: initialize LoRA A and B from `N(0, init_sigma)` so scales match NAMM. Log `param_vector[:namm_dim].norm()` vs `param_vector[namm_dim:].norm()` every 10 iters.

**Detection:** CMA-ES sigma collapses within 20-30 iters; `sample_D` eigenvalues show no variance in LoRA subspace.

**Phase:** Phase 2 (Mode A design decision before first run).

---

### Pitfall 4: Single Fitness Score Cannot Attribute Credit to NAMM and LoRA

**What goes wrong:**
NAMM has 200 iters of QASPER adaptation already. LoRA perturbations that initially degrade QASPER will be rejected — LoRA never escapes its zero-init safe zone. In Mode B both ES objects see the same scalar fitness and cannot decompose credit.

**Prevention:** Freeze NAMM for first N iters of joint training to let LoRA find its direction. Use multi-task fitness: `α·qasper + β·narrativeqa + γ·countdown`. Log NAMM-only and LoRA-only diagnostic ablation runs.

**Detection:** LoRA weight norms stay near initialization after 50+ iters; joint fitness matches NAMM-only baseline.

**Phase:** Phase 2 — diagnostic logging required before any run.

---

### Pitfall 5: PEFT merge_adapter() During Population Evaluation Corrupts Base Weights

**What goes wrong:**
If `model.generate()` triggers PEFT's merge/unmerge cycle internally, base LLM weights get permanently modified each call. Repeated calls per population member per generation accumulate rounding errors — fitness scores within a generation become incomparable; base model degrades across generations.

**Prevention:** Never call `merge_adapter()` during training. Use `disable_adapter_layers()` / `enable_adapter_layers()`. Assert before+after one full population eval: `model.model.layers[0].self_attn.q_proj.weight` is bit-for-bit identical. Consider bypassing PeftModel for inference: manual `output = base_out + lora_B @ lora_A @ input`.

**Detection:** Fitness variance increases across iterations without mean improvement; first pop member consistently outscores last.

**Phase:** Phase 1 — unit test required.

---

### Pitfall 6: Existing NAMM Checkpoints Incompatible with Joint param_size (Mode A)

**What goes wrong:**
`ckpt.pt` saves `raw_evolution_algorithm.state_dict()` with tensors shaped `[param_size]` and `[param_size, param_size]`. Mode A changes param_size → `load_state_dict` fails with size mismatch on `self.mean`, `self.best_member`, `self.C`, `self.x`, `self.y`. Cannot warm-start from the best NAMM checkpoint.

**Prevention:** Mode B keeps CMA-ES `param_size` as NAMM-only; store LoRA ES state separately — **zero migration cost** (strong reason to start with Mode B). For Mode A: implement `_load_ckpt_partial()` that pads tensors with zeros/identity in LoRA dimensions. Version checkpoints: add `checkpoint['mode']` and `checkpoint['lora_config']`.

**Detection:** `RuntimeError: size mismatch for mean`; joint training forced to start from iteration 0.

**Phase:** Phase 2 — checkpoint format design before writing joint training loop.

---

### Pitfall 7: PEFT Auto-Detection Silently Injects Zero Modules on Custom Class

**What goes wrong:**
`get_peft_model()` uses `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` keyed by class name. `WrappedLlamaForCausalLM` and `LlamaMemoryModel` are not in this mapping. Without explicit `target_modules`, PEFT may inject LoRA into zero modules — no error, no warning, training proceeds identically to NAMM-only.

**Prevention:** Always specify `target_modules=["q_proj", "v_proj"]` explicitly in `LoraConfig`. Assert: `lora_count = sum(1 for n, _ in model.named_modules() if 'lora_A' in n); assert lora_count == num_layers * len(target_modules)`.

**Detection:** `sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)` returns 0.

**Phase:** Phase 1 — explicit assertion required.

---

## Moderate Pitfalls

### Pitfall 8: VRAM Budget for LoRA Population Copies

LoRA weight storage is small (r=4, q+v, 8 pop copies: ~13.6 MB). Activation memory at r=8 targeting all linear layers: ~10-15% VRAM increase per member. Store population LoRA weights on CPU, move per-member to GPU during evaluation (follows existing NAMM pattern). Profile with `torch.cuda.memory_allocated()` before first training run.

**Phase:** Phase 1 — profile before training.

---

### Pitfall 9: ES for LoRA at pop_size=8 Has Weak Signal-to-Noise

NAMM works with pop_size=8 because NAMM params control discrete high-impact cache decisions. LoRA perturbations have small continuous marginal effect on task score → much weaker ES signal. The es-fine-tuning-paper used pop=30. Expect slower convergence; budget 500+ iterations. Use larger `sigma_lora` (try 0.1) and larger `samples_batch_size` (32-64) in Mode B.

**Phase:** Phase 2 — hyperparameter search required.

---

### Pitfall 10: Task Transferability Confounded by QASPER Training Bias

Training on QASPER may cause LoRA to learn QASPER-specific extractive artifacts that harm NarrativeQA and Countdown. "No transferability" could mean negative transfer. Always run NAMM-only baseline on all three tasks first. Use multi-task fitness from the start. Report 3+ seeds.

**Phase:** Phase 3 — evaluation protocol locked before interpreting results.

---

### Pitfall 11: Countdown Task Requires Custom Data Pipeline

Countdown is not in LongBench. Naive integration fails at dataset loading or produces zero scores from the wrong metric. Implement `evaluate_countdown()` in `MemoryHFEvaluator` with exact-match scoring, or format as LongBench-compatible JSON. Test independently before adding to ES loop.

**Phase:** Phase 3.

---

## Minor Pitfalls

### Pitfall 12: bfloat16 LoRA Weights Underflow at Small Ranks

LLaMA 3.2-1B runs in bfloat16; PEFT defaults LoRA to model dtype. At r=1-2, LoRA low-rank products may round to zero (7-bit mantissa). **Set `lora_dtype=torch.float32`** in `LoraConfig`.

**Phase:** Phase 1.

---

### Pitfall 13: Hydra param_size Config Conflict in Mode A

`param_size` computed from `memory_policy.param_size` in `main.py`. In Mode A this must change. Stale cached configs or resume runs load the old NAMM-only size. In Mode B: do not change CMA-ES `param_size`. In Mode A: assert `cfg.param_size == namm_param_size + lora_param_size` after config resolution.

**Phase:** Phase 2.

---

## 4x Speedup Note

**Current runtime estimate (post-optimisation):** ~48s/iter → ~3h for 200 iters on 4070Ti.
With 4 ablation conditions: **~12h total** (~1 GPU-day). This is much more manageable than the pre-optimisation 10.7h/run baseline.

---

## Phase Summary

| Phase | Pitfalls to Resolve |
|-------|-------------------|
| Phase 1 | 2, 5, 7 (ordering, unit tests, explicit targets), 8 (VRAM profile), 12 (bfloat16) |
| Phase 2 | 1, 3 (Mode A CMA-ES), 4 (sigma-ES), 6 (ckpt migration), 9 (SNR budget), 13 (Hydra) |
| Phase 3 | 10 (transferability protocol), 11 (Countdown pipeline) |

---

*Sources: `cma_es.py:161` (C matrix), `memory_llms/llama.py:140-200` (wrapping), `memory_trainer.py:1055-1150` (checkpoint), PEFT v0.11.1 docs, ES SNR theory.*
