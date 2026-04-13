# VRAM Memory Audit & Optimisation — LoRA Fine-Tuning Pipeline

## Context

This is the **evo-memory** project: LoRA fine-tuning of Llama-3.2-1B-Instruct on a 5-task LongBench QA subset, with optional NAMM (Neural Attention Memory Model) KV cache eviction. Training runs on a **single GPU with 8 GB usable VRAM** (HAMI-virtualised). Sequences are long-context (4096–6500 tokens), attention is full O(n²) (no flash attention), and evaluation iterates over 180+ task samples.

We have already made several emergency fixes to stop OOM crashes (listed below). Now we need a **systematic, complete audit** of VRAM usage across the entire training lifecycle, followed by a plan and implementation of all safe optimisations — without changing training dynamics or model behaviour.

### Changes already applied

- `batch_size_eval: 1` (was 4) to prevent eval OOM
- `batch_size: 1`, `gradient_accumulation_steps: 16` for training (effective batch = 16)
- `--skip_baseline_eval` flag added to `scripts/run_lora.py` and `grad_lora_finetuning/trainer.py` to skip the pre-training F1 eval and reduce startup VRAM peak
- `torch.cuda.empty_cache()` calls added between val eval / train eval / debug generation in `trainer.py`
- `torch.cuda.empty_cache()` after each task's `evaluate_lb` call in `namm/tasks.py` (+ the missing `import torch`)
- Gradient checkpointing already enabled for M1 (`namm_active=False`)

### Known characteristics

- Two separate batch size fields: `batch_size` (training) vs `batch_size_eval` (piped into Hydra for eval)
- No flash attention in `LlamaMemoryAttention` — full O(n²) attention matrix is the main VRAM driver
- 11 GB → 7 GB oscillation during training is normal: eval peaks higher than training steady-state due to attention matrix materialisation across multiple tasks
- Gradient checkpointing is implemented but only activates when `namm_active=False`

---

## Your task

### Phase 1 — Deep code analysis & memory map

Read and understand every file involved in the LoRA training + evaluation loop. At minimum:

1. **Entry point:** `scripts/run_lora.py` — CLI parsing, config loading, trainer instantiation
2. **Trainer:** `grad_lora_finetuning/trainer.py` — the full `LoRAGradTrainer` class: training loop, eval loop, checkpointing, logging, model setup
3. **Datasets:** `grad_lora_finetuning/datasets.py` — `NTPDataset` and `SFTDataset`, collation, tokenisation
4. **Task evaluation:** `namm/tasks.py` — task definitions, the `evaluate_lb` method, how samples are iterated
5. **LLM wrapper:** `namm/llms/` — the LLM class used during eval (generation, attention, KV cache handling)
6. **Evaluator:** `namm/evaluation/` — evaluator classes, metric computation
7. **Model loading & LoRA injection:** `utils/helpers.py` — how the base model is loaded, how LoRA adapters are applied, dtype, device placement
8. **Config files:** `scripts/configs/lora_rh_m1_instruct_5t.yaml` and `lora_rh_m4_instruct_5t.yaml`

For each file, note:
- What tensors are allocated and their shapes (given Llama-3.2-1B dimensions + sequence length up to 6500)
- Where `torch.no_grad()` is or isn't used
- Where intermediate activations are retained unnecessarily
- Whether eval-time generation properly cleans up KV caches between samples
- Any tensor `.detach()` or `.cpu()` calls that are missing
- Any lists or dicts that accumulate tensors across steps/epochs without releasing them
- The interaction between PyTorch's autograd graph and evaluation code

### Phase 2 — Write a diagnostic report

Produce a structured report (`MEMORY_AUDIT.md` in the repo root) containing:

1. **Memory lifecycle timeline** — a step-by-step walkthrough of VRAM allocation from model load → first training step → first eval → steady-state training → periodic eval → checkpoint save. Identify the peak at each stage.
2. **Itemised VRAM budget** — estimate the memory cost of each major component:
   - Base model weights (frozen, fp16/bf16)
   - LoRA adapter weights (trainable)
   - Optimizer states (AdamW: 2× trainable params)
   - Gradient tensors
   - Forward pass activations (with and without gradient checkpointing)
   - Attention matrices during forward pass (training vs eval)
   - KV cache during generation (eval)
   - Tokenised batch data
   - Any other buffers (loss accumulators, metric tensors, logged scalars)
3. **Leak & fragmentation inventory** — list every location where tensors may be retained longer than necessary, where Python references prevent garbage collection, or where CUDA memory fragmentation is likely.
4. **Peak analysis** — identify the exact code path that causes the highest VRAM peak and explain why.

### Phase 3 — Optimisation plan

Based on the audit, produce a numbered list of **concrete changes** that reduce peak and steady-state VRAM usage. For each change:
- File and function to modify
- What the change does
- Estimated VRAM saving (even rough)
- Risk level (none / low / medium) for altering training dynamics
- Whether it's a code change or config change

**Hard constraints — do NOT propose any of the following:**
- Changing LoRA rank, alpha, target modules, or dropout
- Changing learning rate, optimizer, scheduler, or weight decay
- Changing the number of epochs, effective batch size, or eval interval  
- Changing the dataset, task set, or filtering parameters
- Changing the model (Llama-3.2-1B-Instruct must stay)
- Removing or weakening gradient checkpointing
- Anything that would alter the mathematical training trajectory (results must be bit-for-bit identical given the same seed)

**Acceptable optimisation categories:**
- Ensuring `torch.no_grad()` wraps all non-training forward passes
- Properly deleting/detaching tensors after use
- Clearing KV caches between eval samples
- Reducing CUDA memory fragmentation (e.g. `empty_cache()` placement, `PYTORCH_CUDA_ALLOC_CONF`)
- Using `torch.inference_mode()` where appropriate
- Ensuring eval-time tensors don't leak into the autograd graph
- Moving metric computation to CPU
- Optimising data loading / collation to avoid unnecessary GPU copies
- Any PyTorch memory-efficiency flags or environment variables
- Garbage collection hints (`gc.collect()`)
- Reducing eval generation `max_new_tokens` if it's set higher than needed
- Chunked or sequential evaluation (one task at a time with cleanup) if not already done
- `pin_memory`, `non_blocking` transfer optimisations
- Anything else that is purely a memory-management improvement with zero impact on training numerics

### Phase 4 — Apply the changes

Implement every change from the plan. For each:
1. State which file you're editing and why
2. Make the edit
3. Verify the edit doesn't introduce syntax errors (run `python -c "import grad_lora_finetuning.trainer"` etc.)

After all edits, produce a summary diff list (file → changes) at the end.

---

## Output expectations

- `MEMORY_AUDIT.md` in the repo root with the full diagnostic (Phases 1–3)
- All code changes applied in-place (Phase 4)
- A final summary of every file touched and what changed

Take your time. Read every file thoroughly before writing anything. The audit must be grounded in actual code, not assumptions.
