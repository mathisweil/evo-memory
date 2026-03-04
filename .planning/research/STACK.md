# Technology Stack

**Project:** NAMM + Gradient-Based LoRA Study (v2.0)
**Researched:** 2026-03-03
**Scope:** NEW components needed for gradient-based LoRA training milestone. Supersedes v1.0 STACK.md (ES-based LoRA), which is now out of scope per GROUP DECISION. Does not re-document PyTorch 2.7, Transformers 4.41.2, PEFT 0.11.1, Hydra, wandb — all already present in `th2`.

---

## Confirmed Environment (th2 conda env)

| Package | Installed Version | Notes |
|---------|------------------|-------|
| PyTorch | 2.7.0+cu128 | Measured |
| Transformers | 4.41.2 | Measured |
| PEFT | 0.11.1 | Measured — LoRA injection already validated in Phase 2 |
| datasets | 2.20.0 | Measured — already used for LongBench loading |
| accelerate | 1.12.0 | Measured — used in evaluator |
| bitsandbytes | 0.49.2 | Measured — NOT used for LoRA here |
| numpy | 1.26.4 | Measured |
| omegaconf | 2.3.0 | Measured |

---

## New Stack Components

### 1. Optimizer: `torch.optim.AdamW` (already installed, new usage)

**No new install required.** AdamW is in PyTorch 2.7.

**Recommended configuration for gradient-based LoRA on LLaMA 3.2-1B:**

```python
# Only optimize LoRA params — base model and NAMM weights stay frozen
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    lora_params,
    lr=2e-4,            # standard LoRA range: 1e-4 to 3e-4; start here
    betas=(0.9, 0.95),  # beta2=0.95 preferred over 0.999 for LLM finetuning
    eps=1e-8,
    weight_decay=0.01,  # small weight decay prevents LoRA weight explosion
)
```

**Why these values:**
- `lr=2e-4`: Lightning AI's 300-experiment LoRA study found 3e-4 optimal; 2e-4 is slightly more conservative for a 1B model starting from an already-capable base. Range 1e-4–3e-4 confirmed valid across multiple independent experiments.
- `betas=(0.9, 0.95)`: The Llama-3 paper and most 1B-scale finetuning uses beta2=0.95, not the Adam default of 0.999. Lower beta2 makes the optimizer more responsive to recent gradient estimates, which matters when adapting rapidly with few training steps.
- `weight_decay=0.01`: Prevents LoRA matrices from growing unboundedly. Do not skip — LoRA A and B matrices can diverge during long training runs without it.
- **Do not use paged AdamW / 8-bit Adam**: bitsandbytes optimizers add quantization that destabilizes LoRA training at the small parameter counts used here (327k params). Standard AdamW at this scale costs ~2.5 MB extra VRAM — negligible.

**Learning rate scheduler:**

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,      # ~5-10% of total steps
    num_training_steps=500,   # total gradient steps; tune per run
)
```

`get_cosine_schedule_with_warmup` is already importable from `transformers` (verified in th2). Use over PyTorch's built-in `CosineAnnealingLR` because it handles the warmup phase in one call and matches the HF ecosystem convention.

**Gradient clipping:**

```python
torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
```

Apply before `optimizer.step()`. Max norm 1.0 is standard for LLM LoRA. Without clipping, early training steps with poorly-initialized LoRA can produce large gradients that destabilize the base model's representations via the LoRA residual path.

---

### 2. DataLoader: `torch.utils.data.DataLoader` + Custom `Dataset` (already installed, new usage)

**No new install required.** Standard PyTorch DataLoader is sufficient.

**The problem:** QASPER documents are 1440–14660 tokens (measured: mean 3619, median 3418). NTP loss requires: (1) full document as input (prefix), (2) shifted-by-one as target. Padding a batch to max_len wastes 3–5x compute. Variable-length batching is needed.

**Recommended pattern — packed NTP sequences:**

```python
from torch.utils.data import Dataset, DataLoader

class QASPERNTPDataset(Dataset):
    """Returns a single tokenized document as (input_ids, labels).

    input_ids: [1, seq_len] — full document tokens
    labels:    [1, seq_len] — same, shifted by 1 in loss function
    """
    def __init__(self, hf_dataset, tokenizer, max_length=4096, prompt_template=None):
        self.samples = []
        for item in hf_dataset:
            # Use context + question as the document for NTP
            text = item['context'] + '\n' + item['input']
            if prompt_template is not None:
                text = prompt_template.format(**item)
            ids = tokenizer(text, return_tensors='pt',
                            truncation=True, max_length=max_length).input_ids
            self.samples.append(ids.squeeze(0))  # [seq_len]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def ntp_collate_fn(batch):
    """Pad sequences in batch to same length for batched forward pass."""
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(batch, batch_first=True, padding_value=0)
    # labels: same as input_ids — cross-entropy loss handles shift internally
    # by computing loss[i] = CrossEntropy(logits[i-1], input_ids[i])
    labels = input_ids.clone()
    labels[labels == 0] = -100  # mask padding from loss
    return input_ids, labels

loader = DataLoader(
    dataset, batch_size=1,  # batch_size=1 per sequence; use grad accumulation
    shuffle=True, collate_fn=ntp_collate_fn, num_workers=0,
)
```

**Why batch_size=1 with gradient accumulation, not batch_size > 1:**
- QASPER sequences range from 1440 to 14660 tokens. Padding a batch of 4 to 14660 tokens each wastes 4x compute on short documents.
- A single 4096-token sequence forward pass with gradient tracking uses ~8–10 GB VRAM on the 4070Ti (12 GB). Two sequences simultaneously risks OOM.
- Use `accumulation_steps=4–8` to get equivalent batch statistics without padding waste.

**Gradient accumulation pattern:**

```python
optimizer.zero_grad()
for accum_step, (input_ids, labels) in enumerate(loader):
    with torch.enable_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / accumulation_steps
    loss.backward()
    if (accum_step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

**IMPORTANT NOTE on datasets 2.20.0:** The existing `task_sampler.py` already loads QASPER via `load_dataset('THUDM/LongBench', 'qasper', split='test', trust_remote_code=True)`. Reuse this existing dataset object — do not load a second copy. The NTP DataLoader wraps the same HF dataset object; no additional downloads.

---

### 3. Gradient-Based Training: `torch.enable_grad()` + PEFT `enable_input_require_grads` pattern

**Critical context:** Every method in `memory_trainer.py` is decorated with `@torch.no_grad()`. The gradient-based LoRA training loop CANNOT be implemented inside `MemoryTrainer` — it operates in no-grad mode by design. This is a feature (ES training never needs gradients), not a bug, but it means gradient LoRA training must live in a **separate trainer class**.

**Required setup for gradient flow through PEFT LoRA in a frozen base model:**

```python
# After apply_lora_adapters() is called:

# 1. Verify LoRA params have requires_grad=True (PEFT sets this)
lora_params = [p for p in model.parameters() if p.requires_grad]
assert len(lora_params) > 0, "No trainable LoRA params found"

# 2. PEFT on frozen base models sometimes loses gradient flow at the first
#    frozen layer. Fix with enable_input_require_grads pattern:
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
# This is the documented PEFT workaround for frozen-base LoRA training
# (equivalent to model.enable_input_require_grads() in newer PEFT versions)
# Confirmed: PeftModel in v0.11.1 does NOT have enable_input_require_grads()
# as a method — the hook pattern above is the correct workaround.

# 3. Run forward pass WITH gradients (outside any torch.no_grad() context)
with torch.enable_grad():
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
loss.backward()
```

**NAMM eviction is compatible with gradient-based LoRA training** because:
- NAMM's `update_cache()` is decorated with `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` and operates on the KV cache in-place. The KV cache tensors are not in the autograd graph (they are detached after each generation step).
- Gradient flow only needs to reach LoRA's A and B matrices via the query/value projections. The eviction happens after the attention output is computed — it does not block gradient flow to q_proj or v_proj weights.
- **Mode frozen**: when NAMM is active but frozen (m4), its parameters have `requires_grad=False`. The eviction policy still runs (modifying the KV cache), but no NAMM gradients are computed. This is correct behavior.

**Gradient checkpointing for long sequences (optional, VRAM reduction):**

```python
# LlamaModel supports gradient checkpointing (verified: supports_gradient_checkpointing=True)
model.model.base_model.gradient_checkpointing_enable()
# Cost: ~30-50% more compute per step (recomputes activations during backward)
# Benefit: ~40-50% VRAM reduction — enables longer sequences or larger batches
# Enable only if 4096-token sequences cause OOM without it
```

**When to enable gradient checkpointing:** Only if a full 4096-token sequence forward+backward causes OOM on the 4070Ti (12 GB). At 1B params + 327k LoRA params + optimizer states, memory budget is tight for very long sequences but likely fits without checkpointing at batch_size=1.

---

### 4. NTP Loss: `CrossEntropyLoss` via HuggingFace Transformers `labels` argument (already installed)

**No new install.** `WrappedLlamaForCausalLM.forward()` already accepts `labels` and returns `outputs.loss`.

**How it works:** When `labels` is passed, `LlamaForCausalLM` internally computes:
```python
# Shift logits and labels for next-token prediction
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = CrossEntropyLoss()(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```
Tokens where `label == -100` are masked from loss (use -100 for padding tokens in labels).

**Integration point:** The existing `WrappedLlamaForCausalLM.forward()` signature already has `labels: Optional[torch.LongTensor] = None`. It delegates to the HF `LlamaForCausalLM.forward()`. Pass labels to get loss automatically. No code changes needed for the NTP loss function itself.

**IMPORTANT:** The existing codebase has `assert labels is None, 'Tensor splitting has not been tested for training'` in the split-processing path of `WrappedLlamaForCausalLM.forward()` (line 360). The gradient LoRA trainer must NOT use `limit_new_tokens` / split processing. Use the full context in a single forward pass — this is what gradient training requires anyway (full backprop through the whole sequence).

---

### 5. Analysis Metrics: `torch` built-ins (no new libraries)

**No new installs needed.** All analysis metrics can be implemented with pure PyTorch:

**Attention entropy:**

```python
def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    attn_weights: [batch, heads, seq_q, seq_k] — softmax outputs from output_attentions=True
    Returns: [batch, heads, seq_q] — entropy per query position
    """
    # Clamp to avoid log(0); softmax outputs are already in [0,1]
    p = attn_weights.clamp(min=1e-9)
    return -(p * p.log()).sum(dim=-1)  # Shannon entropy
```

**How to obtain attention weights from this model:** Pass `output_attentions=True` to `WrappedLlamaForCausalLM.forward()`. The model propagates this flag to all attention layers. Standard LLaMA attention returns softmax-normalized weights when `output_attentions=True` (verified from Transformers 4.41.2 source: `attn_weights = softmax(...); return attn_output, attn_weights, past_key_value`). The NAMM custom attention layers in `stateless_parallel_modules/attention.py` use `F.scaled_dot_product_attention` with `is_causal=True` — this does NOT return attention weights (SDPA's efficient kernel doesn't expose them). To get attention weights for analysis, use `output_attentions=True` on the standard HF attention path (which `WrappedLlamaForCausalLM` uses for the base LLaMA layers, not the NAMM policy network itself).

**Layer-wise token retention:**

```python
def token_retention_rate(key_cache_before, key_cache_after) -> float:
    """
    Compare KV cache token counts before and after NAMM eviction.
    key_cache_before: [batch, heads, seq_before, head_dim]
    key_cache_after:  [batch, heads, seq_after,  head_dim]
    Returns fraction of tokens retained: seq_after / seq_before
    """
    return key_cache_after.shape[-2] / key_cache_before.shape[-2]
```

Access the KV cache sizes from `past_key_values` before and after `memory_policy.update_cache()` runs. This is already observable in the existing forward pass — no code changes to the model needed, only logging hooks in the trainer.

**Why no external analysis library:**
- TransformerLens (widely recommended for mechanistic interpretability) requires converting the model to its `HookedTransformer` format. This would break NAMM's custom attention layers and the existing KV-cache eviction infrastructure. The cost of converting exceeds the benefit for this project's analysis questions.
- BertViz is Jupyter-notebook-only visualization; not useful for quantitative metrics logged to wandb.
- Pure PyTorch entropy + retention calculations are 5 lines each and produce exactly the scalars needed for wandb logging without any framework dependency.

---

### 6. Pipeline Orchestration: Hydra `multirun` + shell script (already installed)

**No new install.** Hydra is already the config system.

**For m3 (LoRA then NAMM) and m4 (NAMM then LoRA) multi-stage pipelines:**

```yaml
# cfgs/run/pipeline_m3.yaml — Stage 1: gradient LoRA (NAMM frozen/absent)
# cfgs/run/pipeline_m4.yaml — Stage 1: CMA-ES NAMM, Stage 2: gradient LoRA with NAMM active
```

Orchestration pattern: sequential shell script calling `python main.py` twice with different configs, passing the Stage 1 checkpoint path as `init_from` for Stage 2. Hydra already supports `init_from` for checkpoint loading.

```bash
#!/usr/bin/env bash
# run_m3_pipeline.sh
python main.py run=pipeline_m3_stage1 init_from=null out_dir=${OUT_DIR}/stage1
python main.py run=pipeline_m3_stage2 init_from=${OUT_DIR}/stage1/ckpt.pt out_dir=${OUT_DIR}/stage2
```

**Why not Hydra multirun for this:** Hydra multirun runs configs in parallel or sequence but does not pass outputs (checkpoint paths) between runs. A simple shell script calling two separate `python main.py` invocations is cleaner and more debuggable. The checkpoint path between stages is a runtime value that multirun cannot wire automatically.

**Config key for NAMM active/frozen/absent:**

```yaml
# namm_mode: active | frozen | absent
# - active:  NAMM evicts KV cache tokens during LoRA training forward pass
# - frozen:  NAMM policy weights are loaded but requires_grad=False; eviction still runs
# - absent:  policy=none, cache_size=4096 (full cache, no eviction)
namm_mode: active
```

This single key replaces the need for separate code paths for m1–m4. The gradient LoRA trainer checks `namm_mode` at init and sets NAMM policy parameters accordingly.

---

## Recommended LoRA Configuration for Gradient Training

Same PEFT config as Phase 2, but with one critical addition for gradient training:

```python
peft_config = LoraConfig(
    r=4,
    target_modules=['q_proj', 'v_proj'],
    lora_alpha=4,           # alpha=rank: no effective scaling (gamma=1.0)
    lora_dropout=0.05,      # ADD dropout for gradient training — regularization
    bias='none',
    # NOTE: lora_dropout=0.0 was correct for ES (no gradient flow)
    # For gradient-based training, small dropout (0.05) prevents overfitting
    # on the small NTP dataset (200 QASPER documents)
)
```

**Change from ES config:** `lora_dropout=0.05` (was 0.0). Dropout is useless for ES (gradient-free). For gradient training, even minimal dropout matters because the training set is only 200 QASPER documents — regularization is critical to prevent memorization.

**Keep float32 for LoRA weights:** The existing `apply_lora_adapters()` already forces float32 LoRA weights. This is correct for gradient training too — gradient accumulation in bfloat16 with LoRA at r=4 produces underflow at small learning rates. Keep the float32 cast.

---

## Compute Budget Estimates (4070Ti, gradient training)

| Operation | VRAM Usage | Time per Step |
|-----------|-----------|---------------|
| Forward pass, 4096-token sequence, no grad | ~6 GB | ~0.8s |
| Forward + backward, 4096-token sequence | ~10–11 GB | ~2.5s |
| Forward + backward with grad checkpointing | ~6–7 GB | ~4.5s |
| AdamW optimizer step (327k LoRA params) | +10 MB | negligible |

**Estimated training time:**
- 500 gradient steps × batch_size=1 × accum=4 ≈ 125 effective batches
- ~2.5s/step × 500 steps ≈ **21 minutes** per training run on 4070Ti
- This is 8–10x faster than a 200-iter CMA-ES NAMM run (~3h)
- m4 pipeline total (Stage 1 NAMM ~3h + Stage 2 LoRA ~21min) ≈ **3.5h**

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| trl (TRL library) | Heavy dependency, incompatible with NAMM's custom attention forward pass; SFTTrainer wraps the model in ways that break `WrappedLlamaForCausalLM`'s KV eviction logic | Manual training loop (50 lines) |
| TransformerLens | Requires converting model to HookedTransformer — breaks NAMM attention layers | Pure PyTorch attention entropy (5 lines) |
| BertViz | Notebook-only visualization; no wandb integration | wandb.log() with entropy scalars |
| 8-bit/paged AdamW | Quantized optimizer destabilizes 327k-param LoRA training | Standard torch.optim.AdamW |
| QLoRA / bitsandbytes quantization | Incompatible with NAMM's stateless attention layers and pop-member evaluation pattern | Standard fp32 LoRA |
| Accelerate Trainer | Adds DDP complexity not needed for single-GPU; incompatible with NAMM's multi-pop-member evaluation loop | Single-GPU torch training loop |
| PEFT upgrade to 0.12.x+ | Unknown compatibility with `WrappedLlamaForCausalLM`; 0.11.1 is validated | Keep 0.11.1 |

---

## Integration Points with Existing Codebase

| New Component | Integration Point | What Changes |
|---------------|-----------------|-------------|
| `LoRAGradTrainer` (new class in `memory_trainer.py` or separate file) | Separate from `MemoryTrainer` — MUST NOT use `@torch.no_grad()` | New class; existing `MemoryTrainer` untouched |
| `QASPERNTPDataset` (new in `task_sampler.py` or separate file) | Reuses `load_dataset` result from existing `TaskSampler` | New class; wraps existing HF dataset object |
| AdamW optimizer | Only operates on LoRA params (`p.requires_grad=True`) | Instantiated in `LoRAGradTrainer.__init__` |
| `get_cosine_schedule_with_warmup` | Called after AdamW init in `LoRAGradTrainer.__init__` | New import, already in transformers |
| `enable_input_require_grads` hook | Applied once after `apply_lora_adapters()` in `LoRAGradTrainer.__init__` | Forward hook on embedding layer |
| Attention entropy logging | `output_attentions=True` flag on periodic eval forward pass | No model changes; periodic logging in trainer |
| Token retention logging | Log `past_key_values[i][0].shape[-2]` before/after `update_cache()` | Instrument `memory_policy/base.py:update_cache` with optional logging callback |
| `namm_mode` Hydra key | Checked in `LoRAGradTrainer.__init__` to freeze/unfreeze NAMM policy params | New config key in pipeline yaml configs |

---

## Version Compatibility

| Package | Version | Compatibility Notes |
|---------|---------|---------------------|
| PEFT 0.11.1 | + Transformers 4.41.2 | Validated in Phase 2; do not upgrade either |
| PyTorch 2.7 | + CUDA 12.8 | Installed on sideswipe/prowl; gradient checkpointing works |
| datasets 2.20.0 | + QASPER NTP DataLoader | `load_dataset(..., trust_remote_code=True)` required for LongBench |
| torch.optim.AdamW | PyTorch 2.7 built-in | `fused=True` option available for speed; test before enabling |
| `get_cosine_schedule_with_warmup` | Transformers 4.41.2 | Already available; import from `transformers` |

---

## Sources

- Direct codebase inspection: `memory_trainer.py` (all methods `@torch.no_grad()`), `memory_llms/llama.py` (existing `apply_lora_adapters()`), `stateless_parallel_modules/attention.py` (SDPA without attention weight output), `memory_policy/base.py` (`update_cache` with `@custom_fwd`) — HIGH confidence
- PyTorch 2.7 docs: `torch.optim.AdamW` signature verified — HIGH confidence
- Transformers 4.41.2 source: LlamaAttention output_attentions behavior verified — HIGH confidence
- Measured QASPER token lengths: LongBench dataset, 200 samples, mean=3619, max=14660 — HIGH confidence
- PEFT 0.11.1: `PeftModel.enable_input_require_grads` does NOT exist (verified); forward hook pattern is correct workaround — HIGH confidence
- LM gradient checkpointing: `LlamaModel.supports_gradient_checkpointing=True` verified in th2 — HIGH confidence
- AdamW lr=2-3e-4 for LoRA: Lightning AI 300-experiment study + multiple corroborating sources — MEDIUM confidence (single-task GPT finetuning, not QASPER long-context; treat as starting point)
- Beta2=0.95 for LLM finetuning: Llama-3 paper + community consensus — MEDIUM confidence
- Compute estimates: extrapolated from measured 48s/iter CMA-ES run; gradient training not yet measured — LOW confidence (will update after first gradient training run)

---

*v2.0 stack research for: NAMM + Gradient-Based LoRA Study*
*Researched: 2026-03-03*
*Previous STACK.md (v1.0, ES-based LoRA) content archived in git history; ES-based LoRA is now out of scope.*
