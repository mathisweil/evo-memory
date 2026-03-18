# LoRA Gradient Fine-Tuning of Base LLM Weights

Standard gradient-based fine-tuning of LLaMA 3.2-1B-Instruct via low-rank adapters (LoRA), using next-token prediction or supervised fine-tuning on LongBench tasks.

This is the **m1 condition** in the interaction study: LoRA-only training with no NAMM eviction active. It serves as the gradient-based counterpart to ES fine-tuning, updating a tiny fraction of parameters (~0.5%) through backpropagation rather than perturbing all 1.24B weights with evolutionary strategies.

---

## What is LoRA Gradient Fine-Tuning?

LoRA (Low-Rank Adaptation) injects small trainable matrices into frozen transformer layers. For each target linear layer (e.g., `q_proj`), the original weight matrix `W` is augmented with a low-rank decomposition:

```
W' = W + (B @ A) * (alpha / rank)
```

where `A` (rank x in_features) and `B` (out_features x rank) are the only trainable parameters. The base weight `W` remains frozen.

Key properties:
- **PEFT library** injects adapters into `q_proj` and `v_proj` attention projections across all 16 layers
- With rank=8, this adds ~0.5% trainable parameters relative to the 1.24B base model
- LoRA A/B matrices are forced to **float32** regardless of base model dtype (bfloat16 underflow at sigma=0.001 scale)
- The `lm_head` is explicitly frozen and never included in the LoRA parameter set

### NTP vs SFT Modes

Two training objectives are supported:

| Mode      | Dataset            | Loss                   | Use case              |
| --------- | ------------------ | ---------------------- | --------------------- |
| **NTP**   | `LBNTPDataset`     | CE on all tokens       | General long-doc LM   |
| **SFT**   | `LBSFTDataset`     | CE on answer tokens    | Task-specific answers  |

NTP mode tokenises each document's `context` field and left-truncates to `max_seq_len`. SFT mode wraps the full prompt in `apply_chat_template`, appends the gold answer, and masks the prompt portion in labels so only answer tokens contribute to the loss.

### How it Differs from ES

ES fine-tuning perturbs **all** 1.24B base model parameters using population-based search with no backward pass. LoRA fine-tuning uses standard backpropagation through a tiny adapter subspace (~6M parameters at rank=8). The gradient signal is exact rather than estimated from reward-noise correlation.

---

## Parameters

### LoRA hyperparameters (CLI / YAML)

| Parameter                | Default            | Meaning                        |
| ------------------------ | ------------------ | ------------------------------ |
| `--run_name`             | (required)         | Name for this run              |
| `--config`               | `None`             | YAML config file               |
| `--experiment`           | auto               | Experiment ID, or creates new  |
| `--method`               | `lora_grad`        | Training method identifier     |
| `--lora_rank`            | `8`                | Rank of LoRA decomposition     |
| `--lora_target_modules`  | `[q_proj, v_proj]` | Layers receiving LoRA adapters |
| `--lora_alpha`           | `None` (= rank)    | LoRA scaling (alpha / rank)    |
| `--lora_dropout`         | `0.0`              | Dropout on LoRA outputs        |
| `--learning_rate`        | `2e-4`             | AdamW learning rate            |
| `--weight_decay`         | `0.01`             | AdamW weight decay             |
| `--max_grad_norm`        | `1.0`              | Gradient clipping threshold    |
| `--warmup_ratio`         | `0.03`             | Cosine warmup fraction         |
| `--num_epochs`           | `3`                | Training epochs                |
| `--batch_size`           | `1`                | Micro-batch size               |
| `--grad_accum_steps`     | `16`               | Steps before optim update      |
| `--max_seq_len`          | `3500`             | Max token seq length           |
| `--sft_mode`             | `False`            | Answer-only loss masking       |

### Data filtering and splits

| Parameter                | Default | Meaning                          |
| ------------------------ | ------- | -------------------------------- |
| `--filter_by_tokens`     | `None`  | Drop samples > N tokens         |
| `--filter_answers_by_tokens` | `64`| Drop samples with long answers   |
| `--train_split`          | `0.8`   | Fraction for training            |
| `--val_split`            | `0.1`   | Fraction for validation          |
| `--split_seed`           | `42`    | Seed for split reproducibility   |

### Evaluation and checkpointing

| Parameter             | Default | Meaning                                     |
| --------------------- | ------- | ------------------------------------------- |
| `--eval_interval`     | `40`    | Run F1 eval every N gradient update steps   |
| `--log_interval`      | `10`    | Print metrics every N gradient update steps |
| `--batch_size_eval`   | `None`  | Inference batch size for F1 eval            |
| `--gcs`               | `True`  | Enable GCS experiment management            |
| `--resume_checkpoint` | `None`  | Path to checkpoint for resuming training    |

### Model/task config (from Hydra)

| Parameter      | Default         | Meaning                        |
| -------------- | --------------- | ------------------------------ |
| `--run_config` | `namm_bam_i1_*` | Hydra config for model/task    |
| `--cache_size` | `None`          | Override cache size (for NAMM) |
| `--override`   | `[]`            | Extra Hydra config overrides   |

---

## Quick Start

**LoRA NTP training (default):**
```bash
python scripts/run_lora.py \
    --config scripts/lora_default.yaml \
    --run_name m1_ntp_test
```

**LoRA SFT training (answer-only loss):**
```bash
python scripts/run_lora.py \
    --config scripts/lora_default.yaml \
    --run_name m1_sft_test \
    --sft_mode
```

**Quick smoke test:**
```bash
python scripts/run_lora.py \
    --run_name smoke \
    --num_epochs 1 \
    --eval_interval 5 \
    --max_seq_len 512
```

**Resume from checkpoint:**
```bash
python scripts/run_lora.py \
    --config scripts/lora_default.yaml \
    --run_name m1_test \
    --resume_checkpoint results/lora_grad/42/ckpt.pt
```

---

## Training Workflow

```
# 1. Load LLaMA 3.2-1B-Instruct in bfloat16
# 2. Enable gradient checkpointing (recompute activations to save VRAM)
# 3. Inject LoRA adapters via PEFT into q_proj/v_proj (all 16 layers)
# 4. Force LoRA A/B matrices to float32
# 5. Freeze everything except LoRA params (including lm_head)
# 6. Build AdamW optimizer over LoRA params only
# 7. Build cosine LR scheduler with linear warmup

# Baseline evaluation on val set (before training)
baseline_f1 = evaluate_f1(split='val')

for epoch in range(num_epochs):                       # 3 epochs
    for batch in dataloader:                           # batch_size=1
        # Forward pass -> get hidden states (skip_lm_head=True)
        hidden_states = model(input_ids, skip_lm_head=True)

        # Chunked cross-entropy over lm_head (512 tokens at a time)
        loss = chunked_cross_entropy(lm_head, hidden_states, labels)

        # Scale by gradient accumulation and backward
        (loss / gradient_accumulation_steps).backward()

        if accumulated == gradient_accumulation_steps:  # every 16 micro-batches
            clip_grad_norm_(lora_params, max_grad_norm=1.0)
            optimizer.step()                            # AdamW update
            scheduler.step()                            # cosine LR decay
            optimizer.zero_grad()
            global_step += 1

            if global_step % eval_interval == 0:        # every 40 steps
                val_f1 = evaluate_f1(split='val')
                save_best_checkpoint_if_improved(val_f1)
                save_rolling_checkpoint()

# Load best checkpoint (highest val F1) for final test evaluation
final_test_f1 = evaluate_f1(split='test')
```

### Why `skip_lm_head=True` and chunked cross-entropy?

LLaMA 3.2 has a 128,256-token vocabulary. Computing `lm_head(hidden_states)` materialises a `[batch, seq_len, 128256]` logits tensor, which causes OOM on consumer GPUs. Instead:

1. The model returns raw hidden states (`skip_lm_head=True`)
2. The trainer applies `lm_head` in chunks of 512 tokens at a time
3. Cross-entropy is computed per chunk and summed, then divided by total non-masked tokens

This keeps peak memory proportional to `512 * 128256` rather than `seq_len * 128256`.

---

## Data Pipeline

### How TaskSampler provides the split

The same `TaskSampler` used by ES fine-tuning provides a deterministic train/val/test split:

```
TaskSampler.apply_train_val_test_split(train_frac=0.8, val_frac=0.1)
  -> Per task: 80% train / 10% val / 10% test indices
  -> Deterministic via split_seed=42
  -> Same split as ES, enabling fair comparison
```

Pre-split filtering:
1. `filter_by_tokens`: drops samples exceeding a token count threshold
2. `filter_answers_by_tokens` (default 64): drops samples whose shortest gold answer exceeds this length

### How datasets consume data

**NTP mode** (`LongBenchNTPDataset`):
- Loads `context` field from LongBench HuggingFace dataset
- Tokenises and left-truncates to `max_seq_len` (3500)
- All tokens supervised (standard next-token prediction)
- `ntp_pad_collate_fn` right-pads to batch-max length, masks padding with -100

**SFT mode** (`LongBenchSFTDataset`):
- Loads per-task prompt templates from `data/longbench/dataset2prompt.json`
- Wraps prompt in `apply_chat_template`, appends gold answer
- Records `label_start` boundary between prompt and answer
- Samples exceeding `max_seq_len` are discarded (not truncated)
- `sft_pad_collate_fn` masks prompt + padding positions with -100, supervises answer tokens only

### Evaluation via TaskSampler

F1 evaluation uses the same generation-based pipeline as ES:
1. `_set_split_indices(split='val')` loads val indices from TaskSampler
2. `task_sampler.evaluate()` runs full inference (generation + F1 scoring)
3. Weighted average F1 across tasks -> `lb/avg_f1`
4. Best checkpoint selected by highest val F1

---

## Comparing LoRA with ES

| Aspect             | ES Fine-Tuning               | LoRA Grad FT                  |
| ------------------ | ---------------------------- | ----------------------------- |
| **Optimised**      | All 1.24B base params        | ~6M LoRA A/B (~0.5%)          |
| **Optimiser**      | NES (population)             | AdamW (gradient)              |
| **Grad compute**   | Reward-noise estimate        | Exact backprop                |
| **Fwd passes**     | ~6,400 (50x8x16)            | ~300-500 (3 epochs)           |
| **Bwd passes**     | 0                            | Same as fwd                   |
| **Memory**         | Low (no activations)         | Higher (activations stored)   |
| **Grad ckpt**      | N/A                          | Enabled (recompute)           |
| **Non-diff ops**   | Can optimise through NAMM    | Cannot (two-phase fwd)        |
| **Convergence**    | Slow (pop=8, 1.24B-dim)     | Fast (exact, 6M-dim)         |
| **Precision**      | bfloat16                     | float32 LoRA + bf16 base      |
| **Validation**     | Baseline + final only        | Every eval_interval; best ckpt|
| **Data split**     | TaskSampler, seed=42         | TaskSampler, seed=42          |

---

## Troubleshooting

### OOM during training
- Reduce `--max_seq_len` (default 3500; try 2048 or 1024)
- Reduce `--batch_size` to 1 (already default)
- Ensure gradient checkpointing is active (automatic when `--namm_active` is not set)
- The chunked lm_head (512 tokens) should prevent logits OOM; if not, the chunk size can be reduced in `trainer.py`

### OOM during evaluation
- Set `--batch_size_eval` to a smaller value
- Evaluation runs under `torch.no_grad()` so memory usage is lower than training

### LoRA parameters in bfloat16 (underflow)
- `apply_lora_adapters()` forces all `requires_grad=True` parameters to float32
- If you see near-zero gradients, verify LoRA params are float32: `print(next(p for p in model.parameters() if p.requires_grad).dtype)`

### Gradient checkpointing errors
- Gradient checkpointing is **incompatible with NAMM** (`use_cache=True` required for KV-cache eviction)
- It is automatically disabled when `--namm_active` is set
- If you see `use_reentrant` warnings, they are expected; the trainer uses `use_reentrant=False`

### All val F1 scores are 0.0
- Check that `filter_answers_by_tokens` is not filtering out all samples
- Check the debug generation output -- if predictions are empty or garbage, the model may need more training steps
- Verify the model is in `eval()` mode during evaluation (the trainer handles this automatically)

### Resume not restoring optimizer state
- Resume loads LoRA weights, AdamW state, and scheduler state from the checkpoint
- The checkpoint must contain `optimizer_state_dict` and `scheduler_state_dict` keys
- If resuming from a best checkpoint (`best_ckpt.pt`), optimizer/scheduler state is included

---

## Key Files

- `scripts/run_lora.py` -- entry point and argument parser
- `scripts/lora_default.yaml` -- default configuration
- `grad_lora_finetuning/trainer.py` -- LoRAGradTrainer class (training loop, eval, checkpointing)
- `grad_lora_finetuning/datasets.py` -- LongBenchNTPDataset, LongBenchSFTDataset, collate functions
- `namm/llms/llama.py` -- `apply_lora_adapters()` method (PEFT injection, float32 casting)
- `namm/tasks.py` -- TaskSampler with train/val/test split support
- `es_finetuning/device.py` -- device abstraction (TPU > CUDA > CPU)
- `es_finetuning/gcs.py` -- GCS experiment management and checkpointing
