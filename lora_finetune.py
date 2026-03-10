"""
LoRA fine-tuning of Llama 3.2 1B with SFT, matching the ES fine-tuning setup.

Training: supervised fine-tuning on (prompt, answer) pairs from Qasper,
          with cross-entropy loss only on answer tokens.
Evaluation: generation-based F1 (same pipeline as ES fine-tuning).

Supports two modes:
  m1: LoRA SFT without NAMM (no KV cache eviction during training)
  m4: LoRA SFT with NAMM active as a frozen eviction policy

Usage:
  python lora_finetune.py --config-name=config +run=llama32_qasper_lora_m1
  python lora_finetune.py --config-name=config +run=llama32_qasper_lora_m4
"""

import os
import time
import copy
import random
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from main import (
    make_eval_model, make_task_sampler, stochasticity_setup,
    get_dist_info, wandb_init,
)
from utils import empty_gpu_cache


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

def inject_lora(memory_model, cfg):
    """Inject LoRA adapters in-place and freeze non-LoRA parameters."""
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
    )

    inject_adapter_in_model(lora_config, memory_model.model,
                            adapter_name="default")

    # Freeze everything, then unfreeze only LoRA params
    for param in memory_model.parameters():
        param.requires_grad = False
    lora_params = []
    for name, param in memory_model.model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)

    # Enable gradient checkpointing to reduce memory usage
    if hasattr(memory_model.model, 'gradient_checkpointing_enable'):
        memory_model.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in memory_model.parameters())
    print(f"LoRA: {n_lora:,} trainable params / {n_total:,} total "
          f"({100 * n_lora / n_total:.2f}%)")

    return lora_params


# ---------------------------------------------------------------------------
# NAMM checkpoint loading (simplified, matching ES branch style)
# ---------------------------------------------------------------------------

def load_namm_checkpoint(memory_model, memory_policy, ckpt_path, device):
    """Load a trained NAMM checkpoint and set best params on the model."""
    print(f"Loading NAMM checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    evo_state = ckpt['evolution_state']

    best_member = evo_state['best_member']
    params = best_member.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)

    # Load stored normalization buffers (EMA mean/var for embeddings)
    buffers_prefix = 'stored_buffers_to_save.'
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)
        print(f"  Loaded {len(buffers_dict)} normalization buffers")

    print(f"  Loaded NAMM best_member ({best_member.shape[0]} params) "
          f"from iter {ckpt.get('iter_num', '?')}")


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def split_task_samples(task_sampler, train_ratio=0.75, seed=42):
    """Split samples for tasks that appear in both training and test subsets.

    For overlapping tasks (e.g. qasper used for both train and eval):
    - First `train_ratio` fraction goes to training
    - Remaining goes to evaluation (held-out)

    Non-overlapping tasks are left untouched.
    Stores train-only data in task_sampler.lb_train_prompts_per_task / _jsons.
    Replaces lb_prompts_per_task / _jsons with test-only data for overlapping tasks.
    """
    rng = random.Random(seed)
    overlap_tasks = set(task_sampler.lb_training_tasks) & set(task_sampler.lb_test_tasks)

    task_sampler.lb_train_prompts_per_task = {}
    task_sampler.lb_train_jsons_per_task = {}
    task_sampler.num_train_prompts_per_lb_task = {}

    for task_name in task_sampler.lb_training_tasks:
        prompts = task_sampler.lb_prompts_per_task[task_name]
        jsons = task_sampler.lb_jsons_per_task[task_name]

        if task_name in overlap_tasks:
            # Shuffle with fixed seed for reproducibility
            indices = list(range(len(prompts)))
            rng.shuffle(indices)
            split_idx = int(len(indices) * train_ratio)
            train_idxs = indices[:split_idx]
            test_idxs = indices[split_idx:]

            task_sampler.lb_train_prompts_per_task[task_name] = [prompts[i] for i in train_idxs]
            task_sampler.lb_train_jsons_per_task[task_name] = [jsons[i] for i in train_idxs]
            task_sampler.num_train_prompts_per_lb_task[task_name] = len(train_idxs)

            # Replace the main lists with test-only data for evaluation
            task_sampler.lb_prompts_per_task[task_name] = [prompts[i] for i in test_idxs]
            task_sampler.lb_jsons_per_task[task_name] = [jsons[i] for i in test_idxs]
            task_sampler.num_prompts_per_lb_task[task_name] = len(test_idxs)

            print(f"Split {task_name}: {len(train_idxs)} train / {len(test_idxs)} test")
        else:
            # Non-overlapping: all samples go to training
            task_sampler.lb_train_prompts_per_task[task_name] = prompts
            task_sampler.lb_train_jsons_per_task[task_name] = jsons
            task_sampler.num_train_prompts_per_lb_task[task_name] = len(prompts)
            print(f"Train-only {task_name}: {len(prompts)} samples")


def wrap_prompts_chat_template(task_sampler, tokenizer):
    """Wrap all prompts in lb_prompts_per_task with the Llama 3 chat template.

    This replaces raw LongBench prompts with instruct-formatted prompts so the
    model sees its expected special tokens during both training and evaluation.
    Also wraps lb_train_prompts_per_task if it exists (after split_task_samples).
    """
    def _wrap(prompts_dict):
        total = 0
        for task_name in list(prompts_dict.keys()):
            raw_prompts = prompts_dict[task_name]
            wrapped = []
            for prompt in raw_prompts:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                wrapped.append(text)
            prompts_dict[task_name] = wrapped
            total += len(wrapped)
        return total

    n_eval = _wrap(task_sampler.lb_prompts_per_task)
    n_train = 0
    if hasattr(task_sampler, 'lb_train_prompts_per_task'):
        n_train = _wrap(task_sampler.lb_train_prompts_per_task)
    print(f"Wrapped {n_train} train + {n_eval} eval prompts with chat template")


# ---------------------------------------------------------------------------
# Data preparation: SFT pairs (prompt + answer)
# ---------------------------------------------------------------------------

def build_sft_dataset(task_sampler, tokenizer, cfg):
    """Build SFT dataset from (prompt, answer) pairs.

    Returns list of dicts with 'input_ids', 'labels', 'attention_mask'.
    Labels have -100 on prompt tokens (only answer tokens contribute to loss).
    """
    max_seq_len = cfg.lora.max_seq_len
    use_chat = cfg.lora.get('use_chat_template', False)
    samples = []
    n_no_answers = 0
    n_truncated = 0

    has_split = hasattr(task_sampler, 'lb_train_prompts_per_task')
    for task_name in task_sampler.lb_training_tasks:
        if has_split:
            prompts = task_sampler.lb_train_prompts_per_task[task_name]
            jsons = task_sampler.lb_train_jsons_per_task[task_name]
        else:
            prompts = task_sampler.lb_prompts_per_task[task_name]
            jsons = task_sampler.lb_jsons_per_task[task_name]

        for prompt, json_obj in zip(prompts, jsons):
            answers = json_obj.get("answers", [])
            if not answers:
                n_no_answers += 1
                continue
            # Use the first answer as the target
            answer = answers[0] if isinstance(answers, list) else answers

            if use_chat:
                # Prompt is already chat-template wrapped (done earlier).
                # Tokenize the full prompt (includes special tokens from template).
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                # Answer tokens + EOS
                answer_ids = tokenizer.encode(
                    answer, add_special_tokens=False)
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    answer_ids = answer_ids + [eos_id]
            else:
                # Raw LongBench prompt — original tokenization approach
                prompt_ids = tokenizer(
                    prompt, add_special_tokens=True).input_ids
                answer_ids = tokenizer(
                    " " + answer, add_special_tokens=False).input_ids

            prompt_len = len(prompt_ids)
            full_ids = prompt_ids + answer_ids

            # Skip samples that exceed max_seq_len
            if len(full_ids) > max_seq_len:
                n_truncated += 1
                continue

            input_ids = torch.tensor(full_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            # Labels: mask prompt tokens with -100
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            # Skip samples where prompt fills entire sequence (no answer tokens)
            if (labels != -100).sum() == 0:
                continue

            samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            })

    if len(samples) == 0:
        raise RuntimeError("No SFT samples found — check task config / answers")

    lengths = [s['input_ids'].shape[0] for s in samples]
    avg_len = sum(lengths) / len(samples)
    max_len = max(lengths)
    min_len = min(lengths)
    avg_answer_toks = sum((s['labels'] != -100).sum().item() for s in samples) / len(samples)
    print(f"Built {len(samples)} SFT samples "
          f"(skipped {n_no_answers} with no answers, "
          f"{n_truncated} exceeding max_seq_len={max_seq_len})")
    print(f"  Sequence lengths: min={min_len}, avg={avg_len:.0f}, max={max_len}")
    print(f"  Avg answer tokens: {avg_answer_toks:.0f}")
    return samples


def collate_sft(batch, pad_token_id=0):
    """Collate variable-length SFT samples with left-padding."""
    max_len = max(s['input_ids'].shape[0] for s in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, s in enumerate(batch):
        seq_len = s['input_ids'].shape[0]
        # Right-align (left-pad)
        input_ids[i, max_len - seq_len:] = s['input_ids']
        attention_mask[i, max_len - seq_len:] = s['attention_mask']
        labels[i, max_len - seq_len:] = s['labels']

    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# F1 evaluation (same as ES — generation-based)
# ---------------------------------------------------------------------------

def evaluate_f1(task_sampler, memory_evaluator, memory_policy,
                train=False, num_samples=None):
    """Run generation-based F1 evaluation, matching the ES evaluation pipeline.

    Args:
        train: If True, evaluate on training split; if False, on test split.
        num_samples: Number of samples per task (None = all).
    """
    # Set NAMM to use fixed params (index 0)
    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    # Cap num_samples to available samples to avoid np.random.choice error
    if num_samples is not None:
        tasks = task_sampler.lb_training_tasks if train else task_sampler.lb_test_tasks
        min_available = min(
            task_sampler.num_prompts_per_lb_task.get(t, 0) for t in tasks
        ) if tasks else 0
        if num_samples > min_available:
            num_samples = None  # eval all available

    score_dicts = task_sampler.evaluate(
        lm=memory_evaluator,
        train=train,
        evolved_model=False,
        pop_reps=1,
        resample_requests=True,
        sampled_requests_per_task=num_samples,
    )
    return score_dicts[0]  # e.g. {"lb/qasper": 45.2} (0-100 scale)


def debug_generate(task_sampler, memory_evaluator, memory_policy, n=10):
    """Generate on n samples and print prompt excerpt, prediction, and ground truth."""
    from utils_longbench import get_score

    tokenizer = memory_evaluator.tokenizer
    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    for task_name in task_sampler.lb_test_tasks:
        prompts = task_sampler.lb_prompts_per_task[task_name]
        jsons = task_sampler.lb_jsons_per_task[task_name]

        sample_idxs = list(range(min(n, len(prompts))))
        sample_prompts = [prompts[i] for i in sample_idxs]
        sample_jsons = [jsons[i] for i in sample_idxs]

        # Print token counts before generation
        print(f"\n{'='*60}")
        print(f"DEBUG GENERATION: {task_name} ({len(sample_prompts)} samples)")
        print(f"{'='*60}")
        for i, prompt in enumerate(sample_prompts):
            n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            n_chars = len(prompt)
            print(f"  Sample {i}: {n_tokens} tokens, {n_chars} chars")

        # Generate
        outputs = memory_evaluator.evaluate_lb(
            dataset_samples=sample_prompts,
            disable_tqdm=True,
            pop_reps=1,
        )

        for i, (prompt, json_obj, pred) in enumerate(
                zip(sample_prompts, sample_jsons, outputs)):
            answers = json_obj.get("answers", [])
            prompt_tail = prompt[-300:]
            print(f"\n--- Sample {i} ---")
            print(f"PROMPT (last 300 chars): ...{prompt_tail}")
            print(f"GROUND TRUTH: {answers}")
            print(f"PREDICTION:   '{pred}'")
            print(f"PRED LENGTH:  {len(pred)} chars")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sft(model, sft_samples, lora_params, optimizer, cfg,
              task_sampler, memory_evaluator, memory_policy,
              wandb_run=None):
    """SFT training with periodic F1 evaluation matching ES setup."""
    grad_accum = cfg.lora.grad_accum_steps
    max_grad_norm = cfg.lora.max_grad_norm
    eval_every = cfg.lora.eval_every
    checkpoint_every = cfg.lora.checkpoint_every
    num_epochs = cfg.lora.num_epochs
    batch_size = cfg.lora.batch_size
    mini_batch_size = cfg.lora.mini_batch_size
    val_samples = cfg.lora.val_samples
    use_namm = cfg.lora.namm_active
    tokenizer = memory_evaluator.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  pad_token_id was None, set to eos_token_id={tokenizer.eos_token_id}")
    pad_token_id = tokenizer.pad_token_id

    global_step = 0
    best_val_f1 = -1.0

    for epoch in range(num_epochs):
        # Shuffle samples each epoch
        indices = list(range(len(sft_samples)))
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_start in range(0, len(indices), batch_size):
            batch_idxs = indices[batch_start:batch_start + batch_size]
            batch = [sft_samples[i] for i in batch_idxs]
            input_ids, attention_mask, labels = collate_sft(
                batch, pad_token_id=pad_token_id)

            input_ids = input_ids.to(cfg.device)
            attention_mask = attention_mask.to(cfg.device)
            labels = labels.to(cfg.device)

            if use_namm:
                # Reset memory policy state
                model.memory_policy.rotary_offset.zero_()

            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=use_namm,
                    apply_memory_policy=use_namm,
                    limit_new_tokens=input_ids.shape[-1],
                )
                loss = outputs.loss / grad_accum

            loss.backward()
            n_batches += 1

            if n_batches % grad_accum == 0 or \
                    batch_start + batch_size >= len(indices):
                torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                batch_loss = loss.item() * grad_accum
                epoch_loss += batch_loss

                if wandb_run is not None:
                    wandb_run.log({
                        "lora/loss": batch_loss,
                        "lora/epoch": epoch,
                        "lora/global_step": global_step,
                    })

                if global_step % 20 == 0:
                    avg = epoch_loss / max(global_step -
                        (epoch * len(indices) // batch_size // grad_accum), 1)
                    print(f"  epoch {epoch} step {global_step} "
                          f"loss={batch_loss:.4f}")

                # Periodic F1 evaluation (matching ES eval_every)
                if eval_every > 0 and global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        val_scores = evaluate_f1(
                            task_sampler, memory_evaluator, memory_policy,
                            train=False, num_samples=val_samples)
                        debug_generate(task_sampler, memory_evaluator,
                                       memory_policy, n=3)
                    model.train()

                    f1 = val_scores.get("lb/qasper", 0.0)
                    print(f"  [eval step {global_step}] val F1: {f1:.2f}")
                    if wandb_run is not None:
                        for k, v in val_scores.items():
                            wandb_run.log({
                                f"lora/val_{k.replace('/', '_')}": v,
                                "lora/global_step": global_step,
                            })

                    if f1 > best_val_f1:
                        best_val_f1 = f1
                        save_lora_checkpoint(
                            model, optimizer, global_step, epoch,
                            os.path.join(cfg.lora.save_dir, "lora_best.pt"))

                # Periodic checkpoint
                if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                    save_lora_checkpoint(
                        model, optimizer, global_step, epoch,
                        os.path.join(cfg.lora.save_dir, "lora_latest.pt"))

            # Free memory
            if use_namm:
                empty_gpu_cache()

        avg_epoch_loss = epoch_loss / max(n_batches // grad_accum, 1)
        print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}")

    return global_step, best_val_f1


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_lora_checkpoint(model, optimizer, global_step, epoch, path):
    """Save LoRA adapter weights and optimizer state."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',
                exist_ok=True)
    lora_state = {}
    for n, p in model.model.named_parameters():
        if "lora_" in n:
            lora_state[n] = p.data.cpu()

    torch.save({
        'lora_state': lora_state,
        'optimizer_state': optimizer.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
    }, path)
    print(f"  LoRA checkpoint saved to {path}")


def load_lora_checkpoint(model, optimizer, path, device='cpu'):
    """Load LoRA adapter weights and optimizer state."""
    ckpt = torch.load(path, map_location=device)
    for name, param in model.model.named_parameters():
        if "lora_" in name and name in ckpt['lora_state']:
            param.data.copy_(ckpt['lora_state'][name])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f"LoRA checkpoint loaded from {path} "
          f"(step {ckpt.get('global_step', '?')})")
    return ckpt.get('global_step', 0), ckpt.get('epoch', 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='cfgs', config_name='config')
def main(cfg: DictConfig):
    _, global_rank, n_ddp = get_dist_info()
    master_process = global_rank <= 0

    if master_process:
        print(f"Working directory: {os.getcwd()}")
        print(OmegaConf.to_yaml(cfg.lora, resolve=True))

    stochasticity_setup(cfg=cfg, log_prefix='')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # ------------------------------------------------------------------
    # 1. Build model + evaluator
    # ------------------------------------------------------------------
    print("Building model...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, _) = make_eval_model(cfg=cfg)

    memory_model.to(cfg.device)

    # ------------------------------------------------------------------
    # 2. Optionally load NAMM checkpoint (for m4)
    # ------------------------------------------------------------------
    if cfg.lora.namm_ckpt is not None:
        load_namm_checkpoint(
            memory_model, memory_policy, cfg.lora.namm_ckpt, cfg.device)
        # Fix NAMM to use param index 0 for all evals
        batch_idxs = np.zeros([1])
        memory_policy.set_params_batch_idxs(batch_idxs)

    # ------------------------------------------------------------------
    # 3. Inject LoRA adapters (skip if eval-only without training)
    # ------------------------------------------------------------------
    lora_ckpt = cfg.lora.get('lora_ckpt', None)
    if cfg.lora.num_epochs > 0 or lora_ckpt:
        lora_params = inject_lora(memory_model, cfg)
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=cfg.lora.lr,
            weight_decay=cfg.lora.weight_decay,
        )
        if lora_ckpt:
            load_lora_checkpoint(memory_model, optimizer, lora_ckpt,
                                 device=cfg.device)
    else:
        lora_params = []
        optimizer = None
        print("Skipping LoRA injection (eval-only mode, no checkpoint)")

    # ------------------------------------------------------------------
    # 4. Prepare data (SFT pairs + task sampler for evaluation)
    # ------------------------------------------------------------------
    task_sampler = make_task_sampler(cfg=cfg)

    # Split overlapping tasks into train/test portions
    train_split = cfg.lora.get('train_split_ratio', None)
    if train_split is not None and train_split < 1.0:
        split_task_samples(task_sampler, train_ratio=train_split, seed=cfg.seed)

    # Wrap prompts in chat template for instruct models
    if cfg.lora.get('use_chat_template', False):
        wrap_prompts_chat_template(task_sampler, memory_evaluator.tokenizer)

    if cfg.lora.num_epochs > 0:
        sft_samples = build_sft_dataset(
            task_sampler, memory_evaluator.tokenizer, cfg)
    else:
        sft_samples = []

    # ------------------------------------------------------------------
    # 5. Wandb
    # ------------------------------------------------------------------
    wandb_run = None
    if cfg.wandb_log and master_process:
        wandb_run = wandb_init(cfg=cfg)

    # ------------------------------------------------------------------
    # 6. Initial evaluation (baseline before training)
    # ------------------------------------------------------------------
    if not cfg.lora.get('skip_baseline_eval', False):
        print("=== Baseline evaluation (before LoRA training) ===")
        memory_model.eval()
        with torch.no_grad():
            baseline_scores = evaluate_f1(
                task_sampler, memory_evaluator, memory_policy,
                train=False, num_samples=cfg.lora.val_samples)
        for k, v in baseline_scores.items():
            print(f"  {k}: {v:.2f}")
        if wandb_run is not None:
            for k, v in baseline_scores.items():
                wandb_run.log({f"lora/baseline_{k.replace('/', '_')}": v})
        print("=== Baseline debug generation (3 samples) ===")
        with torch.no_grad():
            debug_generate(task_sampler, memory_evaluator, memory_policy, n=3)
    else:
        print("=== Skipping baseline evaluation ===")

    # ------------------------------------------------------------------
    # 7. Train (skip if num_epochs == 0 — eval-only mode)
    # ------------------------------------------------------------------
    os.makedirs(cfg.lora.save_dir, exist_ok=True)
    mode = "m4 (NAMM active)" if cfg.lora.namm_active else "m1 (no NAMM)"
    print(f"=== Training mode: {mode} ===")

    if cfg.lora.num_epochs > 0:
        memory_model.train()
        t0 = time.time()
        global_step, best_val_f1 = train_sft(
            model=memory_model,
            sft_samples=sft_samples,
            lora_params=lora_params,
            optimizer=optimizer,
            cfg=cfg,
            task_sampler=task_sampler,
            memory_evaluator=memory_evaluator,
            memory_policy=memory_policy,
            wandb_run=wandb_run,
        )
        elapsed = time.time() - t0
        print(f"Training complete: {global_step} steps in {elapsed:.1f}s "
              f"({elapsed/60:.1f}min), best_val_F1={best_val_f1:.2f}")
    else:
        global_step = 0
        print("Skipping training (num_epochs=0)")

    # ------------------------------------------------------------------
    # 8. Debug generation (inspect actual outputs)
    # ------------------------------------------------------------------
    print("=== Debug generation (10 samples) ===")
    memory_model.eval()
    with torch.no_grad():
        debug_generate(task_sampler, memory_evaluator, memory_policy, n=10)

    # ------------------------------------------------------------------
    # 9. Final evaluation (full validation set)
    # ------------------------------------------------------------------
    print("=== Final evaluation (all validation samples) ===")
    with torch.no_grad():
        final_scores = evaluate_f1(
            task_sampler, memory_evaluator, memory_policy,
            train=False, num_samples=None)  # All samples
    for k, v in final_scores.items():
        print(f"  {k}: {v:.2f}")
    if wandb_run is not None:
        for k, v in final_scores.items():
            wandb_run.log({f"lora/final_{k.replace('/', '_')}": v})

    # ------------------------------------------------------------------
    # 9. Save final checkpoint
    # ------------------------------------------------------------------
    if optimizer is not None:
        save_lora_checkpoint(memory_model, optimizer, global_step,
                             cfg.lora.num_epochs,
                             os.path.join(cfg.lora.save_dir, "lora_final.pt"))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
