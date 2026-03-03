"""
Standalone LoRA finetuning script for the NAMM framework.

Supports two modes:
  m1/m3: Standard full-sequence LoRA training (no NAMM, no KV cache)
  m4:    LoRA training with NAMM active as a fixed eviction policy
         (truncated BPTT — gradients on final chunk only)

Usage:
  python lora_finetune.py --config-name=config +run=llama32_qasper_lora_m1
  python lora_finetune.py --config-name=config +run=llama32_qasper_lora_m4
"""

import os
import time
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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

    inject_adapter_in_model(memory_model.model, lora_config,
                            adapter_name="default")

    # Freeze everything, then unfreeze only LoRA params
    for param in memory_model.parameters():
        param.requires_grad = False
    lora_params = []
    for name, param in memory_model.model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)

    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in memory_model.parameters())
    print(f"LoRA: {n_lora:,} trainable params / {n_total:,} total "
          f"({100 * n_lora / n_total:.2f}%)")

    return lora_params


# ---------------------------------------------------------------------------
# NAMM checkpoint loading
# ---------------------------------------------------------------------------

def load_namm_checkpoint(memory_model, evolution_algorithm, ckpt_path, device):
    """Load a trained NAMM checkpoint and set the best params on the model."""
    print(f"Loading NAMM checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['evolution_state']

    # Handle buffer loading
    if memory_model.memory_policy_has_buffers_to_merge():
        buffers_prefix = 'stored_buffers_to_save.'
        buffers_dict = {}
        for k, v in list(state_dict.items()):
            if k.startswith(buffers_prefix):
                buffers_dict[k[len(buffers_prefix):]] = v
        evolution_algorithm.store_buffers(buffers=buffers_dict, best=False)

        best_buffers_prefix = 'best_stored_buffers_to_save.'
        best_buffers_dict = {}
        for k, v in list(state_dict.items()):
            if k.startswith(best_buffers_prefix):
                best_buffers_dict[k[len(best_buffers_prefix):]] = v
        if best_buffers_dict:
            evolution_algorithm.store_buffers(buffers=best_buffers_dict,
                                              best=True)
        else:
            evolution_algorithm.store_buffers(buffers=buffers_dict, best=True)

    evolution_algorithm.load_state_dict(state_dict)
    best_params = evolution_algorithm.best_params.unsqueeze(0)
    memory_model.set_memory_params(best_params)

    if memory_model.memory_policy_has_buffers_to_merge():
        best_buffers = evolution_algorithm.get_stored_buffers(best=True)
        if best_buffers:
            memory_model.load_buffers_dict(buffers_dict=best_buffers)

    print(f"NAMM checkpoint loaded (iter {checkpoint.get('iter_num', '?')})")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_dataloader(task_sampler, tokenizer, cfg):
    """Build a DataLoader from the task prompts."""
    prompts = []
    for task, prompt_list in task_sampler.lb_prompts_per_task.items():
        prompts.extend(prompt_list)

    if len(prompts) == 0:
        raise RuntimeError("No training prompts found — check task config")

    print(f"Tokenizing {len(prompts)} prompts "
          f"(max_seq_len={cfg.lora.max_seq_len})...")

    encodings = tokenizer(
        prompts,
        truncation=True,
        max_length=cfg.lora.max_seq_len,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # mask padding tokens in loss

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.lora.batch_size,
        shuffle=True,
        drop_last=False,
    )
    print(f"DataLoader: {len(dataset)} samples, "
          f"batch_size={cfg.lora.batch_size}, "
          f"{len(dataloader)} steps/epoch")
    return dataloader


# ---------------------------------------------------------------------------
# Training: m1/m3 mode (no NAMM)
# ---------------------------------------------------------------------------

def train_standard(model, dataloader, lora_params, optimizer, cfg, wandb=None):
    """Standard full-sequence LoRA training without NAMM."""
    model.train()
    grad_accum = cfg.lora.grad_accum_steps
    max_grad_norm = cfg.lora.max_grad_norm
    global_step = 0

    for epoch in range(cfg.lora.num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (ids, mask, labs) in enumerate(dataloader):
            ids = ids.to(cfg.device)
            mask = mask.to(cfg.device)
            labs = labs.to(cfg.device)

            with torch.enable_grad():
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                    labels=labs,
                    use_cache=False,
                    apply_memory_policy=False,
                )
                loss = outputs.loss / grad_accum

            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            batch_loss = loss.item() * grad_accum
            epoch_loss += batch_loss
            n_batches += 1

            if wandb is not None and (step + 1) % grad_accum == 0:
                wandb.log({
                    "lora/loss": batch_loss,
                    "lora/epoch": epoch,
                    "lora/global_step": global_step,
                    "lora/lr": optimizer.param_groups[0]['lr'],
                })

            if (step + 1) % 50 == 0:
                avg = epoch_loss / n_batches
                print(f"  epoch {epoch} step {step+1}/{len(dataloader)} "
                      f"loss={batch_loss:.4f} avg={avg:.4f}")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}")
        if wandb is not None:
            wandb.log({"lora/epoch_loss": avg_epoch_loss, "lora/epoch": epoch})

    return global_step


# ---------------------------------------------------------------------------
# Training: m4 mode (NAMM active, truncated BPTT)
# ---------------------------------------------------------------------------

def train_with_namm(model, dataloader, lora_params, optimizer, cfg,
                    wandb=None):
    """LoRA training with NAMM active as a frozen eviction policy.

    Uses truncated BPTT: processes most of the sequence without gradients
    (building the NAMM-compressed KV cache), then computes gradients on the
    final chunk(s) only.
    """
    model.train()
    grad_accum = cfg.lora.grad_accum_steps
    max_grad_norm = cfg.lora.max_grad_norm
    grad_chunks = cfg.lora.grad_chunks
    global_step = 0

    # Chunk size should match memory_policy_fixed_delay for proper eviction
    chunk_size = getattr(cfg, 'memory_policy_fixed_delay', 256) or 256

    print(f"m4 mode: chunk_size={chunk_size}, grad_chunks={grad_chunks}, "
          f"cache_size={getattr(cfg, 'cache_size', 'N/A')}")

    for epoch in range(cfg.lora.num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (ids, mask, labs) in enumerate(dataloader):
            ids = ids.to(cfg.device)
            labs = labs.to(cfg.device)
            seq_len = ids.shape[1]

            # Reset memory policy state for each new sample
            model.memory_policy.rotary_offset.zero_()

            # Split sequence into chunks
            chunk_starts = list(range(0, seq_len, chunk_size))
            n_chunks = len(chunk_starts)
            n_nograd = max(0, n_chunks - grad_chunks)

            past_kv = None
            loss = None

            # Phase 1: Context-building (no gradients)
            for i in range(n_nograd):
                start = chunk_starts[i]
                end = min(start + chunk_size, seq_len)
                chunk_ids = ids[:, start:end]

                with torch.no_grad():
                    out = model(
                        input_ids=chunk_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        apply_memory_policy=True,
                        labels=None,
                        limit_new_tokens=None,
                    )
                    past_kv = out.past_key_values

            # Phase 2: Gradient phase (final chunk(s))
            for i in range(n_nograd, n_chunks):
                start = chunk_starts[i]
                end = min(start + chunk_size, seq_len)
                chunk_ids = ids[:, start:end]
                chunk_labs = labs[:, start:end]

                with torch.enable_grad():
                    out = model(
                        input_ids=chunk_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        apply_memory_policy=True,
                        labels=chunk_labs,
                        limit_new_tokens=None,
                    )
                    past_kv = out.past_key_values
                    loss = out.loss / grad_accum

            if loss is not None:
                loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Detach KV cache to free memory
            del past_kv
            empty_gpu_cache()

            batch_loss = loss.item() * grad_accum if loss is not None else 0.0
            epoch_loss += batch_loss
            n_batches += 1

            if wandb is not None and (step + 1) % grad_accum == 0:
                wandb.log({
                    "lora/loss": batch_loss,
                    "lora/epoch": epoch,
                    "lora/global_step": global_step,
                })

            if (step + 1) % 20 == 0:
                avg = epoch_loss / n_batches
                print(f"  epoch {epoch} step {step+1}/{len(dataloader)} "
                      f"loss={batch_loss:.4f} avg={avg:.4f}")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}")
        if wandb is not None:
            wandb.log({"lora/epoch_loss": avg_epoch_loss, "lora/epoch": epoch})

    return global_step


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_lora_checkpoint(model, optimizer, global_step, epoch, path):
    """Save LoRA adapter weights and optimizer state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    print(f"LoRA checkpoint saved to {path}")


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

    # ------------------------------------------------------------------
    # 1. Build model
    # ------------------------------------------------------------------
    print("Building model...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, _) = make_eval_model(cfg=cfg)

    memory_model.to(cfg.device)
    tokenizer = memory_evaluator.tokenizer

    # ------------------------------------------------------------------
    # 2. Optionally load NAMM checkpoint (for m4)
    # ------------------------------------------------------------------
    if cfg.lora.namm_ckpt is not None:
        load_namm_checkpoint(
            memory_model, evolution_algorithm,
            cfg.lora.namm_ckpt, cfg.device)

    # ------------------------------------------------------------------
    # 3. Inject LoRA adapters
    # ------------------------------------------------------------------
    lora_params = inject_lora(memory_model, cfg)

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=cfg.lora.lr,
        weight_decay=cfg.lora.weight_decay,
    )

    # ------------------------------------------------------------------
    # 4. Prepare data
    # ------------------------------------------------------------------
    task_sampler = make_task_sampler(cfg=cfg)
    dataloader = prepare_dataloader(task_sampler, tokenizer, cfg)

    # ------------------------------------------------------------------
    # 5. Wandb
    # ------------------------------------------------------------------
    wandb = None
    if cfg.wandb_log and master_process:
        wandb = wandb_init(cfg=cfg)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    t0 = time.time()
    if cfg.lora.namm_active:
        print("=== Training mode: m4 (NAMM active, truncated BPTT) ===")
        global_step = train_with_namm(
            memory_model, dataloader, lora_params, optimizer, cfg,
            wandb=wandb)
    else:
        print("=== Training mode: m1/m3 (standard LoRA, no NAMM) ===")
        global_step = train_standard(
            memory_model, dataloader, lora_params, optimizer, cfg,
            wandb=wandb)

    elapsed = time.time() - t0
    print(f"Training complete: {global_step} steps in {elapsed:.1f}s "
          f"({elapsed/60:.1f}min)")

    # ------------------------------------------------------------------
    # 7. Save checkpoint
    # ------------------------------------------------------------------
    save_dir = cfg.lora.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lora_ckpt.pt")
    save_lora_checkpoint(memory_model, optimizer, global_step,
                         cfg.lora.num_epochs, save_path)

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
