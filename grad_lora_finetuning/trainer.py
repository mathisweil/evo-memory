"""
trainer.py — Gradient-based LoRA training class for NAMM interaction study.

This module is entirely separate from MemoryTrainer and carries no gradient-disabling
decorators anywhere, so the autograd graph survives from the NTP loss back through
the LLM to the LoRA A/B matrices.

Provides:
  - LoRATrainerConfig: dataclass with all training hyperparameters
  - LoRAGradTrainer: training class with optimizer, scheduler, gradient accumulation,
    checkpoint I/O (including AdamW/scheduler state), NAMM-active mode, and artifact
    contract for Phase 4+ evaluation.

Usage:
    from grad_lora_finetuning.trainer import LoRAGradTrainer, LoRATrainerConfig
    from namm.trainer import WandbConfig

    cfg = LoRATrainerConfig(
        out_dir='exp_local/lora_m1',
        method='lora_grad',
        seed=1337,
        max_seq_len=3500,
        task_names=['qasper', 'narrativeqa', 'passage_retrieval_en'],
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        namm_active=False,
        eval_interval=100,
        log_interval=10,
        always_save_checkpoint=True,
        init_from=None,
        dtype='bfloat16',
    )
    wcfg = WandbConfig(wandb_log=True, wandb_project='...', wandb_run_name='...', wandb_group_name='...')
    trainer = LoRAGradTrainer(model, tokenizer, evaluation_model, task_sampler, memory_policy, cfg, wcfg, device='cuda')
    trainer.train()
"""

import csv
import dataclasses
import os
import shutil
import time
import warnings
import yaml
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from grad_lora_finetuning.datasets import NTPDataset, SFTDataset, pad_collate_fn
from namm.trainer import WandbConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoRATrainerConfig:
    """All hyperparameters for LoRAGradTrainer.

    Locked values (do not override without explicit reason):
      - max_seq_len=3500 (leaves headroom in 4096-token RoPE window)
      - num_epochs=3
      - batch_size=1 (VRAM budget on 4070Ti)
      - gradient_accumulation_steps=16 (effective batch = 16)
      - learning_rate=2e-4
      - max_grad_norm=1.0
      - warmup_ratio=0.03
    """
    out_dir: str
    method: str                     # 'lora_grad' — used in artifact dir: results/{method}/{seed}/
    seed: int                       # 1337 default — used in artifact dir

    max_seq_len: int                # 3500 (locked)
    task_names: List[str]           # e.g. ['qasper', 'narrativeqa', 'passage_retrieval_en']

    num_epochs: int                 # 3 (locked)
    batch_size: int                 # 1 (locked)
    gradient_accumulation_steps: int  # 16 (locked)
    learning_rate: float            # 2e-4 (locked)
    weight_decay: float             # 0.01
    max_grad_norm: float            # 1.0 (locked)
    warmup_ratio: float             # 0.03 (locked)

    namm_active: bool               # False for m1; True for m4-frozen
    eval_interval: int              # save checkpoint every N gradient updates
    log_interval: int               # wandb log every N gradient updates
    always_save_checkpoint: bool
    init_from: Optional[str]        # path to ckpt (NAMM ckpt for m4, LoRA ckpt for resume)
    dtype: str                      # 'bfloat16'
    sft_mode: bool = False          # True -> SFTDataset; False -> NTPDataset
    train_frac: float = 0.7         # fraction of each task's examples used for training
    val_frac: float = 0.15          # fraction used for validation (best checkpoint selection)
    max_conditioning_length: int = 6500  # max prompt tokens for evaluation (skip longer prompts)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LoRAGradTrainer:
    """Gradient-based LoRA training class.

    CRITICAL DESIGN CONSTRAINTS (do not break):
      1. No gradient-disabling decorators on ANY method — the autograd graph must survive.
      2. No AMP / GradScaler — silently downcasts float32 LoRA weights.
      3. No gradient checkpointing — causes wrong gradients with PEFT (HF #23170).
      4. Never call model.merge_adapter() — corrupts base weights.
      5. AdamW is built over LoRA params only (requires_grad=True).
      6. NAMM KV-cache reset between documents uses memory_policy.initialize_buffers()
         and passing past_key_values=None to the next forward call.
    """

    def __init__(
        self,
        model,
        tokenizer,
        evaluation_model,
        task_sampler=None,
        memory_policy=None,
        trainer_config: LoRATrainerConfig = None,
        wandb_config: WandbConfig = None,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluation_model = evaluation_model
        self.task_sampler = task_sampler
        self.memory_policy = memory_policy
        self.trainer_config = trainer_config
        self.wandb_config = wandb_config
        self.device = device
        self.artifact_dir = None
        self._metrics_csv_path = None
        self._wandb_id_path = None

        cfg = self.trainer_config

        # --- Identify LoRA params (PEFT sets requires_grad=True on A/B matrices only) ---
        self.lora_params = [p for p in model.parameters() if p.requires_grad]
        assert len(self.lora_params) > 0, (
            "No requires_grad=True parameters found. "
            "Call model.apply_lora_adapters() before constructing LoRAGradTrainer."
        )
        n_trainable = sum(p.numel() for p in self.lora_params)
        n_total = sum(p.numel() for p in model.parameters())
        n_frozen = n_total - n_trainable
        print(f"LoRAGradTrainer: {len(self.lora_params)} LoRA parameter tensors identified.")
        print(f"  Trainable: {n_trainable:,} ({100*n_trainable/n_total:.2f}%) | "
              f"Frozen: {n_frozen:,} | Total: {n_total:,}")

        # --- PEFT gradient fix (TRAIN-03) ---
        def make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)

        try:
            embed_layer = model.model.get_input_embeddings()
        except AttributeError:
            embed_layer = model.get_input_embeddings()

        embed_layer.register_forward_hook(make_inputs_require_grad)
        print("LoRAGradTrainer: PEFT embedding forward hook registered.")

        # --- ANLYS-01: per-layer retention dict (populated in _train_step) ---
        self._last_retention_dict = {}

        # --- Build dataset and DataLoader ---
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token_id is None:
            finetune_pad_id = tokenizer.convert_tokens_to_ids('<|finetune_right_pad_id|>')
            if finetune_pad_id != tokenizer.unk_token_id and finetune_pad_id is not None:
                tokenizer.pad_token_id = finetune_pad_id
                tokenizer.pad_token = '<|finetune_right_pad_id|>'
                print("LoRAGradTrainer: pad_token_id set to <|finetune_right_pad_id|> (128004)")
            else:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.eos_token
                print("LoRAGradTrainer: pad_token_id set to EOS (finetune_right_pad_id not found)")

        if cfg.sft_mode:
            dataset = SFTDataset(
                task_names=cfg.task_names,
                tokenizer=tokenizer,
                max_seq_len=cfg.max_seq_len,
                max_conditioning_length=cfg.max_conditioning_length,
                seed=cfg.seed,
                train_frac=cfg.train_frac,
            )
            print("LoRAGradTrainer: SFT mode — using SFTDataset with answer-only loss masking.")
        else:
            dataset = NTPDataset(
                task_names=cfg.task_names,
                tokenizer=tokenizer,
                max_seq_len=cfg.max_seq_len,
                seed=cfg.seed,
            )
        collate_fn = partial(
            pad_collate_fn,
            pad_token_id=tokenizer.pad_token_id,
            max_seq_len=cfg.max_seq_len,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True,
        )
        print(f"LoRAGradTrainer: dataset has {len(dataset)} samples, "
              f"{len(self.dataloader)} batches/epoch.")

        # --- Print train/test split summary ---
        self._print_split_summary()

        # --- Build AdamW over LoRA params only ---
        self.optimizer = torch.optim.AdamW(
            self.lora_params,
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=cfg.weight_decay,
        )

        # --- Compute total gradient update steps and build cosine LR scheduler ---
        total_steps = (
            cfg.num_epochs * len(self.dataloader) // cfg.gradient_accumulation_steps
        )
        warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"LoRAGradTrainer: total gradient steps={total_steps}, "
              f"warmup={warmup_steps} steps, lr={cfg.learning_rate}.")

        # --- Optionally resume from checkpoint ---
        self._resume_step = 0
        if cfg.init_from is not None:
            self._resume_step = self._load_ckpt(cfg.init_from)

    # -----------------------------------------------------------------------
    # Dataset summary
    # -----------------------------------------------------------------------

    def _print_split_summary(self):
        """Print train/val/test split summary using task_sampler's pre-computed split."""
        cfg = self.trainer_config
        tokenizer = self.tokenizer

        print(f"\n{'='*60}")
        print(f"DATASET SPLIT SUMMARY (filter <={cfg.max_conditioning_length} tok, then split)")
        print(f"{'='*60}")

        if self.task_sampler is not None and self.task_sampler._train_idxs_per_task is not None:
            total_train = 0
            total_val = 0
            total_test = 0
            for task_name in self.task_sampler.lb_tasks:
                train_idxs = self.task_sampler._train_idxs_per_task.get(task_name, [])
                val_idxs = (self.task_sampler._val_idxs_per_task or {}).get(task_name, [])
                test_idxs = self.task_sampler._test_idxs_per_task.get(task_name, [])
                n_total = self.task_sampler.num_prompts_per_lb_task[task_name]
                n_train, n_val, n_test = len(train_idxs), len(val_idxs), len(test_idxs)
                total_train += n_train
                total_val += n_val
                total_test += n_test

                prompts = self.task_sampler.lb_prompts_per_task[task_name]
                jsons = self.task_sampler.lb_jsons_per_task[task_name]

                def _prompt_lens(idxs):
                    return [len(tokenizer.encode(prompts[i], add_special_tokens=False)) for i in idxs] if len(idxs) else [0]

                def _answer_lens(idxs):
                    lens = []
                    for i in idxs:
                        ans = jsons[i].get('answers', [''])[0] if jsons[i].get('answers') else ''
                        lens.append(len(tokenizer.encode(ans, add_special_tokens=False)))
                    return lens if lens else [0]

                train_lens = _prompt_lens(train_idxs)
                val_lens = _prompt_lens(val_idxs)
                test_lens = _prompt_lens(test_idxs)

                n_filtered = n_total - n_train - n_val - n_test
                print(f"\n  {task_name}: {n_total} total, {n_filtered} filtered -> "
                      f"{n_train} train / {n_val} val / {n_test} test")
                print(f"    Train prompts: min={min(train_lens)}, avg={sum(train_lens)/len(train_lens):.0f}, max={max(train_lens)} tokens")
                if val_lens and val_lens != [0]:
                    print(f"    Val   prompts: min={min(val_lens)}, avg={sum(val_lens)/len(val_lens):.0f}, max={max(val_lens)} tokens")
                print(f"    Test  prompts: min={min(test_lens)}, avg={sum(test_lens)/len(test_lens):.0f}, max={max(test_lens)} tokens")

            print(f"\n  TOTAL: {total_train} train / {total_val} val / {total_test} test prompts across {len(self.task_sampler.lb_tasks)} tasks")
            print(f"  Training samples (after max_seq_len={cfg.max_seq_len} filter): {len(self.dataloader.dataset)}")

        steps_per_epoch = len(self.dataloader) // cfg.gradient_accumulation_steps
        total_steps = cfg.num_epochs * steps_per_epoch
        print(f"\n  Training: {cfg.num_epochs} epochs x {steps_per_epoch} steps/epoch = {total_steps} optimizer steps")
        print(f"  Eval every {cfg.eval_interval} steps ({cfg.eval_interval / max(steps_per_epoch,1):.1f} epochs)")
        print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _freeze_base_weights(self):
        """Ensure all non-LoRA parameters have requires_grad=False.

        Called after model.train() (which re-enables grad for all params) to
        restore the LoRA-only training constraint.
        """
        for n, p in self.model.model.named_parameters():
            if 'lora_' not in n:
                p.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Training step helpers
    # -----------------------------------------------------------------------

    def _train_step(self, batch, past_key_values=None):
        """Run one forward + backward pass.

        Args:
            batch: dict with 'input_ids' and 'labels' (from pad_collate_fn)
            past_key_values: None for NAMM-active mode (reset between docs)

        Returns:
            (unscaled_loss_float, past_key_values_out)
        """
        cfg = self.trainer_config
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # NAMM-active: reset KV cache and memory policy buffers before each doc
        if cfg.namm_active:
            if hasattr(self.model, 'memory_policy'):
                if hasattr(self.model.memory_policy, 'initialize_buffers'):
                    self.model.memory_policy.initialize_buffers()
                elif hasattr(self.model.memory_policy, 'reset_kv_cache'):
                    self.model.memory_policy.reset_kv_cache()
                elif hasattr(self.model.memory_policy, 'reset'):
                    self.model.memory_policy.reset()
                else:
                    print(
                        "WARNING: memory_policy has no known reset method "
                        "(initialize_buffers / reset_kv_cache / reset). "
                        "Proceeding without KV-cache reset."
                    )
            past_key_values = None

        if cfg.namm_active:
            # --- Two-phase forward for memory-efficient NAMM training ---
            answer_mask = (labels[0] != -100)
            if answer_mask.any():
                answer_start = answer_mask.nonzero(as_tuple=True)[0][0].item()
            else:
                answer_start = labels.shape[1]

            chunk_align = self.model.max_new_tokens  # 64
            context_end = (answer_start // chunk_align) * chunk_align
            context_end = max(context_end, 0)

            seq_len = input_ids.shape[1]

            # Phase 1: context tokens under no_grad
            if context_end > 0:
                with torch.no_grad():
                    ctx_outputs = self.model(
                        input_ids=input_ids[:, :context_end],
                        use_cache=True,
                        apply_memory_policy=True,
                        limit_new_tokens=None,
                        output_hidden_states=False,
                        skip_lm_head=True,
                    )
                past_key_values = ctx_outputs.past_key_values
                del ctx_outputs
                torch.cuda.empty_cache()

            # Phase 2: answer tokens with gradients
            phase2_input = input_ids[:, context_end:]
            phase2_pos = torch.arange(
                context_end, seq_len, device=self.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

            outputs = self.model(
                input_ids=phase2_input,
                position_ids=phase2_pos,
                past_key_values=past_key_values,
                use_cache=True,
                apply_memory_policy=True,
                limit_new_tokens=None,
                output_hidden_states=True,
                skip_lm_head=True,
            )

            hidden_states = outputs.hidden_states[-1]
            phase2_labels = labels[:, context_end:]
            shift_hidden = hidden_states[:, :-1, :].contiguous()
            shift_labels = phase2_labels[:, 1:].contiguous()
        else:
            # Single forward pass for non-NAMM mode.
            # Temporarily disable memory_policy_fixed_delay to prevent the
            # model from splitting the input into chunks (not needed without
            # NAMM eviction, and split processing requires use_cache=True
            # which conflicts with gradient checkpointing).
            saved_delay = getattr(self.model, 'memory_policy_fixed_delay', None)
            self.model.memory_policy_fixed_delay = None
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    apply_memory_policy=False,
                    limit_new_tokens=None,
                    output_hidden_states=True,
                    skip_lm_head=True,
                )
            finally:
                self.model.memory_policy_fixed_delay = saved_delay
            hidden_states = outputs.hidden_states[-1]
            shift_hidden = hidden_states[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

        # Chunked cross-entropy
        lm_head = self.model.lm_head

        chunk_size = 512
        seq_len = shift_hidden.shape[1]
        total_loss = torch.tensor(0.0, device=self.device)
        n_tokens = (shift_labels != -100).sum()

        for i in range(0, seq_len, chunk_size):
            chunk_h = shift_hidden[:, i:i+chunk_size, :]
            chunk_logits = lm_head(chunk_h).float()
            chunk_labels = shift_labels[:, i:i+chunk_size].contiguous().view(-1)
            total_loss = total_loss + F.cross_entropy(
                chunk_logits.view(-1, chunk_logits.size(-1)),
                chunk_labels,
                ignore_index=-100,
                reduction='sum',
            )
            del chunk_logits

        loss = total_loss / n_tokens.clamp(min=1)
        loss = loss / cfg.gradient_accumulation_steps
        loss.backward()

        # ANLYS-01: per-layer token retention logging
        self._last_retention_dict = {}
        if cfg.namm_active and hasattr(outputs, 'past_key_values') \
                and outputs.past_key_values is not None:
            n_input = input_ids.shape[1]
            if n_input > 0:
                for layer_i, layer_kv in enumerate(outputs.past_key_values):
                    if layer_kv is not None and layer_kv[0] is not None:
                        n_retained = layer_kv[0].shape[-2]
                        self._last_retention_dict[f'retention/layer_{layer_i}'] = (
                            n_retained / n_input
                        )
                if self._last_retention_dict and all(
                        v >= 1.0 for v in self._last_retention_dict.values()):
                    warnings.warn(
                        "ANLYS-01 WARNING: all retention/layer_i values are >= 1.0. "
                        "NAMM may not be evicting tokens.",
                        stacklevel=2,
                    )

        return loss.item() * cfg.gradient_accumulation_steps, getattr(outputs, 'past_key_values', None)

    def _optimizer_step(self):
        """Clip gradients, step optimizer + scheduler, zero grads."""
        cfg = self.trainer_config
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.lora_params, cfg.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm.item()

    # -----------------------------------------------------------------------
    # Checkpoint I/O
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, step_num: int, val_score: float = None, is_final: bool = False) -> str:
        """Saves the latest checkpoint, tracks 'best' model, and handles WandB artifacts."""
        ckpt_dir = self.artifact_dir if self.artifact_dir is not None else self.trainer_config.out_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')

        # 1. Build and save the state dictionary once
        checkpoint = {
            'lora_state_dict': {
                n: p.data.clone()
                for n, p in self.model.model.named_parameters()
                if p.requires_grad
            },
            'lora_config': {
                'rank': self.model._lora_rank,
                'target_modules': list(self.model._lora_target_modules),
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step_num': step_num,
            'trainer_type': 'lora_grad',
        }

        is_best = False
        if val_score is not None:
            if not hasattr(self, '_best_val_score'):
                self._best_val_score = -1.0
                self._best_step = 0

            if val_score > self._best_val_score:
                self._best_val_score = val_score
                self._best_step = step_num
                is_best = True
            else:
                print(f"  Val F1={val_score:.2f} (best={self._best_val_score:.2f} at step {self._best_step})")

        # Always persist best tracking state so any resumed checkpoint can restore it.
        if hasattr(self, '_best_val_score'):
            checkpoint['best_val_score'] = self._best_val_score
            checkpoint['best_step'] = self._best_step

        torch.save(checkpoint, ckpt_path)
        print(f"Saved latest checkpoint to {ckpt_path} (step {step_num})")

        # 2. Fast file copy for new best models
        if is_best:
            best_path = os.path.join(ckpt_dir, 'best_ckpt.pt')
            shutil.copy2(ckpt_path, best_path)
            print(f"  *** New best val F1={val_score:.2f} at step {step_num} -> copied to {best_path}")

            if self.wandb_config and self.wandb_config.wandb_log and wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"run-{wandb.run.id}-best-ckpt",
                    type="model",
                    metadata={"step": step_num, "val_score": val_score}
                )
                artifact.add_file(best_path)
                config_path = os.path.join(ckpt_dir, 'config.yaml')
                if os.path.exists(config_path):
                    artifact.add_file(config_path)
                wandb.log_artifact(artifact, aliases=["best", f"step-{step_num}"])

        # 3. Upload entire artifact directory at the end of training
        if is_final and self.wandb_config and self.wandb_config.wandb_log and wandb.run is not None:
            run_artifact = wandb.Artifact(
                name=f"run-{wandb.run.id}-final-results",
                type="model",
                description="Contains final weights, best weights, config, and metrics."
            )
            run_artifact.add_dir(ckpt_dir)
            wandb.log_artifact(run_artifact, aliases=["final"])
            print(f"Uploaded final wandb artifact directory: {run_artifact.name}")

        return ckpt_path

    def _load_ckpt(self, load_path: str) -> int:
        """Load checkpoint — handles both LoRA checkpoints and NAMM checkpoints."""
        print(f"Loading checkpoint from {load_path}")
        ckpt = torch.load(load_path, map_location=self.device, weights_only=False)

        # --- NAMM evolution state (m4: frozen NAMM policy) ---
        if 'evolution_state' in ckpt:
            evo_state = ckpt['evolution_state']
            if 'best_member' in evo_state:
                best_params = evo_state['best_member']
                if best_params.dim() == 1:
                    best_params = best_params.unsqueeze(0)
                self.model.set_memory_params(best_params.to(self.device))
                self.model.set_memory_params_batch_idxs(
                    np.zeros([1], dtype=np.int64))
                print(f"Loaded NAMM best_member ({best_params.shape[-1]} params) "
                      f"from evolution_state")
            else:
                print("WARNING: evolution_state has no best_member key")

            buffers_prefix = 'stored_buffers_to_save.'
            buffers_dict = {
                k[len(buffers_prefix):]: v.to(self.device)
                for k, v in evo_state.items()
                if k.startswith(buffers_prefix)
            }
            if buffers_dict:
                self.model.load_buffers_dict(buffers_dict=buffers_dict)
                print(f"Loaded {len(buffers_dict)} NAMM normalization buffers")

        # --- LoRA adapter weights ---
        if 'lora_state_dict' in ckpt:
            loaded = 0
            for n, p in self.model.model.named_parameters():
                if p.requires_grad and n in ckpt['lora_state_dict']:
                    p.data.copy_(ckpt['lora_state_dict'][n])
                    loaded += 1
            print(f"Loaded LoRA state ({loaded} tensors) from {load_path}")

        # --- Optimizer / scheduler (exact resume) ---
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("Restored optimizer + scheduler state (TRAIN-05)")

        # --- Best Score Tracking Resume ---
        step_num = ckpt.get('step_num', 0)
        if 'best_val_score' in ckpt:
            self._best_val_score = ckpt['best_val_score']
            self._best_step = ckpt.get('best_step', step_num)
            print(
                f"Restored best validation score tracking: "
                f"F1={self._best_val_score:.2f} at step {self._best_step}"
            )
        elif 'val_score' in ckpt:
            # Backward compat: old checkpoints only wrote val_score when is_best.
            self._best_val_score = ckpt['val_score']
            self._best_step = step_num
            print(
                f"Restored best validation score tracking (legacy key): "
                f"F1={self._best_val_score:.2f} at step {self._best_step}"
            )

        return step_num

    # -----------------------------------------------------------------------
    # Artifact contract (ARTIFACT-01)
    # -----------------------------------------------------------------------

    def _write_artifact_contract(self, cfg_yaml: str, method: str, seed: int):
        """Create results/{method}/{seed}/ and initialise all artifact files."""
        self.artifact_dir = os.path.join('results', method, str(seed))
        os.makedirs(self.artifact_dir, exist_ok=True)

        with open(os.path.join(self.artifact_dir, 'config.yaml'), 'w') as f:
            f.write(cfg_yaml)

        self._wandb_id_path = os.path.join(self.artifact_dir, 'wandb_run_id.txt')

        self._metrics_csv_path = os.path.join(self.artifact_dir, 'metrics.csv')
        with open(self._metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['step', 'loss', 'grad_norm', 'lr']
            )
            writer.writeheader()

        self._val_csv_path = os.path.join(self.artifact_dir, 'val_metrics.csv')
        self._val_csv_initialized = False

        print(f"Artifact contract initialized: {self.artifact_dir}")

    def _append_metrics(
        self, step: int, loss: float, grad_norm: float, lr: float
    ):
        """Append one row to metrics.csv."""
        with open(self._metrics_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['step', 'loss', 'grad_norm', 'lr']
            )
            writer.writerow(
                {'step': step, 'loss': loss, 'grad_norm': grad_norm, 'lr': lr}
            )

    def _write_wandb_id(self):
        """Write wandb run ID to wandb_run_id.txt."""
        if wandb.run is not None and self._wandb_id_path is not None:
            with open(self._wandb_id_path, 'w') as f:
                f.write(wandb.run.id)

    # -----------------------------------------------------------------------
    # F1 evaluation
    # -----------------------------------------------------------------------

    def _set_split_indices(self, split='val', num_samples=None):
        """Pre-populate task_sampler with indices for a given split."""
        cfg = self.trainer_config
        all_split_idxs = self.task_sampler.get_split_indices(split)
        split_idxs = {}
        for task_n, indices in all_split_idxs.items():
            indices = np.array(indices)
            if num_samples is not None and num_samples < len(indices):
                rng = np.random.RandomState(cfg.seed)
                indices = rng.choice(indices, size=num_samples, replace=False)
            split_idxs[task_n] = indices
        self.task_sampler.latest_sampled_idxs_per_lb_task = split_idxs
        self.task_sampler.latest_lb_tasks_names = list(split_idxs.keys())

    def _evaluate_f1(self, split='val', num_samples=None):
        """Run generation-based F1 evaluation on the specified split."""
        if self.task_sampler is None or self.evaluation_model is None:
            return {}

        if self.memory_policy is not None:
            batch_idxs = np.zeros([1])
            self.memory_policy.set_params_batch_idxs(batch_idxs)

        self._set_split_indices(split=split, num_samples=num_samples)

        score_dicts = self.task_sampler.evaluate(
            lm=self.evaluation_model,
            train=False,
            evolved_model=False,
            pop_reps=1,
            resample_requests=False,
        )
        scores = score_dicts[0] if score_dicts else {}

        all_split_idxs = self.task_sampler.get_split_indices(split)
        weighted_sum = 0.0
        total_samples = 0
        for k, v in scores.items():
            if k.startswith('lb/'):
                n_samples = len(all_split_idxs.get(k[len('lb/'):], []))
                weighted_sum += v * max(n_samples, 1)
                total_samples += max(n_samples, 1)
        if total_samples > 0:
            scores['lb/avg_f1'] = weighted_sum / total_samples

        return scores

    def _debug_generate(self, n=3, split='val'):
        """Generate on n samples from the specified split and print results."""
        if self.task_sampler is None or self.evaluation_model is None:
            return

        cfg = self.trainer_config
        tokenizer = self.tokenizer
        if self.memory_policy is not None:
            batch_idxs = np.zeros([1])
            self.memory_policy.set_params_batch_idxs(batch_idxs)

        for task_name in self.task_sampler.lb_test_tasks:
            prompts = self.task_sampler.lb_prompts_per_task[task_name]
            jsons = self.task_sampler.lb_jsons_per_task[task_name]

            all_split_idxs = self.task_sampler.get_split_indices(split)
            split_idxs = list(all_split_idxs.get(task_name, []))
            sample_idxs = split_idxs[:min(n, len(split_idxs))]
            sample_prompts = [prompts[i] for i in sample_idxs]
            sample_jsons = [jsons[i] for i in sample_idxs]

            print(f"\n{'='*60}")
            print(f"DEBUG GENERATION: {task_name} ({len(sample_prompts)} {split} samples)")
            print(f"{'='*60}")
            for i, prompt in enumerate(sample_prompts):
                n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                print(f"  Sample {i} (idx {sample_idxs[i]}): {n_tokens} tokens")

            outputs = self.evaluation_model.evaluate_lb(
                dataset_samples=sample_prompts,
                disable_tqdm=True,
                pop_reps=1,
                max_gen_tokens=64,
            )

            for i, (prompt, json_obj, pred) in enumerate(
                    zip(sample_prompts, sample_jsons, outputs)):
                answers = json_obj.get("answers", [])
                prompt_tail = prompt[-300:]
                print(f"\n--- Sample {i} (idx {sample_idxs[i]}) ---")
                print(f"PROMPT (last 300 chars): ...{prompt_tail}")
                print(f"GROUND TRUTH: {answers}")
                print(f"PREDICTION:   '{pred}'")

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self):
        """Run full gradient-based LoRA training."""
        cfg = self.trainer_config
        wcfg = self.wandb_config

        # --- Artifact contract ---
        cfg_dict = dataclasses.asdict(cfg)
        cfg_dict['lora_rank'] = self.model._lora_rank
        cfg_dict['lora_target_modules'] = list(self.model._lora_target_modules)
        try:
            cfg_yaml = yaml.dump(cfg_dict, default_flow_style=False)
        except Exception:
            import json
            cfg_yaml = json.dumps(cfg_dict, indent=2)
        self._write_artifact_contract(cfg_yaml, method=cfg.method, seed=cfg.seed)

        # --- wandb init ---
        if wcfg.wandb_log:
            wandb.init(
                project=wcfg.wandb_project,
                entity=wcfg.wandb_entity,
                group=wcfg.wandb_group_name,
                name=wcfg.wandb_run_name,
                config=cfg_dict,
            )
            self._write_wandb_id()

        # --- Baseline evaluation (before training) on val set ---
        if self.task_sampler is not None:
            print("=== Baseline evaluation on VAL set (before LoRA training) ===")
            self.model.eval()
            torch.cuda.empty_cache()
            try:
                with torch.no_grad():
                    baseline_scores = self._evaluate_f1(split='val')
                for k, v in baseline_scores.items():
                    print(f"  {k}: {v:.2f}")
                if wcfg.wandb_log and wandb.run is not None:
                    baseline_dict = {f"lora/baseline_{k.replace('/', '_')}": v
                                     for k, v in baseline_scores.items()}
                    wandb.log(baseline_dict, step=0)
                print("=== Baseline debug generation (3 val samples) ===")
                with torch.no_grad():
                    self._debug_generate(n=3, split='val')
            except (torch.cuda.OutOfMemoryError, ValueError) as e:
                print(f"WARNING: Baseline eval OOM — skipping. ({e})")
                torch.cuda.empty_cache()

        # --- Set model to training mode ---
        self.model.train()
        self._freeze_base_weights()

        # --- Training loop ---
        global_step = self._resume_step
        epoch_loss = 0.0
        epoch_steps = 0
        accum_loss = 0.0
        accum_batches = 0
        t_start = time.time()

        # Calculate resume epoch and batches to skip
        steps_per_epoch = len(self.dataloader) // cfg.gradient_accumulation_steps
        start_epoch = global_step // max(1, steps_per_epoch)
        skip_batches = (global_step % max(1, steps_per_epoch)) * cfg.gradient_accumulation_steps

        for epoch in range(start_epoch, cfg.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{cfg.num_epochs} ===")
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(self.dataloader):
                # Fast-forward dataloader to the exact resume batch
                if epoch == start_epoch and batch_idx < skip_batches:
                    continue

                step_loss, _ = self._train_step(batch)
                accum_loss += step_loss
                accum_batches += 1

                if accum_batches == cfg.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    global_step += 1
                    avg_loss = accum_loss / cfg.gradient_accumulation_steps
                    current_lr = self.scheduler.get_last_lr()[0]
                    epoch_loss += avg_loss
                    epoch_steps += 1

                    if wcfg.wandb_log and wandb.run is not None:
                        wandb.log({
                            'lora/loss': avg_loss,
                            'lora/epoch': epoch,
                            'lora/global_step': global_step,
                            'lora/grad_norm': grad_norm,
                            'lora/lr': current_lr,
                        }, step=global_step)
                        if self._last_retention_dict:
                            wandb.log(self._last_retention_dict, step=global_step)

                    if global_step % cfg.log_interval == 0:
                        elapsed = time.time() - t_start
                        print(
                            f"  epoch {epoch} step {global_step} "
                            f"loss={avg_loss:.4f} | "
                            f"grad_norm {grad_norm:.4f} | lr {current_lr:.2e} | "
                            f"elapsed {elapsed:.1f}s"
                        )

                    if self._metrics_csv_path is not None:
                        self._append_metrics(
                            step=global_step,
                            loss=avg_loss,
                            grad_norm=grad_norm,
                            lr=current_lr,
                        )

                    if global_step % cfg.eval_interval == 0:
                        avg_f1 = None

                        if self.task_sampler is not None:
                            self.model.eval()
                            torch.cuda.empty_cache()
                            try:
                                with torch.no_grad():
                                    val_scores = self._evaluate_f1(split='val')
                                    # Evaluate on a train subsample (same size as val) for overfitting detection
                                    val_n = sum(len(v) for v in self.task_sampler.get_split_indices('val').values())
                                    train_scores = self._evaluate_f1(split='train', num_samples=val_n)
                                    self._debug_generate(n=3, split='val')
                            except (torch.cuda.OutOfMemoryError, ValueError) as e:
                                print(f"WARNING: Periodic eval OOM at step {global_step} — skipping. ({e})")
                                torch.cuda.empty_cache()
                                val_scores = {}
                                train_scores = {}
                            self.model.train()
                            self._freeze_base_weights()

                            if train_scores:
                                score_parts = [f"{k}: {v:.2f}" for k, v in sorted(train_scores.items())]
                                print(f"  [train step {global_step}] {' | '.join(score_parts)}")
                                if wcfg.wandb_log and wandb.run is not None:
                                    train_dict = {f"lora/train_{k.replace('/', '_')}": v
                                                  for k, v in train_scores.items()}
                                    wandb.log(train_dict, step=global_step)

                            if val_scores:
                                score_parts = [f"{k}: {v:.2f}" for k, v in sorted(val_scores.items())]
                                print(f"  [val step {global_step}] {' | '.join(score_parts)}")
                                # Log eval scores to CSV
                                if self._val_csv_path is not None:
                                    row = {'step': global_step}
                                    row.update({f"train_{k.replace('/', '_')}": round(v, 4)
                                                for k, v in sorted(train_scores.items())})
                                    row.update({f"val_{k.replace('/', '_')}": round(v, 4)
                                                for k, v in sorted(val_scores.items())})
                                    if not self._val_csv_initialized:
                                        with open(self._val_csv_path, 'w', newline='') as f:
                                            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                                            writer.writeheader()
                                            writer.writerow(row)
                                        self._val_csv_initialized = True
                                    else:
                                        with open(self._val_csv_path, 'a', newline='') as f:
                                            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                                            writer.writerow(row)
                                if wcfg.wandb_log and wandb.run is not None:
                                    val_dict = {f"lora/val_{k.replace('/', '_')}": v
                                                for k, v in val_scores.items()}
                                    wandb.log(val_dict, step=global_step)

                                avg_f1 = val_scores.get('lb/avg_f1', 0.0)

                        # Single call handles saving the latest checkpoint and copying to 'best' if applicable
                        if cfg.always_save_checkpoint or avg_f1 is not None:
                            self._save_checkpoint(global_step, val_score=avg_f1)

                    accum_loss = 0.0
                    accum_batches = 0

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}")

        # --- Final checkpoint ---
        print(f"\nTraining complete. Total gradient steps: {global_step}")
        self._save_checkpoint(global_step, is_final=True)

        # --- Load best checkpoint for final test evaluation ---
        if hasattr(self, '_best_val_score') and self._best_val_score > 0:
            best_dir = self.artifact_dir if self.artifact_dir else self.trainer_config.out_dir
            best_path = os.path.join(best_dir, 'best_ckpt.pt')
            if os.path.exists(best_path):
                print(f"\n=== Loading best checkpoint (step {self._best_step}, val F1={self._best_val_score:.2f}) for test evaluation ===")
                self._load_ckpt(best_path)

        # --- Final evaluation on TEST set ---
        if self.task_sampler is not None:
            print("=== Debug generation (10 TEST samples) ===")
            self.model.eval()
            with torch.no_grad():
                self._debug_generate(n=10, split='test')

            print("=== Final evaluation on TEST set (all samples) ===")
            with torch.no_grad():
                final_scores = self._evaluate_f1(split='test', num_samples=None)
            for k, v in final_scores.items():
                print(f"  {k}: {v:.2f}")
            if wcfg.wandb_log and wandb.run is not None:
                final_dict = {f"lora/test_{k.replace('/', '_')}": v
                              for k, v in final_scores.items()}
                wandb.log(final_dict, step=global_step)

        if wcfg.wandb_log and wandb.run is not None:
            wandb.finish()

        return global_step
