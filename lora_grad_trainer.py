"""
lora_grad_trainer.py — Gradient-based LoRA training class for NAMM interaction study.

This module is entirely separate from MemoryTrainer and carries no gradient-disabling
decorators anywhere, so the autograd graph survives from the NTP loss back through
the LLM to the LoRA A/B matrices.

Provides:
  - LoRATrainerConfig: dataclass with all training hyperparameters
  - LoRAGradTrainer: training class with optimizer, scheduler, gradient accumulation,
    checkpoint I/O (including AdamW/scheduler state), NAMM-active mode, and artifact
    contract for Phase 4+ evaluation.

Usage:
    from lora_grad_trainer import LoRAGradTrainer, LoRATrainerConfig
    from memory_trainer import WandbConfig

    cfg = LoRATrainerConfig(
        out_dir='exp_local/lora_m1',
        method='lora_grad',
        seed=1337,
        max_seq_len=3500,
        task_names=['qasper', 'narrativeqa', 'passage_retrieval_en'],
        cache_dir='/cs/student/project_msc/2025/csml/gmaralla/.hf_cache',
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
    trainer = LoRAGradTrainer(model, tokenizer, evaluation_model, cfg, wcfg, device='cuda')
    trainer.train()
"""

import csv
import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lora_ntp_dataset import LongBenchNTPDataset, ntp_pad_collate_fn
from memory_trainer import WandbConfig


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
    cache_dir: str                  # HF cache path

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
        trainer_config: LoRATrainerConfig,
        wandb_config: WandbConfig,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluation_model = evaluation_model
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
        print(f"LoRAGradTrainer: {len(self.lora_params)} LoRA parameter tensors identified.")

        # --- PEFT gradient fix (TRAIN-03) ---
        # enable_input_require_grads() is absent in peft v0.11.1.
        # Register a forward hook on the embedding layer so that the output
        # embedding tensor always has requires_grad=True. Without this, the
        # autograd graph is disconnected and loss.backward() computes zero
        # gradients for LoRA matrices in the first decoder layer.
        def make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)

        try:
            # WrappedLlamaForCausalLM exposes the HuggingFace model via .model
            embed_layer = model.model.get_input_embeddings()
        except AttributeError:
            # Fallback: PEFT PeftModel wraps the base model one level deeper
            embed_layer = model.get_input_embeddings()

        embed_layer.register_forward_hook(make_inputs_require_grad)
        print("LoRAGradTrainer: PEFT embedding forward hook registered.")

        # --- ANLYS-01: per-layer retention dict (populated in _train_step) ---
        # Non-empty only when namm_active=True; m1 training path is unaffected.
        self._last_retention_dict = {}

        # --- Build dataset and DataLoader ---
        # LLaMA tokenizer has no pad token — use EOS so collate_fn gets a valid int
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dataset = LongBenchNTPDataset(
            task_names=cfg.task_names,
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            cache_dir=cfg.cache_dir,
            seed=cfg.seed,
        )
        collate_fn = partial(
            ntp_pad_collate_fn,
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
    # Training step helpers
    # -----------------------------------------------------------------------

    def _train_step(self, batch, past_key_values=None):
        """Run one forward + backward pass.

        Args:
            batch: dict with 'input_ids' and 'labels' (from ntp_pad_collate_fn)
            past_key_values: None for NAMM-active mode (reset between docs)

        Returns:
            (unscaled_loss_float, past_key_values_out)
            unscaled_loss is the raw per-step NTP loss for logging.
            past_key_values_out is the updated KV cache (None if namm_active).
        """
        cfg = self.trainer_config
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # NAMM-active: reset KV cache and memory policy buffers before each doc
        if cfg.namm_active:
            # Reset the memory policy's internal attention/position buffers.
            # initialize_buffers() is the confirmed reset method on MemoryPolicy.
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
            # KV cache is reset by passing None (new document)
            past_key_values = None

        # Forward pass — apply_memory_policy=namm_active passes NAMM eviction
        # during forward when namm_active=True (m4-frozen mode).
        # CRITICAL: never pass limit_new_tokens from the training loop.
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            use_cache=True,
            apply_memory_policy=cfg.namm_active,
        )

        loss = outputs.loss / cfg.gradient_accumulation_steps
        loss.backward()

        # ANLYS-01: per-layer token retention logging for NAMM-active training.
        # CONFIRMED: outputs.past_key_values is POST-eviction — memory_policy.update_cache()
        # reassigns outputs.past_key_values before return (llama.py lines 501-509).
        # Each layer_kv[0] has shape [batch, n_heads, seq_len_post_eviction, head_dim].
        # retention_rate = seq_len_post_eviction / n_input  (0.0 to 1.0).
        # This dict is read by train() at log_interval to push to wandb.
        self._last_retention_dict = {}
        if cfg.namm_active and hasattr(outputs, 'past_key_values') \
                and outputs.past_key_values is not None:
            n_input = input_ids.shape[1]
            if n_input > 0:
                for layer_i, layer_kv in enumerate(outputs.past_key_values):
                    if layer_kv is not None and layer_kv[0] is not None:
                        # layer_kv[0]: key cache [batch, n_heads, seq_len_post_eviction, head_dim]
                        n_retained = layer_kv[0].shape[-2]
                        self._last_retention_dict[f'retention/layer_{layer_i}'] = (
                            n_retained / n_input
                        )
                # Runtime guard: warn if all retention values are >= 1.0.
                # If outputs.past_key_values were pre-eviction (a bug), all values equal 1.0.
                # This guard catches that pre/post-eviction confusion early so it does not
                # silently corrupt all ANLYS-01 data for the entire run.
                if self._last_retention_dict and all(
                        v >= 1.0 for v in self._last_retention_dict.values()):
                    import warnings
                    warnings.warn(
                        "ANLYS-01 WARNING: all retention/layer_i values are >= 1.0. "
                        "NAMM may not be evicting tokens (check apply_memory_policy=True "
                        "and that cache_size < sequence length). "
                        "Expected post-eviction KV cache from memory_policy.update_cache().",
                        stacklevel=2,
                    )

        # Return unscaled loss for logging and the updated KV cache
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

    def _save_ckpt(self, step_num: int) -> str:
        """Save LoRA weights, optimizer state, and scheduler state.

        Implements TRAIN-05: optimizer_state_dict and scheduler_state_dict are
        saved so training can be resumed exactly from this step.

        Returns:
            Path to the saved checkpoint file.
        """
        # Determine save directory
        if self.artifact_dir is not None:
            ckpt_dir = self.artifact_dir
        else:
            ckpt_dir = self.trainer_config.out_dir

        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')

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
            'optimizer_state_dict': self.optimizer.state_dict(),    # TRAIN-05
            'scheduler_state_dict': self.scheduler.state_dict(),    # TRAIN-05
            'step_num': step_num,
            'trainer_type': 'lora_grad',
        }

        torch.save(checkpoint, ckpt_path)
        print(f"Saved LoRA checkpoint to {ckpt_path} (step {step_num})")
        return ckpt_path

    def _load_ckpt(self, load_path: str) -> int:
        """Load LoRA weights and optionally optimizer/scheduler state.

        Uses weights_only=False because NAMM checkpoints contain numpy objects.
        Restores optimizer and scheduler state when present (TRAIN-05), enabling
        exact resume from any saved step.

        Args:
            load_path: path to a checkpoint file (NAMM ckpt or LoRA ckpt)

        Returns:
            step_num from checkpoint (0 if not present — fresh NAMM base ckpt)
        """
        print(f"Loading checkpoint from {load_path}")
        # weights_only=False required: NAMM checkpoints contain numpy objects
        # (PyTorch 2.7 changed default to True — see Bug #1 in project memory)
        ckpt = torch.load(load_path, map_location=self.device, weights_only=False)

        if 'lora_state_dict' in ckpt:
            loaded = 0
            for n, p in self.model.model.named_parameters():
                if p.requires_grad and n in ckpt['lora_state_dict']:
                    p.data.copy_(ckpt['lora_state_dict'][n])
                    loaded += 1
            print(f"Loaded LoRA state ({loaded} tensors) from {load_path}")

        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("Restored optimizer + scheduler state (TRAIN-05)")

        return ckpt.get('step_num', 0)

    # -----------------------------------------------------------------------
    # Artifact contract (ARTIFACT-01)
    # -----------------------------------------------------------------------

    def _write_artifact_contract(self, cfg_yaml: str, method: str, seed: int):
        """Create results/{method}/{seed}/ and initialise all artifact files.

        Implements ARTIFACT-01:
          - config.yaml: snapshot of the trainer configuration
          - wandb_run_id.txt: written after wandb.init() via _write_wandb_id()
          - metrics.csv: header written here; rows appended during training

        Args:
            cfg_yaml: YAML string of the configuration (caller serialises)
            method: trainer method name (e.g. 'lora_grad', 'm4_frozen')
            seed: random seed used for this run
        """
        self.artifact_dir = os.path.join('results', method, str(seed))
        os.makedirs(self.artifact_dir, exist_ok=True)

        # 1. Config snapshot
        with open(os.path.join(self.artifact_dir, 'config.yaml'), 'w') as f:
            f.write(cfg_yaml)

        # 2. wandb run ID path — written by _write_wandb_id() after wandb.init()
        self._wandb_id_path = os.path.join(self.artifact_dir, 'wandb_run_id.txt')

        # 3. Metrics CSV — initialise header; rows appended during training
        self._metrics_csv_path = os.path.join(self.artifact_dir, 'metrics.csv')
        with open(self._metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['step', 'loss', 'grad_norm', 'lr']
            )
            writer.writeheader()

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
        """Write wandb run ID to wandb_run_id.txt (called once after wandb.init())."""
        if wandb.run is not None and self._wandb_id_path is not None:
            with open(self._wandb_id_path, 'w') as f:
                f.write(wandb.run.id)

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self):
        """Run full gradient-based LoRA training.

        Outer loop: epochs. Inner loop: dataloader batches.
        Gradient accumulation runs for gradient_accumulation_steps before
        calling _optimizer_step(). Checkpoints saved at eval_interval steps.
        Metrics logged to wandb and metrics.csv at log_interval steps.
        """
        cfg = self.trainer_config
        wcfg = self.wandb_config

        # --- Artifact contract ---
        import dataclasses
        import yaml
        cfg_dict = dataclasses.asdict(cfg)
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
                group=wcfg.wandb_group_name,
                name=wcfg.wandb_run_name,
                config=cfg_dict,
            )
            self._write_wandb_id()

        # --- Set model to training mode ---
        self.model.train()

        # Re-freeze base weights after model.train() (train() sets all params
        # to requires_grad=True; we must restore LoRA-only gradient flow).
        for n, p in self.model.model.named_parameters():
            if 'lora_' not in n:
                p.requires_grad_(False)

        # --- Training loop ---
        global_step = self._resume_step  # resume support
        accum_loss = 0.0
        accum_batches = 0
        t_start = time.time()

        for epoch in range(cfg.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{cfg.num_epochs} ===")
            for batch_idx, batch in enumerate(self.dataloader):
                # Forward + backward (accumulate gradient)
                step_loss, _ = self._train_step(batch)
                accum_loss += step_loss
                accum_batches += 1

                # Optimizer step after accumulating gradient_accumulation_steps batches
                if accum_batches == cfg.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    global_step += 1
                    avg_loss = accum_loss / cfg.gradient_accumulation_steps
                    current_lr = self.scheduler.get_last_lr()[0]

                    # --- Logging ---
                    if global_step % cfg.log_interval == 0:
                        elapsed = time.time() - t_start
                        print(
                            f"step {global_step:6d} | loss {avg_loss:.4f} | "
                            f"grad_norm {grad_norm:.4f} | lr {current_lr:.2e} | "
                            f"elapsed {elapsed:.1f}s"
                        )
                        if wcfg.wandb_log and wandb.run is not None:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/grad_norm': grad_norm,
                                'train/lr': current_lr,
                                'train/step': global_step,
                            }, step=global_step)
                            # Log per-layer retention rates (ANLYS-01).
                            # Non-empty only for namm_active=True; m1 training is unaffected.
                            if self._last_retention_dict:
                                wandb.log(self._last_retention_dict, step=global_step)

                        if self._metrics_csv_path is not None:
                            self._append_metrics(
                                step=global_step,
                                loss=avg_loss,
                                grad_norm=grad_norm,
                                lr=current_lr,
                            )

                    # --- Checkpoint ---
                    if cfg.always_save_checkpoint and (
                        global_step % cfg.eval_interval == 0
                    ):
                        self._save_ckpt(global_step)

                    # Reset accumulators
                    accum_loss = 0.0
                    accum_batches = 0

        # --- Final checkpoint ---
        print(f"\nTraining complete. Total gradient steps: {global_step}")
        self._save_ckpt(global_step)

        if wcfg.wandb_log and wandb.run is not None:
            wandb.finish()

        return global_step
