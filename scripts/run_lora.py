"""LoRA gradient fine-tuning of Llama 3.2 1B with optional NAMM.

Parallel to run_es.py — same experiment management pattern, but dispatches
to LoRAGradTrainer instead of ESTrainer. Runs WITHOUT torch.no_grad()
around the training loop so autograd flows through LoRA A/B matrices.

Usage:
    # LoRA-only (m1 condition):
    python scripts/run_lora.py --config scripts/configs/lora_default.yaml --run_name m1_test

    # LoRA + frozen NAMM (m4-frozen condition):
    python scripts/run_lora.py --config scripts/configs/lora_default.yaml \
        --namm_active --namm_checkpoint path/to/namm.pt --run_name m4_test

    # Quick smoke test:
    python scripts/run_lora.py --run_name test \
        --num_epochs 1 --eval_interval 5 --max_seq_len 512
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

logging.getLogger("transformers.generation.stopping_criteria").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device
from grad_lora_finetuning import LoRAGradTrainer, LoRATrainerConfig
from namm.trainer import WandbConfig
from experiment_utils import (
    get_or_create_experiment, get_or_create_experiment_gcs,
    claim_run_gcs, load_config_defaults, load_hydra_config,
    EXPERIMENTS_DIR,
)


# ── CLI ─────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA gradient fine-tuning of base LLM weights with optional NAMM")

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (see scripts/configs/lora_default.yaml)")

    # Experiment hierarchy
    parser.add_argument("--run_name", type=str, default=None,
                        help="(required) Name for this run")
    parser.add_argument("--experiment", type=int, default=None)
    parser.add_argument("--method", type=str, default="lora_grad")

    # LoRA hyperparameters
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=3500)
    parser.add_argument("--sft_mode", action="store_true", default=False)

    # NAMM config
    parser.add_argument("--namm_active", action="store_true", default=False)
    parser.add_argument("--namm_checkpoint", type=str, default=None)
    parser.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b")
    parser.add_argument("--cache_size", type=int, default=None)

    # Data filtering
    parser.add_argument("--filter_by_tokens", type=int, default=None)
    parser.add_argument("--filter_answers_by_tokens", type=int, default=64)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=40)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--batch_size_eval", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop after N evals with no val F1 improvement (0 = disabled)")

    # Checkpointing & GCS
    parser.add_argument("--gcs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    # Wandb
    parser.add_argument("--wandb_log", type=bool, default=True)
    parser.add_argument("--wandb_entity", type=str, default="SNLP_NAMM")
    parser.add_argument("--wandb_project", type=str, default="Experiments")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group_name", type=str, default=None)

    # Extra Hydra overrides
    parser.add_argument("--override", action="append", default=[])

    load_config_defaults(parser)
    args = parser.parse_args()
    if not args.run_name:
        parser.error("--run_name is required (via CLI or config file)")
    return args


def main():
    args = parse_args()

    # Setup GCS client if requested
    gcs = None
    if args.gcs:
        try:
            from es_finetuning.gcs import GCSClient
            gcs = GCSClient()
        except Exception:
            print("WARNING: GCS not available, running without GCS")
            gcs = None

    # Resolve "latest" NAMM checkpoint from GCS
    if args.namm_checkpoint == "latest" and gcs:
        cache_dir = os.path.join(REPO_ROOT, "exp_local", "pretrained")
        args.namm_checkpoint = gcs.download_latest_pretrained(cache_dir)

    # Setup experiment hierarchy
    if gcs:
        experiment_name, manifest = get_or_create_experiment_gcs(
            gcs, args.experiment)
    else:
        experiment_name, manifest = get_or_create_experiment(args.experiment)

    run_dir = os.path.join(EXPERIMENTS_DIR, experiment_name,
                           args.method, args.run_name)

    if (os.path.exists(run_dir)
            and os.path.exists(os.path.join(run_dir, "results.json"))
            and args.resume_checkpoint is None):
        print(f"ERROR: Run already exists with results: {run_dir}")
        sys.exit(1)

    os.makedirs(run_dir, exist_ok=True)

    if gcs:
        claim_run_gcs(gcs, experiment_name, args.method, args.run_name)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()

    print("=" * 60)
    print("LoRA Gradient Fine-Tuning: Llama 3.2 1B")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Method: {args.method}")
    print(f"Run: {args.run_name}")
    print(f"Output: {run_dir}")
    print(f"LoRA: rank={args.lora_rank}, targets={args.lora_target_modules}")
    print(f"Training: lr={args.learning_rate}, epochs={args.num_epochs}, "
          f"batch={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
    print(f"NAMM active: {args.namm_active}")
    if args.namm_active:
        print(f"NAMM checkpoint: {args.namm_checkpoint}")
    print()

    # 1. Load config and model
    print("Loading config and model via Hydra...")
    hydra_overrides = []
    if args.batch_size_eval is not None:
        hydra_overrides.append(f"batch_size={args.batch_size_eval}")
    if args.cache_size is not None:
        hydra_overrides.append(f"cache_size={args.cache_size}")
        hydra_overrides.append(f"max_memory_length={args.cache_size}")
    hydra_overrides.extend(args.override)
    cfg = load_hydra_config(args.run_config, extra_overrides=hydra_overrides)

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    # 1b. Swap to Recency (passthrough) policy when NAMM is inactive.
    # make_eval_model() always instantiates the full NAMM scoring network
    # from the Hydra config. When namm_active=False, running the full STFT +
    # scoring pipeline on every generated token during eval is unnecessary
    # and makes baseline eval ~10-20x slower.
    if not args.namm_active:
        from namm.policy.base import Recency
        recency_policy = Recency(cache_size=args.cache_size)
        # swap_memory_policy on the evaluator also swaps on its inner model
        # (memory_model), so one call is sufficient.
        memory_evaluator.swap_memory_policy(recency_policy)
        memory_policy = recency_policy

        # The Hydra config (namm_bam_i1_llama32_1b) sets max_memory_length=1024
        # and batch_size="auto" — tuned for NAMM's small KV cache. Without NAMM
        # eviction the KV cache grows to the full context length (~6500 tokens),
        # so auto-detection underestimates memory (calibrated at 1024) and picks
        # a batch size that OOMs or thrashes during actual generation. Fix both:
        memory_evaluator.max_memory_length = memory_evaluator.max_conditioning_length
        if memory_evaluator.batch_size == "auto":
            memory_evaluator.batch_size = 1
        print(f"Swapped to Recency policy (namm_active=False, "
              f"cache_size={args.cache_size}, "
              f"eval_batch_size={memory_evaluator.batch_size})")

    # 2. Load NAMM checkpoint if needed
    if args.namm_checkpoint:
        print(f"Loading NAMM checkpoint: {args.namm_checkpoint}")
        ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                          weights_only=False)
        evo_state = ckpt['evolution_state']
        best_member = evo_state['best_member']
        params = best_member.unsqueeze(0).to(device)
        memory_model.set_memory_params(params)

        buffers_prefix = 'stored_buffers_to_save.'
        buffers_dict = {
            k[len(buffers_prefix):]: v.to(device)
            for k, v in evo_state.items()
            if k.startswith(buffers_prefix)
        }
        if buffers_dict:
            memory_model.load_buffers_dict(buffers_dict=buffers_dict)

        batch_idxs = np.zeros([1])
        memory_policy.set_params_batch_idxs(batch_idxs)
        print(f"  Loaded NAMM best_member ({best_member.shape[0]} params)")

    # 3. Create task sampler
    print("Creating task sampler...")
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=args.train_split, split_seed=args.split_seed)

    # 3b. Token-based filtering
    if args.filter_by_tokens is not None:
        task_sampler.filter_by_token_count(
            memory_evaluator.tokenizer, args.filter_by_tokens)

    task_sampler.filter_answers_by_token_count(
        memory_evaluator.tokenizer, args.filter_answers_by_tokens)

    # 3c. Apply 3-way split for LoRA training with exact token-based filtering.
    # This ensures LoRA uses the same eligible set as NAMM training.
    task_sampler.apply_train_val_test_split(
        train_frac=args.train_split,
        val_frac=args.val_split,
        max_conditioning_length=cfg.get('max_conditioning_length', 6500),
        min_conditioning_length=cfg.get('min_conditioning_length', None),
        tokenizer=memory_evaluator.tokenizer,
    )

    # 3d. Wrap eval prompts in chat template to match SFT training format
    if args.sft_mode:
        task_sampler.apply_chat_template_to_prompts(memory_evaluator.tokenizer)

    # 4. Cast model to bfloat16 and move to device
    memory_model.to(dtype=torch.bfloat16, device=device)

    # 5. Gradient checkpointing (only when NAMM is inactive)
    if not args.namm_active:
        memory_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled (namm_active=False).")

    # 6. Apply LoRA adapters
    memory_model.apply_lora_adapters(
        rank=args.lora_rank,
        target_modules=args.lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(f"LoRA applied: rank={args.lora_rank}, targets={args.lora_target_modules}, "
          f"alpha={args.lora_alpha or args.lora_rank}, dropout={args.lora_dropout}")

    # 7. Build LoRA trainer config
    task_names = list(cfg.task_sampler.tasks)
    lora_cfg = LoRATrainerConfig(
        out_dir=run_dir,
        method=args.method,
        seed=args.split_seed,
        max_seq_len=args.max_seq_len,
        task_names=task_names,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        namm_active=args.namm_active,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        always_save_checkpoint=True,
        init_from=args.resume_checkpoint,
        dtype='bfloat16',
        sft_mode=args.sft_mode,
        train_frac=args.train_split,
        val_frac=args.val_split,
        max_conditioning_length=cfg.get('max_conditioning_length', 6500),
        min_conditioning_length=cfg.get('min_conditioning_length', None),
        max_answer_tokens=args.filter_answers_by_tokens,
        early_stopping_patience=args.early_stopping_patience,
    )

    wandb_cfg = WandbConfig(
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name or args.run_name,
        wandb_group_name=args.wandb_group_name or experiment_name,
    )

    # 8. Build tokenizer for trainer
    with torch.no_grad():
        import hydra
        tokenizer = hydra.utils.call(cfg.tokenizer)

    # 9. Build trainer — NO torch.no_grad() wrapper
    trainer = LoRAGradTrainer(
        model=memory_model,
        tokenizer=tokenizer,
        evaluation_model=memory_evaluator,
        task_sampler=task_sampler,
        memory_policy=memory_policy,
        trainer_config=lora_cfg,
        wandb_config=wandb_cfg,
        device=device,
        gcs_client=gcs,
        experiment_name=experiment_name,
        method=args.method,
        run_name=args.run_name,
    )

    print(f"Trainer: lora_grad [{type(trainer).__name__}]")

    # 10. Train — autograd graph must survive through this call
    trainer.train()

    print("Done.")


if __name__ == "__main__":
    main()
