"""Joint alternating training of NAMM + LLM adapter (ES or LoRA).

Alternates between:
  Stage A: Train NAMM eviction policy (adapter frozen) via CMA-ES
  Stage B: Train LLM adapter (NAMM frozen) via ES or gradient-based LoRA

The base LLM is always frozen. Only the NAMM policy and the adapter (ES
weight perturbations or LoRA A/B matrices) are optimized, in alternation.

Experiments are organized as:
    experiments/experiment_N/joint_{es,lora}/run_name/
        config.json, results.json, namm/, adapter/

Usage:
    # ES adapter path:
    python scripts/run_joint.py --config scripts/configs/joint_default.yaml \
        --run_name joint_es_test --adapter_type es

    # LoRA adapter path:
    python scripts/run_joint.py --config scripts/configs/joint_default.yaml \
        --run_name joint_lora_test --adapter_type lora

    # With pre-trained NAMM checkpoint:
    python scripts/run_joint.py --config scripts/configs/joint_default.yaml \
        --run_name joint_es_warm --adapter_type es \
        --namm_checkpoint exp_local/pretrained/namm.pt

    # Quick smoke test:
    python scripts/run_joint.py --run_name test --adapter_type es \
        --num_outer_loops 2 --namm_iterations_per_stage 3 \
        --adapter_iterations_per_stage 2 --population_size 2 \
        --mini_batch_size 2
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch

logging.getLogger("transformers.generation.stopping_criteria").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
from namm.run_utils import make_eval_model, make_task_sampler
from namm.trainer import MemoryTrainer, WandbConfig
from es_finetuning import ESConfig, ESTrainer
from es_finetuning.device import get_device
from grad_lora_finetuning import LoRAGradTrainer, LoRATrainerConfig
from experiment_utils import (
    get_or_create_experiment,
    load_config_defaults,
    load_hydra_config,
    get_base_llm_param_names,
    make_resample_fn,
    make_evaluate_fn,
    make_full_eval_fn,
    EXPERIMENTS_DIR,
)
from joint_metrics import JointMetricsTracker
from utils.hydra_helpers import assert_fair01_test_size


# ── NAMM ↔ model helpers ────────────────────────────────────────────────

def set_namm_params_from_evo(evolution_algorithm, memory_model,
                             memory_policy, device):
    """Transfer best NAMM params from evolution algorithm into the model."""
    evo_state = evolution_algorithm.state_dict()
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


def load_namm_checkpoint(path, evolution_algorithm, memory_model,
                         memory_policy, device):
    """Load a NAMM checkpoint file and apply it to the model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    evo_state = ckpt['evolution_state']

    # Filter out stored_buffers_to_save.* keys — CMA_ES doesn't expect them.
    # They belong on the model, not the evolution algorithm.
    buffers_prefix = 'stored_buffers_to_save.'
    evo_state_clean = {k: v for k, v in evo_state.items()
                       if not k.startswith(buffers_prefix)}
    evolution_algorithm.load_state_dict(evo_state_clean)

    # Load buffers into the model
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)

    set_namm_params_from_evo(
        evolution_algorithm, memory_model, memory_policy, device)
    print(f"  Loaded NAMM checkpoint from iter {ckpt.get('iter_num', '?')}, "
          f"best_val={ckpt.get('best_val_loss', '?')}")
    return ckpt


def freeze_lora_params(model):
    """Freeze all LoRA parameters (requires_grad -> False)."""
    count = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            count += 1
    return count


def unfreeze_lora_params(model):
    """Unfreeze LoRA A/B parameters (requires_grad -> True)."""
    count = 0
    # After PEFT wrapping, LoRA params contain 'lora_' in their names.
    # The inner model is model.model (WrappedLlama.model = PeftModel).
    inner = model.model if hasattr(model, 'model') else model
    for n, p in inner.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(True)
            count += 1
    return count


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint alternating NAMM + adapter training")

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (see scripts/configs/joint_default.yaml)")

    # Experiment hierarchy
    parser.add_argument("--run_name", type=str, default=None,
                        help="(required) Name for this run")
    parser.add_argument("--experiment", type=int, default=None,
                        help="Experiment ID (default: most recent active or new)")

    # Joint training
    parser.add_argument("--adapter_type", type=str, required=True,
                        choices=["es", "lora"],
                        help="Adapter training method")
    parser.add_argument("--num_outer_loops", type=int, default=2,
                        help="K: number of Stage A → Stage B cycles "
                             "(FAIR-01 M4: 2)")
    parser.add_argument("--namm_iterations_per_stage", type=int, default=100,
                        help="N: CMA-ES iterations per NAMM stage "
                             "(FAIR-01 M4: 100, total = 200 = M2 budget)")
    parser.add_argument("--adapter_iterations_per_stage", type=int, default=25,
                        help="M: ES iterations per adapter stage (ES path)")
    parser.add_argument("--lora_epochs_per_stage", type=int, default=75,
                        help="Epochs per adapter stage (LoRA path) "
                             "(FAIR-01 M4: 75, total = 150 = M1 budget)")
    parser.add_argument("--eval_after_each_loop",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Run full evaluation after each outer loop")
    parser.add_argument("--detailed_diagnostics",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Enable expensive diagnostics (e.g. adapter-"
                             "disabled co-adaptation eval)")

    # NAMM config
    parser.add_argument("--run_config", type=str,
                        default="namm_bam_i1_llama32_1b_5t",
                        help="Hydra run config (FAIR-01 M4: 5-task variant)")
    parser.add_argument("--namm_checkpoint", type=str, default=None,
                        help="Initial NAMM checkpoint (null = from scratch)")
    parser.add_argument("--cache_size", type=int, default=None)
    parser.add_argument("--namm_eval_interval", type=int, default=100,
                        help="NAMM eval interval within each stage")

    # ES hyperparameters
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--population_size", type=int, default=8)
    parser.add_argument("--noise_mode", type=str, default="correlated",
                        choices=["correlated", "iid"])
    parser.add_argument("--initial_seed", type=int, default=33)
    parser.add_argument("--mini_batch_size", type=int, default=16)

    # LoRA hyperparameters
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "v_proj"])
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_batch_size", type=int, default=1,
                        help="Per-device batch size for LoRA training stages "
                             "(NAMM active -> typically 1)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=3500)
    parser.add_argument("--sft_mode",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="FAIR-01 M4: true (chat-template, answer-only loss)")
    parser.add_argument("--lora_eval_interval", type=int, default=999999,
                        help="Eval interval within LoRA stages "
                             "(999999 = skip periodic eval)")
    parser.add_argument("--lora_log_interval", type=int, default=10,
                        help="Logging interval within LoRA stages")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Training dtype for LoRA stages")

    # Data (FAIR-01 fixed)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--min_conditioning_length", type=int, default=4096,
                        help="Drop prompts shorter than this (FAIR-01: 4096)")
    parser.add_argument("--max_conditioning_length", type=int, default=6500,
                        help="Drop prompts longer than this (FAIR-01: 6500)")
    parser.add_argument("--max_answer_tokens", type=int, default=64,
                        help="Drop samples whose shortest answer exceeds "
                             "this token count (FAIR-01: 64)")
    parser.add_argument("--filter_by_tokens", type=int, default=None)
    parser.add_argument("--filter_answers_by_tokens", type=int, default=64)

    # Evaluation batch size
    parser.add_argument("--batch_size", type=int, default=None)

    # Extra Hydra overrides
    parser.add_argument("--override", action="append", default=[])

    load_config_defaults(parser)
    args = parser.parse_args()
    if not args.run_name:
        parser.error("--run_name is required (via CLI or config file)")
    return args


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    method = f"joint_{args.adapter_type}"
    experiment_name, manifest = get_or_create_experiment(args.experiment)
    run_dir: str = os.path.join(EXPERIMENTS_DIR, experiment_name,
                           method, args.run_name)

    if (os.path.exists(run_dir)
            and os.path.exists(os.path.join(run_dir, "results.json"))):
        print(f"ERROR: Run already exists with results: {run_dir}")
        sys.exit(1)

    os.makedirs(run_dir, exist_ok=True)
    namm_dir = os.path.join(run_dir, "namm")
    adapter_dir = os.path.join(run_dir, "adapter")
    os.makedirs(namm_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()
    device_str = str(device)

    total_namm = args.num_outer_loops * args.namm_iterations_per_stage
    print()
    print("=" * 60)
    print(f"FAIR-01 Training Summary  [Joint NAMM + "
          f"{args.adapter_type.upper()}]")
    print("-" * 60)
    print(f"  experiment        : {experiment_name}")
    print(f"  run_name          : {args.run_name}")
    print(f"  output            : {run_dir}")
    print("  -- Compute --")
    print(f"  outer_loops (K)   : {args.num_outer_loops}")
    print(f"  namm_iters/stage  : {args.namm_iterations_per_stage}")
    print(f"  total_namm_iters  : {total_namm}")
    if args.adapter_type == "es":
        total_es = (args.num_outer_loops
                    * args.adapter_iterations_per_stage)
        print(f"  es_iters/stage    : "
              f"{args.adapter_iterations_per_stage}")
        print(f"  total_es_iters    : {total_es}")
        print(f"  sigma             : {args.sigma}")
        print(f"  alpha             : {args.alpha}")
        print(f"  population_size   : {args.population_size}")
        print(f"  noise_mode        : {args.noise_mode}")
        print(f"  mini_batch_size   : {args.mini_batch_size}")
    else:
        total_epochs = (args.num_outer_loops
                        * args.lora_epochs_per_stage)
        eff_batch = (args.lora_batch_size
                     * args.gradient_accumulation_steps)
        print(f"  lora_epochs/stage : {args.lora_epochs_per_stage}")
        print(f"  total_lora_epochs : {total_epochs}")
        print(f"  lora_rank         : {args.lora_rank}")
        print(f"  lora_targets      : {args.lora_target_modules}")
        print(f"  learning_rate     : {args.learning_rate}")
        print(f"  lora_batch_size   : {args.lora_batch_size}")
        print(f"  grad_accum_steps  : "
              f"{args.gradient_accumulation_steps}")
        print(f"  effective_batch   : {eff_batch}")
        print(f"  max_seq_len       : {args.max_seq_len}")
        print(f"  dtype             : {args.dtype}")
    print("  -- Data --")
    print(f"  train_split       : {args.train_split}")
    print(f"  val_split         : {args.val_split}")
    print(f"  split_seed        : {args.split_seed}")
    print(f"  min_cond_length   : {args.min_conditioning_length}")
    print(f"  max_cond_length   : {args.max_conditioning_length}")
    print(f"  max_answer_tokens : {args.max_answer_tokens}")
    print("  -- Memory --")
    print(f"  cache_size        : {args.cache_size}")
    print("=" * 60)
    print()

    # ── 1. Load Hydra config and model ───────────────────────────────────

    print("Loading config and model via Hydra...")
    hydra_overrides = []
    if args.batch_size is not None:
        hydra_overrides.append(f"batch_size={args.batch_size}")
    if args.cache_size is not None:
        hydra_overrides.append(f"cache_size={args.cache_size}")
        hydra_overrides.append(f"max_memory_length={args.cache_size}")
    hydra_overrides.extend(args.override)
    cfg = load_hydra_config(args.run_config, extra_overrides=hydra_overrides)

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    # ── 2. Load initial NAMM checkpoint (optional) ───────────────────────

    if args.namm_checkpoint:
        print(f"Loading initial NAMM checkpoint: {args.namm_checkpoint}")
        load_namm_checkpoint(
            args.namm_checkpoint, evolution_algorithm,
            memory_model, memory_policy, device_str)

    # ── 3. Create task sampler ───────────────────────────────────────────

    print("Creating task sampler...")
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=args.train_split, split_seed=args.split_seed)

    if args.filter_by_tokens is not None:
        task_sampler.filter_by_token_count(
            memory_evaluator.tokenizer, args.filter_by_tokens)
    task_sampler.filter_answers_by_token_count(
        memory_evaluator.tokenizer, args.filter_answers_by_tokens)

    # Apply the 3-way split used by M1/M2/M3 — passing length filters and
    # the tokenizer is load-bearing: without them the splitter ignores the
    # FAIR-01 prompt-length bounds and joint training sees a different
    # eligible set than the other conditions.
    task_sampler.apply_train_val_test_split(
        train_frac=args.train_split,
        val_frac=args.val_split,
        max_conditioning_length=args.max_conditioning_length,
        min_conditioning_length=args.min_conditioning_length,
        tokenizer=memory_evaluator.tokenizer,
    )
    n_test = assert_fair01_test_size(
        task_sampler,
        run_config=args.run_config,
        train_frac=args.train_split,
        val_frac=args.val_split,
        min_conditioning_length=args.min_conditioning_length,
        max_conditioning_length=args.max_conditioning_length,
    )
    test_idxs = task_sampler.get_split_indices('test')
    print(f"  Test split size: {n_test} prompts across "
          f"{len(test_idxs)} tasks (FAIR-01 target: 70)")

    # MemoryTrainer expects these attributes on the task_sampler (it reads
    # training_tasks_subset / test_tasks_subset directly).  TaskSampler
    # stores them as lb_training_tasks / lb_test_tasks, so we alias them.
    if not hasattr(task_sampler, 'training_tasks_subset'):
        task_sampler.training_tasks_subset = task_sampler.lb_training_tasks
    if not hasattr(task_sampler, 'test_tasks_subset'):
        task_sampler.test_tasks_subset = task_sampler.lb_test_tasks

    # ── 4. Create NAMM trainer (reused across all outer loops) ───────────

    print("Creating NAMM trainer...")

    # Instantiate TrainerConfig from Hydra, then override for joint use.
    namm_trainer_config = hydra.utils.instantiate(cfg.trainer_config)
    namm_trainer_config.out_dir = namm_dir
    namm_trainer_config.max_iters = args.namm_iterations_per_stage - 1
    namm_trainer_config.always_save_checkpoint = True
    namm_trainer_config.eval_interval = args.namm_eval_interval
    namm_trainer_config.eval_only = False
    namm_trainer_config.init_from = None

    namm_wandb_config = WandbConfig(
        wandb_log=False,
        wandb_project="Experiments",
        wandb_entity="SNLP_NAMM",
        wandb_run_name=args.run_name,
        wandb_group_name=experiment_name,
    )

    namm_trainer = MemoryTrainer(
        device=device_str,
        evaluation_model=memory_evaluator,
        task_sampler=task_sampler,
        evolution_algorithm=evolution_algorithm,
        trainer_config=namm_trainer_config,
        wandb_config=namm_wandb_config,
        auxiliary_loss=auxiliary_loss,
        scratch=True,
    )

    # ── 5. Apply LoRA adapters if needed ─────────────────────────────────
    #
    # This MUST happen after MemoryTrainer creation, because MemoryTrainer
    # casts the model to bfloat16 in __init__. LoRA then forces float32 on
    # its A/B matrices, which would be lost if we applied LoRA first.
    # Since MemoryTrainer holds a reference to memory_model (not a copy),
    # subsequent LoRA application is visible to the trainer's evaluator.

    if args.adapter_type == "lora":
        memory_model.apply_lora_adapters(
            rank=args.lora_rank,
            target_modules=args.lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        alpha_effective = (args.lora_alpha
                           if args.lora_alpha is not None
                           else 2 * args.lora_rank)
        print(f"LoRA applied: rank={args.lora_rank}, "
              f"targets={args.lora_target_modules}, "
              f"alpha={alpha_effective}")
        # Freeze LoRA for the first NAMM stage
        n_frozen = freeze_lora_params(memory_model)
        print(f"  Froze {n_frozen} LoRA parameter tensors for Stage A")

    # ── 6. Prepare ES components if needed ───────────────────────────────

    if args.adapter_type == "es":
        param_names = get_base_llm_param_names(memory_model)
        print(f"ES: {len(param_names)} base LLM parameters to optimize")

        resample_fn = make_resample_fn(
            task_sampler, args.mini_batch_size, train=True)
        evaluate_fn = make_evaluate_fn(
            task_sampler, memory_evaluator, args.mini_batch_size, train=True)

    # ── 7. Auto-detect batch size ────────────────────────────────────────

    if memory_evaluator.batch_size == "auto":
        print("Auto-detecting optimal batch size...")
        detected_bs = memory_evaluator._detect_batch_size()
        memory_evaluator.batch_size_per_gpu = detected_bs
        memory_evaluator.batch_size = detected_bs
        print(f"  Auto-detected batch size: {detected_bs}")

    # ── 8. Evaluation function ───────────────────────────────────────────

    full_eval_fn = (make_full_eval_fn(task_sampler, memory_evaluator)
                    if args.eval_after_each_loop else None)

    # ── 9. Save config ───────────────────────────────────────────────────

    config_dict = {
        "adapter_type": args.adapter_type,
        "num_outer_loops": args.num_outer_loops,
        "namm_iterations_per_stage": args.namm_iterations_per_stage,
        "run_config": args.run_config,
        "namm_checkpoint": args.namm_checkpoint,
        "cache_size": args.cache_size,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "split_seed": args.split_seed,
        "min_conditioning_length": args.min_conditioning_length,
        "max_conditioning_length": args.max_conditioning_length,
        "max_answer_tokens": args.max_answer_tokens,
    }
    if args.adapter_type == "es":
        config_dict.update({
            "adapter_iterations_per_stage": args.adapter_iterations_per_stage,
            "sigma": args.sigma,
            "alpha": args.alpha,
            "population_size": args.population_size,
            "noise_mode": args.noise_mode,
            "initial_seed": args.initial_seed,
            "mini_batch_size": args.mini_batch_size,
        })
    else:
        config_dict.update({
            "lora_epochs_per_stage": args.lora_epochs_per_stage,
            "lora_rank": args.lora_rank,
            "lora_target_modules": args.lora_target_modules,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "lora_batch_size": args.lora_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_len": args.max_seq_len,
            "sft_mode": args.sft_mode,
            "lora_eval_interval": args.lora_eval_interval,
            "lora_log_interval": args.lora_log_interval,
            "dtype": args.dtype,
        })

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved: {config_path}")

    # ── 10. Alternating training loop ────────────────────────────────────

    history = {
        "outer_loop": [],
        "namm_time_s": [],
        "adapter_time_s": [],
        "eval_scores": [],
        "eval_time_s": [],
    }
    tracker = JointMetricsTracker(
        output_dir=run_dir,
        wandb_run=None,
        detailed_diagnostics=args.detailed_diagnostics,
    )
    total_start = time.time()

    for k in range(args.num_outer_loops):
        print(f"\n{'='*60}")
        print(f"OUTER LOOP {k+1}/{args.num_outer_loops}")
        print(f"{'='*60}")

        # ── Stage A: Train NAMM (adapter frozen) ────────────────────────

        namm_start_iter = k * args.namm_iterations_per_stage
        namm_end_iter = (k + 1) * args.namm_iterations_per_stage - 1

        print(f"\n--- Stage A: Train NAMM "
              f"(iter {namm_start_iter}..{namm_end_iter}) ---")

        # Update NAMM trainer iteration bounds for this stage.
        # MemoryTrainer.train() loops: for iter_num in range(start_iter, max_iters+1)
        namm_trainer.max_iters = namm_end_iter
        if k > 0:
            namm_trainer.start_iter = namm_start_iter
            namm_trainer.force_initial_re_eval = False

        namm_start = time.time()
        with torch.no_grad(), tracker.capture_namm_fitness(namm_trainer):
            namm_trainer.train()
        namm_time = time.time() - namm_start
        print(f"Stage A complete in {namm_time:.1f}s")

        # Transfer best NAMM params to model for Stage B
        set_namm_params_from_evo(
            evolution_algorithm, memory_model, memory_policy, device_str)

        tracker.log_namm_stage_end(
            outer_loop=k,
            evolution_algorithm=evolution_algorithm,
            memory_policy=memory_policy,
            memory_evaluator=memory_evaluator,
            task_sampler=task_sampler,
            stage_time_s=namm_time,
        )

        # Save NAMM checkpoint for this stage
        namm_ckpt_path = os.path.join(namm_dir, f"namm_stage_{k}.pt")
        namm_trainer._save_ckpt(
            iter_num=namm_end_iter, save_path=namm_ckpt_path)
        # Overwrite latest.pt so A4/eval always has a stable path to the
        # most recently completed NAMM stage checkpoint.
        shutil.copy2(namm_ckpt_path,
                     os.path.join(namm_dir, "latest.pt"))
        print(f"  NAMM checkpoint: {namm_ckpt_path}")

        # ── Stage B: Train adapter (NAMM frozen) ────────────────────────

        print(f"\n--- Stage B: Train {args.adapter_type.upper()} adapter "
              f"(NAMM frozen) ---")

        adapter_stage_dir = os.path.join(adapter_dir, f"stage_{k}")
        os.makedirs(adapter_stage_dir, exist_ok=True)

        adapter_start = time.time()

        if args.adapter_type == "es":
            _run_es_stage(
                args, memory_model, param_names,
                evaluate_fn, resample_fn, adapter_stage_dir)

        elif args.adapter_type == "lora":
            _run_lora_stage(
                args, k, cfg, memory_model, memory_evaluator.tokenizer,
                memory_evaluator, task_sampler, memory_policy,
                experiment_name, adapter_stage_dir, device_str)

        adapter_time = time.time() - adapter_start
        print(f"Stage B complete in {adapter_time:.1f}s")

        tracker.log_adapter_stage_end(
            outer_loop=k,
            adapter_type=args.adapter_type,
            memory_model=memory_model,
            stage_dir=adapter_stage_dir,
            stage_time_s=adapter_time,
        )

        # Save adapter checkpoint for this stage
        if args.adapter_type == "es":
            # ES final checkpoint is already saved by ESTrainer
            es_ckpt = os.path.join(
                adapter_stage_dir, "checkpoints", "es_checkpoint_final.pt")
            if os.path.exists(es_ckpt):
                print(f"  ES checkpoint: {es_ckpt}")

        # ── Optional evaluation ──────────────────────────────────────────

        eval_scores = {}
        eval_time = 0.0
        adapter_off_scores = None
        if full_eval_fn:
            print(f"\n--- Evaluating after outer loop {k+1} ---")
            eval_start = time.time()
            # Ensure NAMM params are set for evaluation
            set_namm_params_from_evo(
                evolution_algorithm, memory_model, memory_policy, device_str)
            with torch.no_grad():
                eval_result = full_eval_fn(memory_model)
            eval_time = time.time() - eval_start
            eval_scores = eval_result.get("scores", {})
            for task_name, score in sorted(eval_scores.items()):
                task_short = task_name.split("/")[-1]
                print(f"  {task_short}: {score:.2f}")
            print(f"  Eval time: {eval_time:.1f}s")

            if args.detailed_diagnostics:
                adapter_off_scores = tracker.evaluate_adapter_disabled(
                    memory_model=memory_model, full_eval_fn=full_eval_fn)

        # ── Record history ───────────────────────────────────────────────

        history["outer_loop"].append(k)
        history["namm_time_s"].append(round(namm_time, 1))
        history["adapter_time_s"].append(round(adapter_time, 1))
        history["eval_scores"].append(
            {k: round(v, 4) for k, v in eval_scores.items()})
        history["eval_time_s"].append(round(eval_time, 1))

        tracker.log_outer_loop_end(
            outer_loop=k,
            eval_scores=eval_scores,
            eval_time_s=eval_time,
            adapter_disabled_scores=adapter_off_scores,
            total_time_s=time.time() - total_start,
        )

        # Save intermediate results (overwritten each loop)
        results_path = os.path.join(run_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "config": config_dict,
                "history": history,
                "total_time_s": round(time.time() - total_start, 1),
                "completed_loops": k + 1,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

    # ── Done ─────────────────────────────────────────────────────────────

    tracker.finalize()

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Joint training complete in {total_time:.1f}s "
          f"({total_time / 3600:.2f}h)")
    print(f"Results: {results_path}")
    print(f"Figures: {tracker.figures_dir}")
    print(f"{'='*60}")


# ── Stage B runners ──────────────────────────────────────────────────────

def _run_es_stage(args, memory_model, param_names,
                  evaluate_fn, resample_fn, stage_dir):
    """Run one ES adapter training stage."""
    es_config = ESConfig(
        sigma=args.sigma,
        alpha=args.alpha,
        population_size=args.population_size,
        num_iterations=args.adapter_iterations_per_stage,
        noise_mode=args.noise_mode,
        initial_seed=args.initial_seed,
        mini_batch_size=args.mini_batch_size,
        log_dir=stage_dir,
    )

    es_trainer = ESTrainer(
        model=memory_model,
        param_names=param_names,
        evaluate_fn=evaluate_fn,
        config=es_config,
        pre_step_fn=resample_fn,
    )

    with torch.no_grad():
        es_trainer.train()


def _run_lora_stage(args, stage_idx, cfg, memory_model, tokenizer,
                    memory_evaluator, task_sampler, memory_policy,
                    experiment_name, stage_dir, device):
    """Run one LoRA adapter training stage."""
    # Unfreeze LoRA params for training
    n_unfrozen = unfreeze_lora_params(memory_model)
    print(f"  Unfroze {n_unfrozen} LoRA parameter tensors for Stage B")

    task_names = list(cfg.task_sampler.tasks)
    lora_cfg = LoRATrainerConfig(
        out_dir=stage_dir,
        method=f"joint_lora_stage{stage_idx}",
        seed=args.split_seed,
        max_seq_len=args.max_seq_len,
        task_names=task_names,
        num_epochs=args.lora_epochs_per_stage,
        batch_size=args.lora_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        namm_active=True,
        eval_interval=args.lora_eval_interval,
        log_interval=args.lora_log_interval,
        always_save_checkpoint=True,
        init_from=None,
        dtype=args.dtype,
        sft_mode=args.sft_mode,
        train_frac=args.train_split,
        val_frac=args.val_split,
        max_conditioning_length=args.max_conditioning_length,
        min_conditioning_length=args.min_conditioning_length,
        max_answer_tokens=args.max_answer_tokens,
    )

    lora_wandb_config = WandbConfig(
        wandb_log=False,
        wandb_project="Experiments",
        wandb_entity="SNLP_NAMM",
        wandb_run_name=f"{args.run_name}_lora_stage{stage_idx}",
        wandb_group_name=experiment_name,
    )

    lora_trainer = LoRAGradTrainer(
        model=memory_model,
        tokenizer=tokenizer,
        evaluation_model=memory_evaluator,
        task_sampler=task_sampler,
        memory_policy=memory_policy,
        trainer_config=lora_cfg,
        wandb_config=lora_wandb_config,
        device=device,
    )

    lora_trainer.train()

    # Re-freeze LoRA params for next NAMM stage
    n_frozen = freeze_lora_params(memory_model)
    print(f"  Re-froze {n_frozen} LoRA parameter tensors for next Stage A")


if __name__ == "__main__":
    main()
