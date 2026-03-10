#!/usr/bin/env python3
"""
Shared evaluation entry point for all conditions (FAIR-02).

All conditions (m1, m2, m3, m4-frozen, m4-iterative) must be evaluated
through this script to guarantee:
  - Same eval splits (EVAL_TASKS)
  - Same metrics (EVAL_METRICS)
  - Same wandb group (EVAL_WANDB_GROUP)
  - Same code path (MemoryHFEvaluator — wired in Phase 4)

Usage:
  python run_eval.py --ckpt <path/to/ckpt.pt> --method <m1|m2|m3|m4_frozen|m4_iterative> --seed <int>

Do NOT add per-condition branching to run_eval(). If a condition needs
different eval behavior, fix the checkpoint or model, not this script.
"""
import argparse
import os
import torch
import wandb

# ── FAIR-02 CONSTANTS — do not change without updating all condition configs ──
EVAL_WANDB_GROUP = 'Llama-3.2-1B/grad-lora-study'
EVAL_WANDB_PROJECT = 'memory_evolution_hf'
EVAL_TASKS = ['qasper', 'narrativeqa', 'passage_retrieval_en']
EVAL_METRICS = ['qa_f1', 'qa_f1', 'retrieval_f1']   # one per task
HF_CACHE = '/cs/student/project_msc/2025/csml/gmaralla/.hf_cache'
# ─────────────────────────────────────────────────────────────────────────────

VALID_METHODS = ['base', 'm1', 'm1_sft', 'm2', 'm3', 'm4_frozen', 'm4_iterative']


def parse_args():
    p = argparse.ArgumentParser(description='Shared eval for all conditions (FAIR-02)')
    p.add_argument('--ckpt', required=False, default=None,
                   help='Path to checkpoint file (ckpt.pt). Not required for --method base.')
    p.add_argument('--method', required=True, choices=VALID_METHODS,
                   help='Condition name: m1 | m2 | m3 | m4_frozen | m4_iterative')
    p.add_argument('--seed', type=int, default=1337, help='Random seed used for training run')
    p.add_argument('--device', default='cuda', help='Device to use for eval')
    p.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    p.add_argument('--artifact-dir', default=None,
                   help='Results artifact dir (default: results/{method}/{seed}/)')
    p.add_argument('--run-config', default=None,
                   help='Override Hydra run config name (e.g. recency_baseline_llama32_1b_instruct)')
    p.add_argument('--cache-size', type=int, default=None,
                   help='Override cache_size (e.g. 1024, 2048, 4096)')
    p.add_argument('--held-out-eval', action='store_true',
                   help='Evaluate only on held-out last 20%% of each task (matches train_frac=0.8 SFT split)')
    p.add_argument('--train-frac', type=float, default=0.8,
                   help='Train fraction used during SFT — held-out starts at this index (default 0.8)')
    p.add_argument('--wandb-group', default=None,
                   help='Override wandb group name (default: EVAL_WANDB_GROUP constant)')
    return p.parse_args()


def load_model_from_ckpt(ckpt_path: str, device: str, method: str, artifact_dir: str,
                         run_config_override: str = None, cache_size_override: int = None):
    """
    Load model from checkpoint. Handles both lora_grad and namm_es checkpoint types.

    Strategy:
      - Read saved trainer config from artifact_dir/config.yaml to get init_from,
        lora_rank, lora_target_modules.
      - Select Hydra run config based on method:
          m1          -> full_cache_baseline_llama32_1b (no NAMM, full KV cache)
          m4_frozen   -> namm_bam_eval_llama32_1b  (BAM policy, NAMM architecture)
          m2, m3      -> namm_bam_eval_llama32_1b  (NAMM architecture)
          fallback    -> full_cache_baseline_llama32_1b
      - Call initialize_cfg + make_eval_model to reconstruct the model.
      - Apply LoRA adapters and load LoRA weights from ckpt_path.

    Returns:
        (model, tokenizer, evaluator, task_sampler)
    """
    # --- Lazy imports to keep 'python run_eval.py --help' fast and import-safe ---
    import yaml
    from hydra.core.global_hydra import GlobalHydra
    from utils_hydra import initialize_cfg
    from main import make_eval_model, make_task_sampler

    # 1. Read saved trainer config from artifact_dir/config.yaml
    config_yaml_path = os.path.join(artifact_dir, 'config.yaml')
    saved_cfg = {}
    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, 'r') as f:
            saved_cfg = yaml.safe_load(f) or {}
    else:
        print(f"WARNING: {config_yaml_path} not found. Using defaults for LoRA config.")

    namm_ckpt_path = saved_cfg.get('init_from', None)
    lora_rank = saved_cfg.get('lora_rank', 8)
    lora_target_modules = saved_cfg.get('lora_target_modules', ['q_proj', 'v_proj'])

    # 2. Guard for m4_frozen: init_from must be non-None
    #    LoRAGradTrainer.train() serialises LoRATrainerConfig via dataclasses.asdict()
    #    and writes it to config.yaml via _write_artifact_contract, so init_from is
    #    always present. None here means the checkpoint was trained without a NAMM base.
    if method == 'm4_frozen':
        assert namm_ckpt_path is not None, (
            f"m4_frozen eval requires init_from in {artifact_dir}/config.yaml "
            f"but it was None. Check that _write_artifact_contract serialized "
            f"LoRATrainerConfig.init_from correctly."
        )

    # 3. Select Hydra run config based on method (or use --run-config override)
    if run_config_override:
        run_cfg_name = run_config_override
    elif method in ('m4_frozen', 'm2', 'm3'):
        run_cfg_name = 'namm_bam_eval_llama32_1b'
    elif method in ('base', 'm1', 'm1_sft'):
        run_cfg_name = 'full_cache_baseline_llama32_1b_instruct'
    else:
        run_cfg_name = 'full_cache_baseline_llama32_1b'

    # 4. Build Hydra overrides
    hydra_overrides = [
        f'run@_global_={run_cfg_name}',
        f'wandb_log=false',       # eval script manages wandb independently
        f'eval_only=true',
    ]
    if method in ('m4_frozen', 'm2', 'm3') and namm_ckpt_path is not None:
        hydra_overrides.append(f'init_from={namm_ckpt_path}')
    if cache_size_override is not None:
        hydra_overrides.append(f'cache_size={cache_size_override}')

    # 5. Initialize Hydra config (handle singleton — can only call initialize once/process)
    GlobalHydra.instance().clear()
    cfg = initialize_cfg('cfgs', hydra_overrides=hydra_overrides)

    # 6. Build model, evaluator, task_sampler via the standard make_eval_model path
    (memory_policy, model, evaluator, evo_alg, aux_loss) = make_eval_model(cfg)

    # Retrieve tokenizer from evaluator (MemoryHFEvaluator stores it)
    tokenizer = evaluator.tokenizer if hasattr(evaluator, 'tokenizer') else None

    # Pass tokenizer to TaskSampler so prompts are wrapped with apply_chat_template
    # for instruct models (matches the format used during SFT training).
    task_sampler = make_task_sampler(
        cfg,
        store_gen_outputs=True,
        store_gen_outputs_path=os.path.join(artifact_dir, 'eval_outputs'),
        tokenizer=tokenizer,
    )

    # 7. Apply LoRA adapters and load LoRA weights for lora_grad conditions
    lora_conditions = ('m1', 'm1_sft', 'm2', 'm3', 'm4_frozen', 'm4_iterative')
    # 'base' is intentionally NOT in lora_conditions — base LLaMA eval skips LoRA entirely
    if method == 'base':
        print("Base LLaMA eval: skipping LoRA adapter injection (no checkpoint needed).")
        # No LoRA, no checkpoint loading — pure base model evaluation
    elif method in lora_conditions:
        # Load LoRA weights from the checkpoint
        if ckpt_path is None:
            raise ValueError(f"--ckpt is required for method '{method}' (LoRA checkpoint needed)")
        lora_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Read rank and target_modules from checkpoint lora_config if present —
        # this is more reliable than config.yaml which may not save lora_rank.
        # Falls back to config.yaml values, then to defaults.
        if 'lora_config' in lora_ckpt:
            lora_rank = lora_ckpt['lora_config'].get('rank', lora_rank)
            lora_target_modules = lora_ckpt['lora_config'].get('target_modules', lora_target_modules)
            print(f"LoRA config from checkpoint: rank={lora_rank}, targets={lora_target_modules}")

        # apply_lora_adapters injects PEFT LoRA A/B matrices into the base model
        model.apply_lora_adapters(
            rank=lora_rank,
            target_modules=lora_target_modules,
        )
        if 'lora_state_dict' in lora_ckpt:
            loaded = 0
            for n, p in model.model.named_parameters():
                if p.requires_grad and n in lora_ckpt['lora_state_dict']:
                    p.data.copy_(lora_ckpt['lora_state_dict'][n])
                    loaded += 1
            print(f"Loaded LoRA weights: {loaded} tensors from {ckpt_path}")
        else:
            print(f"WARNING: no 'lora_state_dict' in checkpoint {ckpt_path}. "
                  f"Available keys: {list(lora_ckpt.keys())}")

    # 8. Move model to device
    model.to(device)

    return model, tokenizer, evaluator, task_sampler


def run_eval(model, tokenizer, evaluator, task_sampler, artifact_dir: str,
             held_out_eval: bool = False, train_frac: float = 0.8):
    """
    Run LongBench evaluation on all EVAL_TASKS using shared MemoryHFEvaluator.

    No per-condition branching here — same evaluator, same tasks, same metrics.
    task_sampler already knows which tasks to evaluate (lb_3subset_eval config).

    Args:
        held_out_eval : If True, evaluate only on the last (1-train_frac) fraction of each
                        task's examples — i.e. the examples NOT seen during SFT training.
        train_frac    : Must match the train_frac used in LongBenchSFTDataset (default 0.8).

    Returns:
        results: dict mapping task name (no 'lb/' prefix) -> float score
    """
    import numpy as np

    # Set model to eval mode before running inference
    model.eval()

    if held_out_eval:
        # Pre-populate task_sampler with held-out indices (last 1-train_frac of each task)
        # then call evaluate with resample_requests=False so our indices are used as-is.
        held_out_idxs = {}
        for task_n, n_total in task_sampler.num_prompts_per_lb_task.items():
            start = int(n_total * train_frac)
            held_out_idxs[task_n] = np.arange(start, n_total)
            print(f"  held-out eval: {task_n} indices {start}–{n_total-1} ({n_total - start} examples)")
        task_sampler.latest_sampled_idxs_per_lb_task = held_out_idxs
        task_sampler.latest_lb_tasks_names = list(held_out_idxs.keys())
        score_dicts = task_sampler.evaluate(
            lm=evaluator,
            train=False,
            evolved_model=False,
            resample_requests=False,
        )
    else:
        # Run full evaluation over all tasks in the task_sampler's test split.
        score_dicts = task_sampler.evaluate(
            lm=evaluator,
            train=False,
            evolved_model=False,
        )

    # score_dicts is a list of dicts (one per pop_rep); we use index 0 (pop_size=1 for eval).
    # Keys are 'lb/{task}' e.g. 'lb/qasper'.
    raw = score_dicts[0] if len(score_dicts) > 0 else {}

    results = {}
    for task in EVAL_TASKS:
        score = raw.get(f'lb/{task}', None)
        results[task] = score
        print(f"  {task}: {score}")

    # Save per-task score files to artifact_dir/eval_outputs/
    eval_dir = os.path.join(artifact_dir, 'eval_outputs')
    os.makedirs(eval_dir, exist_ok=True)
    for task, score in results.items():
        score_file = os.path.join(eval_dir, f'{task}_score.txt')
        with open(score_file, 'w') as f:
            f.write(str(score) if score is not None else 'None')

    return results


def log_results(results: dict, method: str, seed: int, artifact_dir: str):
    """Log results to wandb (EVAL_WANDB_GROUP) and save eval_outputs/ to artifact_dir."""
    for task, score in results.items():
        if score is not None:
            wandb.log({f'eval/{task}': score})

    # Log bar chart — each task as a row so wandb renders bars not lines
    if wandb.run is not None:
        table = wandb.Table(columns=["task", "score"])
        for task, score in results.items():
            if score is not None:
                table.add_data(task, score)
        wandb.log({"eval/scores_bar": wandb.plot.bar(
            table, "task", "score", title="LongBench Eval Scores")})

    # Save eval_outputs to artifact_dir
    eval_dir = os.path.join(artifact_dir, 'eval_outputs')
    os.makedirs(eval_dir, exist_ok=True)
    for task, score in results.items():
        score_file = os.path.join(eval_dir, f'{task}_score.txt')
        with open(score_file, 'w') as f:
            f.write(str(score) if score is not None else 'None')


def main():
    args = parse_args()

    artifact_dir = args.artifact_dir or os.path.join('results', args.method, str(args.seed))
    os.makedirs(artifact_dir, exist_ok=True)

    print(f"=== FAIR-02 Eval | method={args.method} | seed={args.seed} ===")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Tasks: {EVAL_TASKS}")
    print(f"Wandb group: {EVAL_WANDB_GROUP}")

    if not args.no_wandb:
        # Build descriptive run name
        is_recency = args.run_config and 'recency' in args.run_config
        if args.method == 'base':
            # llama-instruct-{cache_size}-recency  or  llama-instruct-full-cache
            if is_recency and args.cache_size:
                eval_run_name = f'llama-instruct-{args.cache_size}-recency'
            else:
                eval_run_name = 'llama-instruct-full-cache'
        else:
            # eval-{method}[-recency_baseline][-cs{N}]-seed{seed}
            run_name_parts = [f'eval-{args.method}']
            if is_recency:
                run_name_parts.append('recency_baseline')
            if args.cache_size:
                run_name_parts.append(f'cs{args.cache_size}')
            if args.held_out_eval:
                run_name_parts.append('heldout')
            run_name_parts.append(f'seed{args.seed}')
            eval_run_name = '-'.join(run_name_parts)

        wandb_group = args.wandb_group or EVAL_WANDB_GROUP
        wandb.init(
            project=EVAL_WANDB_PROJECT,
            group=wandb_group,
            name=eval_run_name,
            config={'method': args.method, 'seed': args.seed, 'ckpt': args.ckpt,
                    'run_config': args.run_config, 'cache_size': args.cache_size},
        )

    # Load model, tokenizer, evaluator, and task_sampler from checkpoint
    model, tokenizer, evaluator, task_sampler = load_model_from_ckpt(
        ckpt_path=args.ckpt,
        device=args.device,
        method=args.method,
        artifact_dir=artifact_dir,
        run_config_override=args.run_config,
        cache_size_override=args.cache_size,
    )

    # Run evaluation on all EVAL_TASKS
    results = run_eval(model, tokenizer, evaluator, task_sampler, artifact_dir,
                       held_out_eval=args.held_out_eval, train_frac=args.train_frac)

    # Log to wandb and write score files
    log_results(results, args.method, args.seed, artifact_dir)

    print(f"\n=== Eval complete | method={args.method} seed={args.seed} ===")
    for task, score in results.items():
        print(f"  {task}: {score}")

    if not args.no_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
