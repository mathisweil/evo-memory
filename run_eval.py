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
    return p.parse_args()


def load_model_from_ckpt(ckpt_path: str, device: str, method: str, artifact_dir: str):
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

    # 3. Select Hydra run config based on method
    if method in ('m4_frozen', 'm2', 'm3'):
        run_cfg_name = 'namm_bam_eval_llama32_1b'
    else:
        # m1, m1_sft, base, m4_iterative all use full-cache baseline (no NAMM)
        run_cfg_name = 'full_cache_baseline_llama32_1b'

    # 4. Build Hydra overrides
    hydra_overrides = [
        f'run@_global_={run_cfg_name}',
        f'wandb_log=false',       # eval script manages wandb independently
        f'eval_only=true',
    ]
    if method in ('m4_frozen', 'm2', 'm3') and namm_ckpt_path is not None:
        hydra_overrides.append(f'init_from={namm_ckpt_path}')

    # 5. Initialize Hydra config (handle singleton — can only call initialize once/process)
    GlobalHydra.instance().clear()
    cfg = initialize_cfg('cfgs', hydra_overrides=hydra_overrides)

    # 6. Build model, evaluator, task_sampler via the standard make_eval_model path
    (memory_policy, model, evaluator, evo_alg, aux_loss) = make_eval_model(cfg)
    task_sampler = make_task_sampler(
        cfg,
        store_gen_outputs=True,
        store_gen_outputs_path=os.path.join(artifact_dir, 'eval_outputs'),
    )

    # Retrieve tokenizer from evaluator (MemoryHFEvaluator stores it)
    tokenizer = evaluator.tokenizer if hasattr(evaluator, 'tokenizer') else None

    # 7. Apply LoRA adapters and load LoRA weights for lora_grad conditions
    lora_conditions = ('m1', 'm1_sft', 'm2', 'm3', 'm4_frozen', 'm4_iterative')
    # 'base' is intentionally NOT in lora_conditions — base LLaMA eval skips LoRA entirely
    if method == 'base':
        print("Base LLaMA eval: skipping LoRA adapter injection (no checkpoint needed).")
        # No LoRA, no checkpoint loading — pure base model evaluation
    elif method in lora_conditions:
        # apply_lora_adapters injects PEFT LoRA A/B matrices into the base model
        model.apply_lora_adapters(
            rank=lora_rank,
            target_modules=lora_target_modules,
        )
        # Load LoRA weights from the checkpoint
        if ckpt_path is None:
            raise ValueError(f"--ckpt is required for method '{method}' (LoRA checkpoint needed)")
        lora_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def run_eval(model, tokenizer, evaluator, task_sampler, artifact_dir: str):
    """
    Run LongBench evaluation on all EVAL_TASKS using shared MemoryHFEvaluator.

    No per-condition branching here — same evaluator, same tasks, same metrics.
    task_sampler already knows which tasks to evaluate (lb_3subset_eval config).

    Returns:
        results: dict mapping task name (no 'lb/' prefix) -> float score
    """
    # Set model to eval mode before running inference
    model.eval()

    # Run full evaluation over all tasks in the task_sampler's test split.
    # task_sampler.evaluate() calls evaluate_lb_tasks_for_pop internally,
    # which calls evaluator.evaluate_lb() for each task.
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
        wandb.init(
            project=EVAL_WANDB_PROJECT,
            group=EVAL_WANDB_GROUP,
            name=f'eval-{args.method}-seed{args.seed}',
            config={'method': args.method, 'seed': args.seed, 'ckpt': args.ckpt},
        )

    # Load model, tokenizer, evaluator, and task_sampler from checkpoint
    model, tokenizer, evaluator, task_sampler = load_model_from_ckpt(
        ckpt_path=args.ckpt,
        device=args.device,
        method=args.method,
        artifact_dir=artifact_dir,
    )

    # Run evaluation on all EVAL_TASKS
    results = run_eval(model, tokenizer, evaluator, task_sampler, artifact_dir)

    # Log to wandb and write score files
    log_results(results, args.method, args.seed, artifact_dir)

    print(f"\n=== Eval complete | method={args.method} seed={args.seed} ===")
    for task, score in results.items():
        print(f"  {task}: {score}")

    if not args.no_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
