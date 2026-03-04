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

Do NOT add per-condition branching to this file. If a condition needs
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

VALID_METHODS = ['m1', 'm2', 'm3', 'm4_frozen', 'm4_iterative']


def parse_args():
    p = argparse.ArgumentParser(description='Shared eval for all conditions (FAIR-02)')
    p.add_argument('--ckpt', required=True, help='Path to checkpoint file (ckpt.pt)')
    p.add_argument('--method', required=True, choices=VALID_METHODS,
                   help='Condition name: m1 | m2 | m3 | m4_frozen | m4_iterative')
    p.add_argument('--seed', type=int, default=1337, help='Random seed used for training run')
    p.add_argument('--device', default='cuda', help='Device to use for eval')
    p.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    p.add_argument('--artifact-dir', default=None,
                   help='Results artifact dir (default: results/{method}/{seed}/)')
    return p.parse_args()


def load_model_from_ckpt(ckpt_path: str, device: str):
    """
    Load model from checkpoint. Handles both lora_grad and namm_es checkpoint types.

    Phase 3 skeleton — full implementation in Phase 4.
    # TODO (Phase 4): instantiate WrappedLlamaForCausalLM, apply LoRA if lora_grad,
    #   load weights via _load_ckpt, instantiate MemoryHFEvaluator for eval loop.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer_type = ckpt.get('trainer_type', 'namm_es')
    print(f"Checkpoint trainer_type: {trainer_type}")
    raise NotImplementedError("load_model_from_ckpt: full implementation in Phase 4")


def run_eval(model, tokenizer, evaluator, artifact_dir: str):
    """
    Run LongBench evaluation on all EVAL_TASKS using shared MemoryHFEvaluator.

    No per-condition branching here — same evaluator, same tasks, same metrics.

    Phase 3 skeleton — full implementation in Phase 4.
    # TODO (Phase 4): replace placeholder loop with real MemoryHFEvaluator calls:
    #   from memory_evaluator import MemoryHFEvaluator
    #   for task, metric in zip(EVAL_TASKS, EVAL_METRICS):
    #       score = evaluator.evaluate(task)
    #       results[task] = score
    """
    results = {}
    for task, metric in zip(EVAL_TASKS, EVAL_METRICS):
        print(f"Evaluating: {task} (metric: {metric})")
        results[task] = None   # placeholder — Phase 4 fills this in
    return results


def log_results(results: dict, method: str, seed: int, artifact_dir: str):
    """Log results to wandb (EVAL_WANDB_GROUP) and save eval_outputs/ to artifact_dir."""
    for task, score in results.items():
        if score is not None:
            wandb.log({f'eval/{task}': score})
    # Save eval_outputs to artifact_dir
    eval_dir = os.path.join(artifact_dir, 'eval_outputs')
    os.makedirs(eval_dir, exist_ok=True)
    # TODO (Phase 4): write per-task score files


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

    # TODO (Phase 4): implement load_model_from_ckpt and run_eval
    print("run_eval.py skeleton ready (Phase 3). Full implementation in Phase 4.")


if __name__ == '__main__':
    main()
