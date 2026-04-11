"""Evaluate plain LLaMA 3.2-1B-Instruct (no NAMM, no eviction) on NAMM splits.

Uses the exact same dataset filtering and train/val/test splitting as NAMM
training so results are directly comparable.

Usage:
    env CUDA_VISIBLE_DEVICES=0 python scripts/eval_plain_llama.py \
        --run_config namm_bam_i1_llama32_1b_5t \
        --filter_by_length 8192

    # Evaluate only on val and test splits:
    env CUDA_VISIBLE_DEVICES=0 python scripts/eval_plain_llama.py \
        --splits val test
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hydra import compose, initialize
from namm.run_utils import make_task_sampler
from namm.evaluation.evaluator import MemoryHFEvaluator
from es_finetuning.device import get_device
import hydra


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate plain LLaMA (no NAMM) on NAMM training splits")
    parser.add_argument("--run_config", type=str,
                        default="namm_bam_i1_llama32_1b_5t",
                        help="Hydra run config name (must match NAMM training)")
    parser.add_argument("--filter_by_length", type=int, default=8192,
                        help="Max conditioning length for the evaluator "
                             "(set large to avoid KV truncation)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/plain_baseline_5t",
                        help="Parent directory. Each run creates a unique "
                             "subfolder inside it so results are never "
                             "overwritten.")
    parser.add_argument("--run_label", type=str, default=None,
                        help="Optional short label appended to the per-run "
                             "subfolder name (e.g. 'ext', 'rerun'). The "
                             "subfolder is always {label_}{timestamp}.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test", "extended_test"],
                        help="Which splits to evaluate. 'extended_test' = "
                             "test ∪ prompts in (max_conditioning_length, "
                             "extended_max_conditioning_length] tokens.")
    parser.add_argument("--extended_max_conditioning_length", type=int,
                        default=8192,
                        help="Upper token bound for the extended_test split "
                             "(default: 8192). Only used if 'extended_test' "
                             "is in --splits.")
    parser.add_argument("--task_config", type=str, default=None,
                        help="Override task config (e.g. rh_ood_eval_3t for OOD eval)")
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()
    print(f"Device: {device}")

    # ── Load Hydra config (same as NAMM training) ───────────────────────────
    overrides = [
        f"run@_global_={args.run_config}",
        f"filter_by_length={args.filter_by_length}",
        "wandb_log=false",
    ]
    if args.task_config is not None:
        overrides.append(f"task@_global_={args.task_config}")
    with initialize(version_base=None, config_path="../config",
                    job_name="eval_plain_llama"):
        cfg = compose(config_name="config", overrides=overrides)

    # ── Load plain LLaMA (NO NAMM wrapper) ──────────────────────────────────
    print("Loading plain LLaMA model...")
    with torch.no_grad():
        pretrained_llm = hydra.utils.call(cfg.pretrained_llm, _convert_="object")
        tokenizer = hydra.utils.call(cfg.tokenizer)
    pretrained_llm = pretrained_llm.to(device)
    pretrained_llm.eval()
    print(f"Model loaded: {pretrained_llm.config._name_or_path}")

    # ── Wrap in MemoryHFEvaluator (no eviction) ─────────────────────────────
    # Plain LLaMA has no memory_policy attribute, so is_memory_model=False
    # and no KV eviction will happen.  Set max_memory_length large so the
    # evaluator never truncates the KV cache.
    evaluator = MemoryHFEvaluator(
        model=pretrained_llm,
        tokenizer=tokenizer,
        eval_max_batch_size=args.batch_size,
        batch_size=args.batch_size,
        max_conditioning_length=args.filter_by_length,
        max_memory_length=args.filter_by_length,
        max_gen_tokens=cfg.get("max_gen_tokens", 512),
        add_bos_token=cfg.get("add_bos_token", True),
        device=device,
    )
    assert not evaluator.is_memory_model, (
        "Plain LLaMA should NOT be detected as a memory model")
    print("MemoryHFEvaluator created (is_memory_model=False, no eviction)")

    # ── Build task sampler with EXACT same filtering as NAMM training ────────
    print("Creating task sampler...")
    task_sampler = make_task_sampler(cfg=cfg)
    # Capture per-prompt model generations so post-hoc analyses don't require
    # re-running the model.
    task_sampler.store_gen_outputs = True

    # 1) Filter answers by token count (same as run_namm.py)
    max_answer_tok = cfg.get('max_answer_tokens', cfg.get('max_new_tokens', 64))
    task_sampler.filter_answers_by_token_count(tokenizer, max_answer_tok)
    print(f"Filtered answers by token count (max_answer_tokens={max_answer_tok})")

    # 2) Apply 3-way split (same fractions and length bounds as NAMM training)
    train_frac = cfg.get('train_frac', 0.7)
    val_frac = cfg.get('val_frac', 0.15)
    max_cond = cfg.get('split_max_conditioning_length',
                       cfg.get('max_conditioning_length', 6500))
    min_cond = cfg.get('min_conditioning_length', None)
    ext_max_cond = (args.extended_max_conditioning_length
                    if "extended_test" in args.splits else None)
    if ext_max_cond is not None and args.filter_by_length < ext_max_cond:
        print(f"WARNING: --filter_by_length={args.filter_by_length} < "
              f"--extended_max_conditioning_length={ext_max_cond}; "
              f"prompts above ~{args.filter_by_length/1.3:.0f} words were "
              f"already dropped at construction. Pass --filter_by_length "
              f"{ext_max_cond} (or higher) to keep them.")
    task_sampler.apply_train_val_test_split(
        train_frac=train_frac,
        val_frac=val_frac,
        max_conditioning_length=max_cond,
        min_conditioning_length=min_cond,
        tokenizer=tokenizer,
        extended_max_conditioning_length=ext_max_cond,
    )
    print(f"Applied 3-way split: train_frac={train_frac}, val_frac={val_frac}, "
          f"max_cond={max_cond}, min_cond={min_cond}")
    if ext_max_cond is not None:
        print(f"  extended_max_conditioning_length={ext_max_cond}")

    # Show per-task sample counts
    for task_n, n in task_sampler.num_prompts_per_lb_task.items():
        print(f"  Task: {task_n}, total samples: {n}")

    # ── Evaluate on requested splits ─────────────────────────────────────────
    all_results = {}
    generations_per_split = {}
    for split in args.splits:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on split: {split}")
        print('=' * 60)
        with torch.no_grad():
            score_dicts = task_sampler.evaluate(
                lm=evaluator,
                train=(split == 'train'),
                evolved_model=False,
                pop_reps=1,
                resample_requests=True,
                sampled_requests_per_task=None,
                split=split,
                performance_per_request=True,
            )
        scores = score_dicts[0]
        per_prompt_f1 = scores.pop('performance_per_request', {})
        scores['per_prompt_f1'] = {
            task: {int(k): float(v) for k, v in task_dict.items()}
            for task, task_dict in per_prompt_f1.items()
        }
        # micro_mean_f1: prompt-count-weighted mean — comparable to the
        # val_lb_avg_f1 reported by LoRA training.
        all_prompt_scores = []
        for task_dict in per_prompt_f1.values():
            all_prompt_scores.extend(task_dict.values())
        if all_prompt_scores:
            scores['micro_mean_f1'] = float(np.mean(all_prompt_scores)) * 100.0
            scores['n_prompts_total'] = int(len(all_prompt_scores))
            scores['n_prompts_per_task'] = {
                task: len(d) for task, d in per_prompt_f1.items()}
        all_results[split] = scores

        # Snapshot generations for this split
        gens = task_sampler.last_gen_outputs or {}
        generations_per_split[split] = {
            task: [
                {**d, 'prompt_idx': int(d.get('prompt_idx', -1))}
                for d in dicts
            ]
            for task, dicts in gens.items()
        }

        # Print per-task results
        print(f"\nResults ({split}):")
        print("-" * 40)
        task_scores = []
        for k, v in sorted(scores.items()):
            if k in ('per_prompt_f1', 'n_prompts_total',
                     'n_prompts_per_task', 'micro_mean_f1'):
                continue
            if not isinstance(v, (int, float)):
                continue
            print(f"  {k}: {v:.4f}")
            if k.startswith('lb/'):
                task_scores.append(v)
        if task_scores:
            macro_f1 = np.mean(task_scores)
            print(f"  --- macro mean F1: {macro_f1:.4f}")
        if 'micro_mean_f1' in scores:
            print(f"  --- micro mean F1: {scores['micro_mean_f1']:.4f} "
                  f"(n={scores.get('n_prompts_total', '?')})")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY: Plain LLaMA baseline (no NAMM, no eviction)")
    print(f"  Config: {args.run_config}")
    print(f"  filter_by_length: {args.filter_by_length}")
    print('=' * 60)

    # Collect all task keys across splits (skip non-numeric metadata fields)
    _SKIP_METADATA = {'per_prompt_f1', 'n_prompts_total',
                      'n_prompts_per_task', 'micro_mean_f1'}
    all_task_keys = sorted(set(
        k for split_scores in all_results.values() for k in split_scores
        if k not in _SKIP_METADATA))

    # Header
    header = f"{'Task':<30}" + "".join(f"{s:>10}" for s in args.splits)
    print(header)
    print("-" * len(header))

    for task_key in all_task_keys:
        row = f"{task_key:<30}"
        for split in args.splits:
            val = all_results.get(split, {}).get(task_key, float('nan'))
            row += f"{val:>10.2f}"
        print(row)

    # Mean row
    row = f"{'MEAN':<30}"
    for split in args.splits:
        vals = [v for k, v in all_results.get(split, {}).items()
                if k not in _SKIP_METADATA and isinstance(v, (int, float))
                and k.startswith('lb/')]
        mean_val = np.mean(vals) if vals else float('nan')
        row += f"{mean_val:>10.2f}"
    print("-" * len(header))
    print(row)

    # ── Save results ─────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = (f"{args.run_label}_{timestamp}"
                 if args.run_label else timestamp)
    output_dir = os.path.join(REPO_ROOT, args.output_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Run output dir: {output_dir}")

    results_payload = {
        "type": "plain_llama_baseline",
        "timestamp": timestamp,
        "config": {
            "run_config": args.run_config,
            "filter_by_length": args.filter_by_length,
            "batch_size": args.batch_size,
            "splits": args.splits,
            "max_answer_tokens": int(max_answer_tok),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "max_conditioning_length": int(max_cond) if max_cond else None,
            "min_conditioning_length": int(min_cond) if min_cond else None,
            "extended_max_conditioning_length": (
                int(ext_max_cond) if ext_max_cond else None),
        },
        "results": {
            split: {
                k: (v if k in ('per_prompt_f1', 'n_prompts_per_task')
                    else (int(v) if k == 'n_prompts_total' else float(v)))
                for k, v in scores.items()
            }
            for split, scores in all_results.items()
        },
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    gens_path = os.path.join(output_dir, "generations.json")
    with open(gens_path, "w") as f:
        json.dump(generations_per_split, f, indent=2)
    print(f"Generations saved: {gens_path}")


if __name__ == "__main__":
    main()
