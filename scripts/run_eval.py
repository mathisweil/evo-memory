"""Evaluate an ES fine-tuned checkpoint on the full Qasper validation set.

Usage:
    # Evaluate ES fine-tuned model:
    python run_eval.py \
        --es_checkpoint experiments/es_namm_runs/.../checkpoints/es_checkpoint_final.pt \
        --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt

    # Baseline (no ES fine-tuning, just NAMM + base LLM):
    python run_eval.py \
        --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt
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
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device
from experiment_utils import load_config_defaults


class Tee:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def get_output_dir(args):
    """Derive output directory from args."""
    if args.output_dir:
        return args.output_dir
    if args.es_checkpoint:
        # .../checkpoints/es_checkpoint_final.pt -> .../ (experiment dir)
        ckpt_dir = os.path.dirname(os.path.abspath(args.es_checkpoint))
        if os.path.basename(ckpt_dir) == "checkpoints":
            return os.path.dirname(ckpt_dir)
        return ckpt_dir
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ES fine-tuned model")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (see scripts/configs/eval_default.yaml)")
    parser.add_argument("--es_checkpoint", type=str, default=None,
                        help="Path to ES fine-tuned checkpoint (omit for baseline)")
    parser.add_argument("--namm_checkpoint", type=str, default=None,
                        help="Path to pre-trained NAMM scoring network checkpoint")
    parser.add_argument("--run_config", type=str,
                        default="namm_bam_i1_llama32_1b")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Inference batch size (default: use config value)")
    parser.add_argument("--filter_by_length", type=int, default=None,
                        help="Drop samples longer than this (approx, word-based). None = mid-crop instead")
    parser.add_argument("--filter_by_tokens", type=int, default=None,
                        help="Drop samples exceeding this many tokens (exact, uses tokenizer)")
    parser.add_argument("--filter_answers_by_tokens", type=int, default=64,
                        help="Drop samples whose shortest answer exceeds this many tokens")
    parser.add_argument("--cache_size", type=int, default=None,
                        help="Override cache size for NAMM eviction")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save eval log. Auto-derived from --es_checkpoint if not set")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (0 = greedy)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Samples per question (averaged for final score)")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Train/test split fraction (must match training)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed for deterministic train/test split")
    load_config_defaults(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve "latest" NAMM checkpoint from GCS
    if args.namm_checkpoint == "latest":
        from es_finetuning.gcs import GCSClient
        _gcs = GCSClient()
        cache_dir = os.path.join(REPO_ROOT, "exp_local", "pretrained")
        args.namm_checkpoint = _gcs.download_latest_pretrained(cache_dir)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()

    # Set up logging to experiment directory
    output_dir = get_output_dir(args)
    tee = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        label = "baseline" if not args.es_checkpoint else "es"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(output_dir, f"eval_{label}_{timestamp}.log")
        tee = Tee(log_path)
        sys.stdout = tee
        print(f"Logging to: {log_path}")

    # Load config
    overrides = [
        f"run@_global_={args.run_config}",
        "wandb_log=false",
        "wandb_project=Experiments",
    ]
    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")
    if args.filter_by_length is not None:
        overrides.append(f"filter_by_length={args.filter_by_length}")
    if args.cache_size is not None:
        overrides.append(f"cache_size={args.cache_size}")
        overrides.append(f"max_memory_length={args.cache_size}")
    with initialize(version_base=None, config_path="../config",
                    job_name="es_eval"):
        cfg = compose(config_name="config", overrides=overrides)

    # Build model
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)
    memory_model.to(device)
    memory_evaluator.device = device

    # Load NAMM weights
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
        print(f"  NAMM loaded ({best_member.shape[0]} params)")

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    # Load ES fine-tuned base LLM weights (if provided)
    if args.es_checkpoint:
        print(f"Loading ES checkpoint: {args.es_checkpoint}")
        es_state = torch.load(args.es_checkpoint, map_location="cpu",
                              weights_only=False)
        is_delta = es_state.pop("__format__", None) == "delta"
        model_params = dict(memory_model.named_parameters())
        loaded = 0
        for name, val in es_state.items():
            if name in model_params:
                param_dev = model_params[name].device
                if is_delta:
                    model_params[name].data.add_(val.to(param_dev))
                else:
                    model_params[name].data.copy_(val.to(param_dev))
                loaded += 1
        print(f"  Loaded {loaded}/{len(es_state)} ES-tuned parameters"
              f" ({'delta' if is_delta else 'absolute'} format)")
    else:
        print("No ES checkpoint -- evaluating base LLM weights (baseline)")

    # Create task sampler and evaluate on full val set
    print("Creating task sampler...")
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=args.train_split, split_seed=args.split_seed)

    # Exact token-based filtering
    if args.filter_by_tokens is not None:
        task_sampler.filter_by_token_count(
            memory_evaluator.tokenizer, args.filter_by_tokens)

    # Filter answers by token count (defaults to per-task max_gen_tokens)
    task_sampler.filter_answers_by_token_count(
        memory_evaluator.tokenizer, args.filter_answers_by_tokens)

    # Show val set sizes
    for task_n, n in task_sampler.num_prompts_per_lb_task.items():
        print(f"  Task: {task_n}, total samples: {n}")

    # Generation kwargs
    gen_kwargs = {}
    if args.temperature > 0:
        gen_kwargs = {"temperature": args.temperature, "do_sample": True}
        print(f"Temperature: {args.temperature}, num_samples: {args.num_samples}")

    print("\nEvaluating on FULL validation set...")
    all_score_dicts = []
    for sample_i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"  Sample {sample_i + 1}/{args.num_samples}...")
        with torch.no_grad():
            score_dicts = task_sampler.evaluate(
                lm=memory_evaluator,
                train=False,
                evolved_model=False,
                pop_reps=1,
                resample_requests=True,
                sampled_requests_per_task=None,
                model_kwargs=gen_kwargs,
            )
        all_score_dicts.append(score_dicts[0])

    # Average across samples
    avg_scores = {}
    for k in all_score_dicts[0]:
        values = [d[k] for d in all_score_dicts]
        avg_scores[k] = sum(values) / len(values)

    print("\n" + "=" * 50)
    print("RESULTS (full validation set)")
    if args.num_samples > 1:
        print(f"  (averaged over {args.num_samples} samples, temperature={args.temperature})")
    print("=" * 50)
    for k, v in sorted(avg_scores.items()):
        print(f"  {k}: {v:.4f}")
    qasper = avg_scores.get("lb/qasper", 0.0)
    print(f"\n  Qasper F1: {qasper:.2f} (0-100 scale) = {qasper/100:.4f}")
    if args.num_samples > 1:
        qasper_values = [d.get("lb/qasper", 0.0) for d in all_score_dicts]
        std = (sum((v - qasper) ** 2 for v in qasper_values) / len(qasper_values)) ** 0.5
        print(f"  Qasper F1 std: {std:.2f} (over {args.num_samples} samples)")

    # Save results.json for generate_report.py
    if output_dir:
        eval_results = {
            "type": "eval",
            "config": {
                "es_checkpoint": args.es_checkpoint,
                "namm_checkpoint": args.namm_checkpoint,
                "cache_size": args.cache_size,
                "run_config": args.run_config,
                "filter_by_length": args.filter_by_length,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
            },
            "scores": dict(avg_scores),
        }
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"  Results saved: {results_path}")

    if tee:
        tee.close()


if __name__ == "__main__":
    main()
