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
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (REPO_ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra import compose, initialize
from run_namm import make_eval_model, make_task_sampler


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
    parser.add_argument("--es_checkpoint", type=str, default=None,
                        help="Path to ES fine-tuned checkpoint (omit for baseline)")
    parser.add_argument("--namm_checkpoint", type=str, default=None,
                        help="Path to pre-trained NAMM scoring network checkpoint")
    parser.add_argument("--run_config", type=str,
                        default="namm_bam_i1_llama32_1b")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Inference batch size (default: use config value)")
    parser.add_argument("--filter_by_length", type=int, default=None,
                        help="Drop samples longer than this (tokens). None = mid-crop instead")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save eval log. Auto-derived from --es_checkpoint if not set")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        "wandb_project=es_eval",
    ]
    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")
    if args.filter_by_length is not None:
        overrides.append(f"filter_by_length={args.filter_by_length}")
    with initialize(version_base=None, config_path="../cfgs",
                    job_name="es_eval"):
        cfg = compose(config_name="config", overrides=overrides)

    # Build model
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)
    memory_model.cuda()

    # Load NAMM weights
    if args.namm_checkpoint:
        print(f"Loading NAMM checkpoint: {args.namm_checkpoint}")
        ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                          weights_only=False)
        evo_state = ckpt['evolution_state']
        best_member = evo_state['best_member']
        params = best_member.unsqueeze(0).to('cuda')
        memory_model.set_memory_params(params)

        buffers_prefix = 'stored_buffers_to_save.'
        buffers_dict = {
            k[len(buffers_prefix):]: v.to('cuda')
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
        model_params = dict(memory_model.named_parameters())
        loaded = 0
        for name, val in es_state.items():
            if name in model_params:
                model_params[name].data.copy_(val.to(model_params[name].device))
                loaded += 1
        print(f"  Loaded {loaded}/{len(es_state)} ES-tuned parameters")
    else:
        print("No ES checkpoint -- evaluating base LLM weights (baseline)")

    # Create task sampler and evaluate on full val set
    print("Creating task sampler...")
    task_sampler = make_task_sampler(cfg=cfg)

    # Show val set sizes
    for task_n, n in task_sampler.num_prompts_per_lb_task.items():
        print(f"  Task: {task_n}, total samples: {n}")

    print("\nEvaluating on FULL validation set...")
    with torch.no_grad():
        score_dicts = task_sampler.evaluate(
            lm=memory_evaluator,
            train=False,
            evolved_model=False,
            pop_reps=1,
            resample_requests=True,
            sampled_requests_per_task=None,
        )

    print("\n" + "=" * 50)
    print("RESULTS (full validation set)")
    print("=" * 50)
    for k, v in sorted(score_dicts[0].items()):
        print(f"  {k}: {v:.4f}")
    qasper = score_dicts[0].get("lb/qasper", 0.0)
    print(f"\n  Qasper F1: {qasper:.2f} (0-100 scale) = {qasper/100:.4f}")

    if tee:
        tee.close()


if __name__ == "__main__":
    main()
