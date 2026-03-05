"""ES fine-tuning of Llama 3.2 1B base weights with NAMM on Qasper.

Uses evo-memory's existing evaluation infrastructure (TaskSampler,
MemoryHFEvaluator) as the reward signal, and es_finetuning library
for the ES optimizer loop.

Usage:
    # From evo-memory repo root:
    python run_es_finetuning.py

    # Override defaults:
    python run_es_finetuning.py \
        --num_iterations 150 \
        --population_size 8 \
        --mini_batch_size 4 \
        --sigma 0.001 \
        --alpha 0.0005 \
        --noise_mode correlated \
        --log_dir es_runs

    # Quick smoke test (2 iterations, pop=2):
    python run_es_finetuning.py \
        --num_iterations 2 \
        --population_size 2 \
        --mini_batch_size 2
"""

import argparse
import os
import sys

import numpy as np
import torch

# Ensure evo-memory root is on the path (for namm package, utils, etc.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hydra import compose, initialize
from run_namm_training import make_eval_model, make_task_sampler
from es_finetuning import ESConfig, ESTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES fine-tuning of base LLM weights with NAMM evaluation")

    # ES hyperparameters
    parser.add_argument("--sigma", type=float, default=0.001,
                        help="Noise scale for weight perturbations")
    parser.add_argument("--alpha", type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument("--population_size", type=int, default=8,
                        help="Population members per ES iteration")
    parser.add_argument("--num_iterations", type=int, default=150,
                        help="Total ES iterations")
    parser.add_argument("--noise_mode", type=str, default="correlated",
                        choices=["correlated", "iid"],
                        help="Noise mode: correlated (shared seed) or iid (per-param)")
    parser.add_argument("--initial_seed", type=int, default=33,
                        help="Initial random seed")
    parser.add_argument("--mini_batch_size", type=int, default=4,
                        help="Qasper samples per population eval")

    # Checkpointing and logging
    parser.add_argument("--checkpoint_every", type=int, default=25,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--eval_every", type=int, default=25,
                        help="Run validation every N iterations")
    parser.add_argument("--log_dir", type=str, default="experiments/es_runs",
                        help="Directory for TensorBoard logs and checkpoints")

    # NAMM config
    parser.add_argument("--namm_checkpoint", type=str, default=None,
                        help="Path to pre-trained NAMM scoring network checkpoint")
    parser.add_argument("--run_config", type=str,
                        default="namm_bam_i1_llama32_1b",
                        help="Hydra run config name from cfgs/run/")

    # Evaluator batching
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Inference batch size for the evaluator (default: use config value)")

    # Data splits
    parser.add_argument("--train_samples", type=int, default=150,
                        help="Number of Qasper samples for training (from start)")
    parser.add_argument("--val_samples", type=int, default=50,
                        help="Number of Qasper samples for validation (from end)")

    return parser.parse_args()


def load_hydra_config(run_config, extra_overrides=None):
    """Load the full evo-memory config using Hydra compose API."""
    extra_overrides = extra_overrides or []
    config_path = os.path.join(SCRIPT_DIR, "cfgs")
    with initialize(version_base=None, config_path="cfgs",
                    job_name="es_finetuning"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"run@_global_={run_config}",
                "wandb_log=false",
                "wandb_project=es_finetuning",
            ] + extra_overrides,
        )
    return cfg


def make_resample_fn(task_sampler, mini_batch_size, train=True):
    """Create a resample_fn that draws a shared batch once per ES iteration.

    Called via ESTrainer.pre_step_fn before evaluating population members,
    so all members are compared on the same data — reducing gradient noise.
    """
    def resample_fn():
        task_sampler.resample_requests(
            train=train,
            sampled_requests_per_task=mini_batch_size,
        )
    return resample_fn


def make_evaluate_fn(task_sampler, evaluator, mini_batch_size, train=True,
                     resample=False):
    """Create the evaluate_fn closure for ESTrainer.

    Args:
        task_sampler: evo-memory TaskSampler instance.
        evaluator: evo-memory MemoryHFEvaluator instance.
        mini_batch_size: Number of Qasper samples per evaluation.
        train: If True, sample from training split; if False, from val split.
        resample: If True, resample a new batch each call. Use for validation
            (called once, not in a per-member loop). Training evaluate_fn
            should leave this False — resampling is done by pre_step_fn.

    Returns:
        Callable(model) -> float that returns a scalar reward.
    """
    def evaluate_fn(model):
        # Evaluate on the current batch (resampled by pre_step_fn for training,
        # or resampled here for validation).
        # pop_reps=1 since we're evaluating a single set of
        # (perturbed) base weights, not a NAMM population.
        # NAMM eviction still runs because the model is a WrappedLlamaForCausalLM.
        score_dicts = task_sampler.evaluate(
            lm=evaluator,
            train=train,
            evolved_model=False,
            pop_reps=1,
            resample_requests=resample,
            sampled_requests_per_task=mini_batch_size,
        )
        # score_dicts[0] is e.g. {"lb/qasper": 45.2} (0-100 scale)
        score = score_dicts[0].get("lb/qasper", 0.0) / 100.0
        return score

    return evaluate_fn


def get_base_llm_param_names(model):
    """Get parameter names for the base LLM only (not NAMM scoring network).

    Uses base_model_param_keys set by WrappedLlamaForCausalLM at init time.
    Falls back to filtering out memory_policy params if needed.
    """
    if hasattr(model, "base_model_param_keys"):
        # base_model_param_keys contains the original checkpoint keys.
        # We need to verify they match named_parameters() keys.
        model_params = dict(model.named_parameters())
        valid_keys = [k for k in model.base_model_param_keys if k in model_params]
        if valid_keys:
            return valid_keys

    # Fallback: exclude memory_policy params
    return [
        name for name, _ in model.named_parameters()
        if not name.startswith("memory_policy.")
    ]


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print("ES Fine-Tuning: Llama 3.2 1B + NAMM on Qasper")
    print("=" * 60)
    print(f"Config: sigma={args.sigma}, alpha={args.alpha}, "
          f"pop={args.population_size}, iter={args.num_iterations}")
    print(f"Noise mode: {args.noise_mode}")
    print(f"Mini-batch size: {args.mini_batch_size}")
    print()

    # 1. Load config and instantiate model via Hydra
    print("Loading config and model via Hydra...")
    hydra_overrides = []
    if args.eval_batch_size is not None:
        hydra_overrides += [
            f"batch_size={args.eval_batch_size}",
            f"eval_max_batch_size={args.eval_batch_size}",
        ]
    cfg = load_hydra_config(args.run_config, extra_overrides=hydra_overrides)

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    # Move model to GPU
    memory_model.cuda()

    # 2. Optionally load pre-trained NAMM scoring network weights
    if args.namm_checkpoint:
        print(f"Loading NAMM checkpoint: {args.namm_checkpoint}")
        ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                          weights_only=False)
        evo_state = ckpt['evolution_state']

        # Extract best_member (flat param vector for NAMM scoring network)
        best_member = evo_state['best_member']
        params = best_member.unsqueeze(0).to('cuda')
        memory_model.set_memory_params(params)

        # Load stored normalization buffers (EMA mean/var for embeddings)
        buffers_prefix = 'stored_buffers_to_save.'
        buffers_dict = {
            k[len(buffers_prefix):]: v.to('cuda')
            for k, v in evo_state.items()
            if k.startswith(buffers_prefix)
        }
        if buffers_dict:
            memory_model.load_buffers_dict(buffers_dict=buffers_dict)
            print(f"  Loaded {len(buffers_dict)} normalization buffers")

        print(f"  Loaded NAMM best_member ({best_member.shape[0]} params) "
              f"from iter {ckpt.get('iter_num', '?')}, "
              f"best_val={ckpt.get('best_val_loss', '?')}")

    # Set NAMM to use fixed params (index 0) for all evals
    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    # 3. Create task sampler
    print("Creating task sampler...")
    task_sampler = make_task_sampler(cfg=cfg)

    # 4. Get base LLM param names (not NAMM params)
    param_names = get_base_llm_param_names(memory_model)
    print(f"Base LLM parameters to optimize: {len(param_names)}")

    # 5. Build evaluate_fn closures
    #    resample_fn is called once per ES iteration (pre_step_fn) so all
    #    population members are evaluated on the same batch — this matches
    #    how NAMM's CMA-ES trainer works and reduces gradient noise.
    print("Building evaluation functions...")
    resample_fn = make_resample_fn(
        task_sampler, args.mini_batch_size, train=True)
    evaluate_fn = make_evaluate_fn(
        task_sampler, memory_evaluator, args.mini_batch_size, train=True)
    validate_fn = make_evaluate_fn(
        task_sampler, memory_evaluator, args.val_samples, train=False,
        resample=True)  # Validation runs once, not in a per-member loop

    # 6. Configure and run ES
    es_config = ESConfig(
        sigma=args.sigma,
        alpha=args.alpha,
        population_size=args.population_size,
        num_iterations=args.num_iterations,
        noise_mode=args.noise_mode,
        initial_seed=args.initial_seed,
        mini_batch_size=args.mini_batch_size,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        log_dir=args.log_dir,
    )

    trainer = ESTrainer(
        model=memory_model,
        param_names=param_names,
        evaluate_fn=evaluate_fn,
        config=es_config,
        validate_fn=validate_fn,
        pre_step_fn=resample_fn,
        metadata={
            "namm_checkpoint": args.namm_checkpoint,
            "run_config": args.run_config,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "num_base_params": len(param_names),
        },
    )

    print()
    print("Starting ES training...")
    with torch.no_grad():
        trainer.train()

    print("Done.")


if __name__ == "__main__":
    main()
