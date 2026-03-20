"""ES fine-tuning of Llama 3.2 1B base weights with NAMM on Qasper.

Uses evo-memory's existing evaluation infrastructure (TaskSampler,
MemoryHFEvaluator) as the reward signal, and the es_finetuning package
for the ES optimizer loop.

Experiments are organized as:
    experiments/experiment_N/{es_namm,es_only}/run_name/
        config.json, results.json, examples.json, checkpoints/

Usage:
    # New experiment, auto-incremented:
    python scripts/run_es.py --run_name cache1024_i50 \
        --namm_checkpoint exp_local/pretrained/namm.pt \
        --cache_size 1024 --num_iterations 50

    # Add a run to an existing active experiment:
    python scripts/run_es.py --experiment 3 --run_name cache2048_i50 ...

    # With GCS checkpointing and spot preemption handling:
    python scripts/run_es.py --gcs --checkpoint_every 10 --run_name cache1024 ...

    # Quick smoke test:
    python scripts/run_es.py --run_name test \
        --num_iterations 2 --population_size 2 --mini_batch_size 2
"""

import argparse
import atexit

import yaml
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
from es_finetuning import ESConfig, ESTrainer
from es_finetuning.device import get_device
from namm.evaluation.longbench import get_all_scores
from experiment_utils import (
    get_or_create_experiment,
    get_or_create_experiment_gcs,
    claim_run_gcs,
    load_hydra_config,
    get_base_llm_param_names,
    EXPERIMENTS_DIR,
)

def _sync_xla_cache_to_gcs():
    """Upload local XLA compilation cache to GCS for persistence across VMs."""
    cache_dir = os.environ.get("XLA_PERSISTENT_CACHE_PATH", "")
    bucket = os.environ.get("GCS_BUCKET", "")
    if not cache_dir or not bucket or not os.path.isdir(cache_dir):
        return
    import subprocess
    print(f"Syncing XLA cache to gs://{bucket}/xla_cache ...")
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", cache_dir, f"gs://{bucket}/xla_cache"],
        check=False,
    )


atexit.register(_sync_xla_cache_to_gcs)


# ── Evaluation function factories ───────────────────────────────────────

def make_resample_fn(task_sampler, mini_batch_size, train=True):
    def resample_fn():
        task_sampler.resample_requests(
            train=train,
            sampled_requests_per_task=mini_batch_size,
        )
    return resample_fn


def _gen_model_kwargs(temperature):
    """Build model_kwargs dict for generation temperature."""
    if temperature > 0:
        return {"temperature": temperature, "do_sample": True}
    return {}


def make_evaluate_fn(task_sampler, evaluator, mini_batch_size, train=True,
                     temperature=0.0, num_samples=1):
    gen_kwargs = _gen_model_kwargs(temperature)

    def evaluate_fn(model):
        scores = []
        for _ in range(num_samples):
            score_dicts = task_sampler.evaluate(
                lm=evaluator,
                train=train,
                evolved_model=False,
                pop_reps=1,
                resample_requests=False,
                sampled_requests_per_task=mini_batch_size,
                model_kwargs=gen_kwargs,
            )
            scores.append(score_dicts[0].get("lb/qasper", 0.0) / 100.0)
        return sum(scores) / len(scores)
    return evaluate_fn


def make_full_eval_fn(task_sampler, evaluator,
                      temperature=0.0, num_samples=1):
    gen_kwargs = _gen_model_kwargs(temperature)

    def full_eval_fn(model):
        all_score_dicts = []
        for _ in range(num_samples):
            score_dicts = task_sampler.evaluate(
                lm=evaluator,
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
            avg_scores[k] = sum(d[k] for d in all_score_dicts) / len(all_score_dicts)
        if task_sampler._test_idxs_per_task is not None:
            num_data_samples = sum(
                len(v) for v in task_sampler._test_idxs_per_task.values())
        else:
            num_data_samples = sum(task_sampler.num_prompts_per_lb_task.values())
        return {"scores": avg_scores, "num_samples": num_data_samples}
    return full_eval_fn


def make_examples_fn(task_sampler, evaluator, n_examples=10):
    """Create a function that captures Q/A examples with model predictions."""
    def examples_fn(model):
        task_name = task_sampler.lb_test_tasks[0]  # e.g. "lb/qasper"
        task_short = task_name.split("/")[1]  # e.g. "qasper"
        all_prompts = task_sampler.lb_prompts_per_task[task_name]
        all_jsons = task_sampler.lb_jsons_per_task[task_name]

        n = min(n_examples, len(all_prompts))
        idxs = np.random.choice(len(all_prompts), size=n, replace=False)

        prompts = [all_prompts[i] for i in idxs]
        jsons_subset = [all_jsons[i] for i in idxs]

        max_gen_tokens = task_sampler.lb_task2maxlen[task_name]
        stop_gen = task_sampler.lb_taskstopgen[task_name]

        predictions = evaluator.evaluate_lb(
            dataset_samples=prompts,
            max_gen_tokens=max_gen_tokens,
            stop_gen=stop_gen,
            pop_reps=1,
        )

        # Per-sample F1 scores
        answers_list = [j["answers"] for j in jsons_subset]
        all_classes = jsons_subset[0].get("all_classes")
        per_sample_scores = get_all_scores(
            task_short, predictions, answers_list, all_classes)

        examples = []
        for i, (json_obj, pred, score) in enumerate(
                zip(jsons_subset, predictions, per_sample_scores)):
            context = json_obj.get("context", "")
            context_snippet = context[:500] + "..." if len(context) > 500 else context
            examples.append({
                "question": json_obj.get("input", ""),
                "context_snippet": context_snippet,
                "gold_answers": json_obj["answers"],
                "prediction": pred,
                "f1_score": round(score, 4),
            })

        return examples
    return examples_fn


# ── CLI ─────────────────────────────────────────────────────────────────

def _load_config_defaults(parser):
    """Load defaults from a YAML config file specified by --config."""
    # Pre-parse just --config without erroring on other args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f)
        # Set defaults from config file; CLI args will override
        parser.set_defaults(**{k: v for k, v in cfg.items()
                               if v is not None})


def parse_args():
    parser = argparse.ArgumentParser(
        description="ES fine-tuning of base LLM weights with NAMM evaluation")

    # Config file (defaults that CLI args override)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (see scripts/es_default.yaml)")

    # Experiment hierarchy
    parser.add_argument("--run_name", type=str, default=None,
                        help="(required) Name for this run (e.g. cache1024_i50)")
    parser.add_argument("--experiment", type=int, default=None,
                        help="Experiment ID (default: most recent active, or create new)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["es_namm", "es_only", "es_recency"],
                        help="Method type (default: auto-detect from --namm_checkpoint)")

    # ES hyperparameters
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--population_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--noise_mode", type=str, default="correlated",
                        choices=["correlated", "iid"])
    parser.add_argument("--initial_seed", type=int, default=33)
    parser.add_argument("--mini_batch_size", type=int, default=16)

    # NAMM config
    parser.add_argument("--namm_checkpoint", type=str, default=None)
    parser.add_argument("--run_config", type=str,
                        default="namm_bam_i1_llama32_1b")

    # Evaluator batching
    parser.add_argument("--batch_size", type=int, default=None)

    # Data filtering
    parser.add_argument("--filter_by_length", type=int, default=None)
    parser.add_argument("--filter_by_tokens", type=int, default=None,
                        help="Drop samples exceeding this many tokens (exact, uses tokenizer)")
    parser.add_argument("--filter_answers_by_tokens", type=int, default=64,
                        help="Drop samples whose shortest answer exceeds this many tokens")

    # Cache size (NAMM)
    parser.add_argument("--cache_size", type=int, default=None)

    # Resume from checkpoint
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    # Data splits
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="Fraction of data for training (rest is held-out eval)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed for deterministic train/test split")

    # Q/A examples
    parser.add_argument("--n_examples", type=int, default=10,
                        help="Number of Q/A examples to capture")

    # GCS and checkpointing
    parser.add_argument("--gcs", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable GCS experiment management and checkpointing")
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="Rolling checkpoint every M iterations, keep last 2 (0 = final only)")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Permanent save every N iterations, never deleted (0 = disabled)")

    # Generation temperature and sampling
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature for training evals (0 = greedy)")
    parser.add_argument("--eval_temperature", type=float, default=0.0,
                        help="Generation temperature for full eval (0 = greedy)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Samples per question during training (averaged for fitness)")
    parser.add_argument("--eval_num_samples", type=int, default=1,
                        help="Samples per question during full eval (averaged for score)")

    # Extra Hydra overrides
    parser.add_argument("--override", action="append", default=[])

    _load_config_defaults(parser)
    args = parser.parse_args()
    if not args.run_name:
        parser.error("--run_name is required (via CLI or config file)")
    return args


def main():
    args = parse_args()

    # Auto-detect method
    if args.method is None:
        args.method = "es_namm" if args.namm_checkpoint else "es_only"

    # Setup GCS client if requested
    gcs = None
    preemption_handler = None
    if args.gcs:
        from es_finetuning.gcs import GCSClient
        from es_finetuning.preemption import PreemptionHandler
        gcs = GCSClient()
        preemption_handler = PreemptionHandler()

    # Resolve "latest" NAMM checkpoint from GCS
    if args.namm_checkpoint == "latest":
        from es_finetuning.gcs import GCSClient
        _gcs = gcs if gcs is not None else GCSClient()
        cache_dir = os.path.join(REPO_ROOT, "exp_local", "pretrained")
        args.namm_checkpoint = _gcs.download_latest_pretrained(cache_dir)

    # Setup experiment hierarchy
    if gcs:
        experiment_name, manifest = get_or_create_experiment_gcs(
            gcs, args.experiment)
    else:
        experiment_name, manifest = get_or_create_experiment(args.experiment)

    run_dir = os.path.join(EXPERIMENTS_DIR, experiment_name,
                           args.method, args.run_name)

    # Auto-resume from GCS if checkpoint exists
    resumed_history = None
    resumed_training_state = None
    if gcs and args.resume_checkpoint is None:
        ckpt_path, training_state = gcs.download_latest_checkpoint(
            experiment_name, args.method, args.run_name, run_dir)
        if ckpt_path:
            args.resume_checkpoint = ckpt_path
            resumed_training_state = training_state
            if training_state:
                resumed_history = training_state.get("history")
            print(f"Auto-resuming from GCS checkpoint: {ckpt_path}")

    if (os.path.exists(run_dir)
            and os.path.exists(os.path.join(run_dir, "results.json"))
            and args.resume_checkpoint is None):
        print(f"ERROR: Run already exists with results: {run_dir}")
        sys.exit(1)

    os.makedirs(run_dir, exist_ok=True)

    # Claim run in GCS manifest
    if gcs:
        claim_run_gcs(gcs, experiment_name, args.method, args.run_name)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()

    print("=" * 60)
    print("ES Fine-Tuning: Llama 3.2 1B + NAMM on Qasper")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Method: {args.method}")
    print(f"Run: {args.run_name}")
    print(f"Output: {run_dir}")
    print(f"Config: sigma={args.sigma}, alpha={args.alpha}, "
          f"pop={args.population_size}, iter={args.num_iterations}")
    print(f"Noise mode: {args.noise_mode}")
    print(f"Mini-batch size: {args.mini_batch_size}")
    if args.temperature > 0 or args.eval_temperature > 0:
        print(f"Temperature: train={args.temperature}, eval={args.eval_temperature}")
        print(f"Num samples: train={args.num_samples}, eval={args.eval_num_samples}")
    if gcs:
        print(f"GCS: gs://{gcs.bucket_name}/ (checkpoint every "
              f"{args.checkpoint_every} iter)")
    print()

    # 1. Load config and model
    print("Loading config and model via Hydra...")
    hydra_overrides = []
    if args.batch_size is not None:
        hydra_overrides.append(f"batch_size={args.batch_size}")
    if args.filter_by_length is not None:
        hydra_overrides.append(f"filter_by_length={args.filter_by_length}")
    if args.cache_size is not None:
        hydra_overrides.append(f"cache_size={args.cache_size}")
        hydra_overrides.append(f"max_memory_length={args.cache_size}")
    hydra_overrides.extend(args.override)
    cfg = load_hydra_config(args.run_config, extra_overrides=hydra_overrides)

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    memory_model.to(device)

    # 2. Load NAMM checkpoint
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
            print(f"  Loaded {len(buffers_dict)} normalization buffers")

        print(f"  Loaded NAMM best_member ({best_member.shape[0]} params) "
              f"from iter {ckpt.get('iter_num', '?')}, "
              f"best_val={ckpt.get('best_val_loss', '?')}")

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    # 3. Create task sampler
    print("Creating task sampler...")
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=args.train_split, split_seed=args.split_seed)

    # 3b. Exact token-based filtering (replaces approximate word-based filter)
    if args.filter_by_tokens is not None:
        task_sampler.filter_by_token_count(
            memory_evaluator.tokenizer, args.filter_by_tokens)

    # Filter answers by token count (defaults to per-task max_gen_tokens)
    task_sampler.filter_answers_by_token_count(
        memory_evaluator.tokenizer, args.filter_answers_by_tokens)

    # 4. Auto-detect batch size
    if memory_evaluator.batch_size == "auto":
        print("Auto-detecting optimal batch size...")
        detected_bs = memory_evaluator._detect_batch_size()
        memory_evaluator.batch_size_per_gpu = detected_bs
        memory_evaluator.batch_size = detected_bs
        print(f"  Auto-detected batch size: {detected_bs}")
    else:
        print(f"  Using fixed batch size: {memory_evaluator.batch_size}")

    # 5. Get base LLM param names
    param_names = get_base_llm_param_names(memory_model)
    print(f"Base LLM parameters to optimize: {len(param_names)}")

    # 7. Build evaluation functions
    print("Building evaluation functions...")
    resample_fn = make_resample_fn(
        task_sampler, args.mini_batch_size, train=True)
    evaluate_fn = make_evaluate_fn(
        task_sampler, memory_evaluator, args.mini_batch_size, train=True,
        temperature=args.temperature, num_samples=args.num_samples)
    full_eval_fn = make_full_eval_fn(
        task_sampler, memory_evaluator,
        temperature=args.eval_temperature, num_samples=args.eval_num_samples)
    examples_fn = make_examples_fn(
        task_sampler, memory_evaluator, n_examples=args.n_examples)

    # 8. Configure and run ES
    es_config = ESConfig(
        sigma=args.sigma,
        alpha=args.alpha,
        population_size=args.population_size,
        num_iterations=args.num_iterations,
        noise_mode=args.noise_mode,
        initial_seed=args.initial_seed,
        mini_batch_size=args.mini_batch_size,
        log_dir=run_dir,
        checkpoint_every=args.checkpoint_every if args.gcs else 0,
        save_every=args.save_every,
        temperature=args.temperature,
        eval_temperature=args.eval_temperature,
        num_samples=args.num_samples,
        eval_num_samples=args.eval_num_samples,
    )

    trainer = ESTrainer(
        model=memory_model,
        param_names=param_names,
        evaluate_fn=evaluate_fn,
        config=es_config,
        pre_step_fn=resample_fn,
        full_eval_fn=full_eval_fn,
        examples_fn=examples_fn,
        metadata={
            "namm_checkpoint": args.namm_checkpoint,
            "run_config": args.run_config,
            "train_split": args.train_split,
            "split_seed": args.split_seed,
            "num_base_params": len(param_names),
            "experiment": experiment_name,
            "method": args.method,
            "run_name": args.run_name,
        },
        resume_from=args.resume_checkpoint,
        gcs_client=gcs,
        preemption_handler=preemption_handler,
        experiment_name=experiment_name,
        method=args.method,
        run_name=args.run_name,
    )

    # 8. Load ES checkpoint AFTER trainer init (so initial_params snapshot
    #    captures the true base weights, not the resumed weights)
    numpy_state_restored = False
    if args.resume_checkpoint:
        print(f"Loading ES checkpoint weights: {args.resume_checkpoint}")

        es_state = torch.load(args.resume_checkpoint, map_location="cpu",
                              weights_only=True)
        model_params = dict(memory_model.named_parameters())
        is_delta = es_state.pop("__format__", None) == "delta"
        loaded = 0
        for name, val in es_state.items():
            if name in model_params:
                val = val.to(device=model_params[name].device,
                             dtype=model_params[name].dtype)
                if is_delta:
                    model_params[name].data.add_(val)
                else:
                    model_params[name].data.copy_(val)
                loaded += 1
        fmt = "delta" if is_delta else "full (legacy)"
        print(f"  Loaded {loaded} parameter tensors ({fmt} format)")

        # Restore numpy RNG state from training state (if available).
        # Must happen AFTER weight loading but BEFORE train() which
        # would otherwise call np.random.seed() and destroy this state.
        if (resumed_training_state
                and "numpy_random_state" in resumed_training_state):
            from es_finetuning.trainer import restore_numpy_state
            restore_numpy_state(resumed_training_state["numpy_random_state"])
            numpy_state_restored = True
            print("  Restored numpy random state from training state")

    print()
    print("Starting ES training...")
    with torch.no_grad():
        trainer.train(resumed_history=resumed_history,
                      numpy_state_restored=numpy_state_restored)

    print("Done.")


if __name__ == "__main__":
    main()
