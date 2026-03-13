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
    python scripts/run_es.py --run_name test --method es_only \
        --run_config full_cache_es_llama32_1b_tpu \
        --num_iterations 2 --population_size 2 \
        --mini_batch_size 18 --batch_size 18 --no-gcs
"""

import argparse
import atexit
import json
import logging
import os
import shutil
import socket
import sys
import time
from datetime import datetime

import numpy as np
import torch

logging.getLogger("transformers.generation.stopping_criteria").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (REPO_ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydra import compose, initialize
from run_namm import make_eval_model, make_task_sampler
from es_finetuning import (
    ESConfig,
    ESTrainer,
    ExactTpuPopulationExecutor,
    SingleProcessPopulationExecutor,
)
from es_finetuning.bucketing import (
    build_bucketed_request_pools,
    iter_compile_warmup_requests,
    make_bucketed_resample_fn,
)
from es_finetuning.device import get_device
from es_finetuning.tpu_guardrails import (
    is_tpu_device,
    validate_tpu_batch_settings,
)
from utils.longbench import get_all_scores

EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
MANIFEST_PATH = os.path.join(EXPERIMENTS_DIR, "manifest.json")


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


def _maybe_enable_xla_cache_sync(args, is_tpu):
    """Enable optional XLA cache sync, validating required prerequisites."""
    if not args.sync_xla_cache:
        return
    if not args.gcs:
        print("WARNING: --sync-xla-cache requested but GCS is disabled (--no-gcs).")
        return
    if not is_tpu:
        print("WARNING: --sync-xla-cache requested but active device is not TPU.")
        return

    cache_dir = os.environ.get("XLA_PERSISTENT_CACHE_PATH", "")
    bucket = os.environ.get("GCS_BUCKET", "")

    if not cache_dir:
        print("WARNING: --sync-xla-cache requested but XLA_PERSISTENT_CACHE_PATH is unset.")
        return
    if not os.path.isdir(cache_dir):
        print(f"WARNING: --sync-xla-cache requested but cache dir does not exist: {cache_dir}")
        return
    if not bucket:
        print("WARNING: --sync-xla-cache requested but GCS_BUCKET is unset.")
        return
    if shutil.which("gsutil") is None:
        print("WARNING: --sync-xla-cache requested but 'gsutil' is not available.")
        return

    atexit.register(_sync_xla_cache_to_gcs)
    print(
        "XLA cache sync on exit enabled: "
        f"{cache_dir} -> gs://{bucket}/xla_cache"
    )


# ── Local experiment manifest management ──────────────────────────────

def _load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"experiments": {}}


def _save_manifest(manifest):
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _next_experiment_id(manifest):
    existing = [int(k.split("_")[1]) for k in manifest["experiments"]
                if k.startswith("experiment_")]
    return max(existing, default=0) + 1


def get_or_create_experiment(experiment_id=None):
    """Get an existing active experiment or create a new one.

    If no experiment_id is given, uses the most recent active experiment.
    If no active experiments exist, creates a new one.

    Returns (experiment_name, manifest).
    """
    manifest = _load_manifest()

    if experiment_id is not None:
        name = f"experiment_{experiment_id}"
        if name not in manifest["experiments"]:
            raise ValueError(f"{name} does not exist. Available: "
                             f"{list(manifest['experiments'].keys())}")
        status = manifest["experiments"][name]["status"]
        if status != "active":
            raise ValueError(f"{name} is '{status}', not 'active'. "
                             "Cannot add runs to a non-active experiment.")
        return name, manifest

    # Find most recent active experiment
    active = [(name, info) for name, info in manifest["experiments"].items()
              if info["status"] == "active"]
    if active:
        # Sort by ID (highest = most recent)
        active.sort(key=lambda x: int(x[0].split("_")[1]), reverse=True)
        name = active[0][0]
        print(f"Using active experiment: {name}")
        return name, manifest

    # No active experiments — create new one
    new_id = _next_experiment_id(manifest)
    name = f"experiment_{new_id}"
    manifest["experiments"][name] = {
        "status": "active",
        "created": datetime.now().isoformat(),
    }
    _save_manifest(manifest)
    print(f"Created new experiment: {name}")
    return name, manifest


# ── GCS experiment manifest management ────────────────────────────────

def get_or_create_experiment_gcs(gcs, experiment_id=None):
    """GCS-backed version of get_or_create_experiment."""

    if experiment_id is not None:
        name = f"experiment_{experiment_id}"

        def ensure_exists(manifest):
            if name not in manifest["experiments"]:
                raise ValueError(
                    f"{name} does not exist. Available: "
                    f"{list(manifest['experiments'].keys())}")
            status = manifest["experiments"][name]["status"]
            if status != "active":
                raise ValueError(
                    f"{name} is '{status}', not 'active'.")
            return manifest

        manifest = gcs.update_manifest(ensure_exists)
        return name, manifest

    # Try to find active experiment or create one
    manifest, generation = gcs.load_manifest()
    active = [(n, i) for n, i in manifest["experiments"].items()
              if i["status"] == "active"]
    if active:
        active.sort(key=lambda x: int(x[0].split("_")[1]), reverse=True)
        name = active[0][0]
        print(f"Using active experiment: {name}")
        return name, manifest

    # Create new one
    existing = [int(k.split("_")[1]) for k in manifest["experiments"]
                if k.startswith("experiment_")]
    new_id = max(existing, default=0) + 1
    name = f"experiment_{new_id}"

    def create(m):
        m["experiments"][name] = {
            "status": "active",
            "created": datetime.now().isoformat(),
        }
        return m

    manifest = gcs.update_manifest(create)
    print(f"Created new experiment: {name}")
    return name, manifest


def claim_run_gcs(gcs, experiment_name, method, run_name):
    """Atomically register a run as 'running' in the GCS manifest."""
    vm_id = os.environ.get("VM_ID", socket.gethostname())
    run_key = f"{method}/{run_name}"

    def updater(manifest):
        exp = manifest["experiments"].setdefault(experiment_name, {})
        runs = exp.setdefault("runs", {})
        if run_key in runs and runs[run_key].get("status") == "running":
            raise RuntimeError(
                f"Run {run_key} already claimed by "
                f"{runs[run_key].get('vm_id', 'unknown')}")
        runs[run_key] = {
            "status": "running",
            "vm_id": vm_id,
            "started": datetime.now().isoformat(),
        }
        return manifest

    gcs.update_manifest(updater)
    print(f"Claimed run {run_key} on {vm_id}")


# ── Hydra config loading ────────────────────────────────────────────────

def load_hydra_config(run_config, extra_overrides=None):
    extra_overrides = extra_overrides or []
    with initialize(version_base=None, config_path="../cfgs",
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


def _build_hydra_overrides(args):
    hydra_overrides = []
    if args.batch_size is not None:
        hydra_overrides.append(f"batch_size={args.batch_size}")
    if args.filter_by_length is not None:
        hydra_overrides.append(f"filter_by_length={args.filter_by_length}")
    if args.cache_size is not None:
        hydra_overrides.append(f"cache_size={args.cache_size}")
        hydra_overrides.append(f"max_memory_length={args.cache_size}")
    hydra_overrides.extend(args.override)
    return hydra_overrides


def _load_namm_checkpoint_into_model(args, memory_model, device):
    if not args.namm_checkpoint:
        return

    print(f"Loading NAMM checkpoint: {args.namm_checkpoint}")
    ckpt = torch.load(args.namm_checkpoint, map_location="cpu",
                      weights_only=False)
    evo_state = ckpt["evolution_state"]

    best_member = evo_state["best_member"]
    params = best_member.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)

    buffers_prefix = "stored_buffers_to_save."
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)
        print(f"  Loaded {len(buffers_dict)} normalization buffers")

    print(
        f"  Loaded NAMM best_member ({best_member.shape[0]} params) "
        f"from iter {ckpt.get('iter_num', '?')}, "
        f"best_val={ckpt.get('best_val_loss', '?')}"
    )


def _maybe_restore_resume_checkpoint(
    memory_model,
    resume_checkpoint,
    resumed_training_state,
):
    """Load a saved ES checkpoint into the model if present."""
    numpy_state_restored = False
    if resume_checkpoint:
        print(f"Loading ES checkpoint weights: {resume_checkpoint}")

        es_state = torch.load(
            resume_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model_params = dict(memory_model.named_parameters())
        is_delta = es_state.pop("__format__", None) == "delta"
        loaded = 0
        for name, val in es_state.items():
            if name in model_params:
                if is_delta:
                    model_params[name].data.add_(val.to(model_params[name].dtype))
                else:
                    model_params[name].data.copy_(val)
                loaded += 1
        fmt = "delta" if is_delta else "full (legacy)"
        print(f"  Loaded {loaded} parameter tensors ({fmt} format)")

        if (resumed_training_state
                and "numpy_random_state" in resumed_training_state):
            from es_finetuning.trainer import restore_numpy_state
            restore_numpy_state(resumed_training_state["numpy_random_state"])
            numpy_state_restored = True
            print("  Restored numpy random state from training state")

    return numpy_state_restored


def _resolve_tpu_worker_count(explicit_count=None):
    if explicit_count is not None:
        if explicit_count <= 1:
            raise ValueError("--worker-count must be > 1 for TPU multichip execution.")
        return explicit_count

    env_candidates = (
        "WORLD_SIZE",
        "TPU_NUM_DEVICES",
        "PJRT_LOCAL_PROCESS_COUNT",
    )
    for env_name in env_candidates:
        raw_value = os.environ.get(env_name)
        if raw_value and raw_value.isdigit() and int(raw_value) > 1:
            return int(raw_value)

    try:
        import torch_xla.runtime as xr
        detected = int(xr.global_runtime_device_count())
        if detected > 1:
            return detected
    except Exception:
        pass

    try:
        import torch_xla.core.xla_model as xm
        detected = int(xm.xrt_world_size())
        if detected > 1:
            return detected
    except Exception:
        pass

    raise ValueError(
        "Unable to determine TPU worker count automatically. "
        "Pass --worker-count explicitly."
    )


def _print_bucket_summary(bucket_pools, is_master):
    if not is_master or not bucket_pools:
        return
    print("Active TPU prompt buckets:")
    for bucket_len, per_task_indices in sorted(bucket_pools.items()):
        task_summary = ", ".join(
            f"{task_name}={len(indices)}"
            for task_name, indices in sorted(per_task_indices.items())
        )
        print(f"  <= {bucket_len} tokens: {task_summary}")


def _maybe_build_bucketed_resample_fn(
    task_sampler,
    evaluator,
    *,
    sampled_requests_per_task,
    enable_bucketing,
    is_master,
):
    if not enable_bucketing:
        return None, None

    training_tasks = {
        task_name: task_sampler.lb_prompts_per_task[task_name]
        for task_name in task_sampler.lb_training_tasks
    }
    if not training_tasks:
        return None, None

    max_prompt_conditioning = None
    if evaluator.max_conditioning_length is not None:
        max_prompt_conditioning = (
            evaluator.max_conditioning_length - evaluator.max_gen_toks
        )
        if max_prompt_conditioning <= 0:
            max_prompt_conditioning = None

    bucket_pools = build_bucketed_request_pools(
        task_prompts=training_tasks,
        tokenizer=evaluator.tokenizer,
        sampled_requests_per_task=sampled_requests_per_task,
        max_prompt_conditioning=max_prompt_conditioning,
        add_special_tokens=evaluator.add_bos_token,
    )
    if not bucket_pools:
        if is_master:
            print(
                "WARNING: No eligible TPU prompt buckets found; "
                "falling back to the default random resampler."
            )
        return None, None

    _print_bucket_summary(bucket_pools, is_master)
    return (
        make_bucketed_resample_fn(
            task_sampler,
            bucket_pools,
            sampled_requests_per_task=sampled_requests_per_task,
        ),
        bucket_pools,
    )


def _run_compile_warmup(
    *,
    model,
    evaluate_fn,
    task_sampler,
    bucket_pools,
    sampled_requests_per_task,
    is_master,
):
    if not bucket_pools:
        return

    if is_master:
        print("Running TPU compile warmup across active prompt buckets...")

    for bucket_len, requests_dict in iter_compile_warmup_requests(
            bucket_pools,
            sampled_requests_per_task=sampled_requests_per_task):
        if is_master:
            print(f"  Warmup bucket <= {bucket_len} tokens")
        task_sampler.set_requests_per_task(requests_dict)
        evaluate_fn(model)


# ── Evaluation function factories ───────────────────────────────────────

def make_resample_fn(task_sampler, mini_batch_size, train=True):
    def resample_fn():
        task_sampler.resample_requests(
            train=train,
            sampled_requests_per_task=mini_batch_size,
        )
        return task_sampler.get_requests_per_task()
    return resample_fn


def make_evaluate_fn(task_sampler, evaluator, mini_batch_size, train=True):
    def evaluate_fn(model):
        score_dicts = task_sampler.evaluate(
            lm=evaluator,
            train=train,
            evolved_model=False,
            pop_reps=1,
            resample_requests=False,
            sampled_requests_per_task=mini_batch_size,
        )
        score = score_dicts[0].get("lb/qasper", 0.0) / 100.0
        return score
    return evaluate_fn


def make_full_eval_fn(task_sampler, evaluator):
    def full_eval_fn(model):
        score_dicts = task_sampler.evaluate(
            lm=evaluator,
            train=False,
            evolved_model=False,
            pop_reps=1,
            resample_requests=True,
            sampled_requests_per_task=None,
        )
        scores = score_dicts[0]
        num_samples = sum(task_sampler.num_prompts_per_lb_task.values())
        return {"scores": scores, "num_samples": num_samples}
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


# ── Base LLM parameter detection ────────────────────────────────────────

def get_base_llm_param_names(model):
    if hasattr(model, "base_model_param_keys"):
        model_params = dict(model.named_parameters())
        valid_keys = [k for k in model.base_model_param_keys if k in model_params]
        if valid_keys:
            return valid_keys
    return [
        name for name, _ in model.named_parameters()
        if not name.startswith("memory_policy.")
    ]


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ES fine-tuning of base LLM weights with NAMM evaluation")

    # Experiment hierarchy
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for this run (e.g. cache1024_i50)")
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

    # Cache size (NAMM)
    parser.add_argument("--cache_size", type=int, default=None)

    # Resume from checkpoint
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    # Data splits
    parser.add_argument("--train_samples", type=int, default=150)

    # Q/A examples
    parser.add_argument("--n_examples", type=int, default=10,
                        help="Number of Q/A examples to capture")
    parser.add_argument("--skip-full-eval", action="store_true",
                        help="Skip baseline and final full-validation evaluation")
    parser.add_argument("--skip-examples", action="store_true",
                        help="Skip capturing final Q/A examples")

    # GCS and checkpointing
    parser.add_argument("--gcs", dest="gcs", action="store_true",
                        help="Enable GCS experiment management and checkpointing")
    parser.add_argument("--no-gcs", dest="gcs", action="store_false",
                        help="Disable GCS experiment management and checkpointing")
    parser.set_defaults(gcs=True)
    parser.add_argument("--sync-xla-cache", dest="sync_xla_cache", action="store_true",
                        help="Sync local XLA cache to GCS on exit (TPU + --gcs only)")
    parser.add_argument("--no-sync-xla-cache", dest="sync_xla_cache", action="store_false",
                        help="Disable XLA cache sync on exit")
    parser.set_defaults(sync_xla_cache=False)
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="Save checkpoint every N iterations (0 = final only)")

    # Execution and benchmarking
    parser.add_argument(
        "--execution-backend",
        type=str,
        default="single_process",
        choices=["single_process", "tpu_multichip_exact"],
        help="Execution backend for ES population evaluation.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Worker count for TPU multichip execution. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--benchmark-mode",
        type=str,
        default="off",
        choices=["off", "train_only"],
        help="Benchmark mode. 'train_only' skips full eval/examples and records phase summaries.",
    )
    parser.add_argument(
        "--benchmark-warmup-iterations",
        type=int,
        default=1,
        help="Ignore the first N iterations when computing steady-state benchmark summaries.",
    )
    parser.add_argument(
        "--bucketed-tpu-sampling",
        dest="bucketed_tpu_sampling",
        action="store_true",
        help="Use tokenizer-length buckets to keep TPU training batches shape-stable.",
    )
    parser.add_argument(
        "--no-bucketed-tpu-sampling",
        dest="bucketed_tpu_sampling",
        action="store_false",
        help="Disable TPU length-bucketed minibatch sampling.",
    )
    parser.add_argument(
        "--compile-warmup",
        dest="compile_warmup",
        action="store_true",
        help="Run one training-path warmup pass per active TPU prompt bucket before training.",
    )
    parser.add_argument(
        "--no-compile-warmup",
        dest="compile_warmup",
        action="store_false",
        help="Disable TPU compile warmup.",
    )
    parser.set_defaults(bucketed_tpu_sampling=None, compile_warmup=None)

    # Extra Hydra overrides
    parser.add_argument("--override", action="append", default=[])

    return parser.parse_args()


def _apply_runtime_defaults(args):
    if args.benchmark_mode == "train_only":
        args.skip_full_eval = True
        args.skip_examples = True

    if args.execution_backend == "tpu_multichip_exact":
        if args.bucketed_tpu_sampling is None:
            args.bucketed_tpu_sampling = True
        if args.compile_warmup is None:
            args.compile_warmup = True
    else:
        if args.bucketed_tpu_sampling is None:
            args.bucketed_tpu_sampling = False
        if args.compile_warmup is None:
            args.compile_warmup = False


def _run_training_process(
    args,
    *,
    experiment_name,
    run_dir,
    resumed_history,
    resumed_training_state,
    gcs_client,
    preemption_handler,
    population_executor,
    is_master,
    worker_rank,
    worker_count,
):
    startup_start = time.time()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()
    is_tpu = is_tpu_device(device)
    if is_master:
        _maybe_enable_xla_cache_sync(args, is_tpu)

    if args.execution_backend == "tpu_multichip_exact" and not is_tpu:
        raise RuntimeError(
            "The 'tpu_multichip_exact' backend requires an active XLA device."
        )

    if is_master:
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
        print(f"Execution backend: {args.execution_backend}")
        print(f"Workers: {worker_count} (rank {worker_rank})")
        if gcs_client:
            print(f"GCS: gs://{gcs_client.bucket_name}/ (checkpoint every "
                  f"{args.checkpoint_every} iter)")
        if args.sync_xla_cache:
            print(
                "XLA cache sync requested: enabled when TPU+GCS prerequisites "
                "are met"
            )
        print()

    # 1. Load config and model
    if is_master:
        print("Loading config and model via Hydra...")
    cfg = load_hydra_config(
        args.run_config,
        extra_overrides=_build_hydra_overrides(args),
    )

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    memory_model.to(device)

    if is_tpu:
        fixed_batch_size = validate_tpu_batch_settings(
            memory_evaluator.batch_size,
            args.mini_batch_size,
            context="training",
        )
        memory_evaluator.batch_size_per_gpu = fixed_batch_size
        memory_evaluator.batch_size = fixed_batch_size
        if is_master:
            print(f"TPU mode: using fixed batch size {fixed_batch_size}")

    # 2. Load NAMM checkpoint
    _load_namm_checkpoint_into_model(args, memory_model, device)

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    # 3. Create task sampler
    if is_master:
        print("Creating task sampler...")
    task_sampler = make_task_sampler(cfg=cfg)

    # 4. Auto-detect batch size
    if memory_evaluator.batch_size == "auto":
        if is_master:
            print("Auto-detecting optimal batch size...")
        detected_bs = memory_evaluator._detect_batch_size()
        memory_evaluator.batch_size_per_gpu = detected_bs
        memory_evaluator.batch_size = detected_bs
        if is_master:
            print(f"  Auto-detected batch size: {detected_bs}")
    elif is_master:
        print(f"  Using fixed batch size: {memory_evaluator.batch_size}")

    # 5. Get base LLM param names
    param_names = get_base_llm_param_names(memory_model)
    if is_master:
        print(f"Base LLM parameters to optimize: {len(param_names)}")

    # 6. Build evaluation functions
    if is_master:
        print("Building evaluation functions...")
    resample_fn = make_resample_fn(
        task_sampler, args.mini_batch_size, train=True)
    evaluate_fn = make_evaluate_fn(
        task_sampler, memory_evaluator, args.mini_batch_size, train=True)

    bucket_pools = None
    if is_tpu:
        bucketed_resample_fn, bucket_pools = _maybe_build_bucketed_resample_fn(
            task_sampler,
            memory_evaluator,
            sampled_requests_per_task=args.mini_batch_size,
            enable_bucketing=args.bucketed_tpu_sampling,
            is_master=is_master,
        )
        if bucketed_resample_fn is not None:
            resample_fn = bucketed_resample_fn

    if worker_count > 1:
        resample_step = {"value": 0}
        base_resample_fn = resample_fn

        def synchronized_resample_fn():
            requests_dict = None
            if is_master:
                requests_dict = base_resample_fn()
            requests_dict = population_executor.broadcast_object(
                f"sampled_requests_iter_{resample_step['value']}",
                requests_dict,
            )
            if requests_dict is not None:
                task_sampler.set_requests_per_task(requests_dict)
            resample_step["value"] += 1
            return requests_dict

        resample_fn = synchronized_resample_fn

    full_eval_fn = None
    if is_master and not args.skip_full_eval:
        full_eval_fn = make_full_eval_fn(task_sampler, memory_evaluator)
    examples_fn = None
    if is_master and not args.skip_examples:
        examples_fn = make_examples_fn(
            task_sampler, memory_evaluator, n_examples=args.n_examples)

    # 7. Configure and run ES
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
        execution_backend=args.execution_backend,
        benchmark_mode=args.benchmark_mode,
        benchmark_warmup_iterations=args.benchmark_warmup_iterations,
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
            "train_samples": args.train_samples,
            "num_base_params": len(param_names),
            "experiment": experiment_name,
            "method": args.method,
            "run_name": args.run_name,
            "skip_full_eval": args.skip_full_eval,
            "skip_examples": args.skip_examples,
            "execution_backend": args.execution_backend,
            "worker_count": worker_count,
            "worker_rank": worker_rank,
            "bucketed_tpu_sampling": args.bucketed_tpu_sampling,
            "compile_warmup": args.compile_warmup,
            "benchmark_mode": args.benchmark_mode,
        },
        resume_from=args.resume_checkpoint,
        gcs_client=gcs_client if is_master else None,
        preemption_handler=preemption_handler,
        experiment_name=experiment_name,
        method=args.method,
        run_name=args.run_name,
        population_executor=population_executor,
        is_master=is_master,
        startup_time_s=0.0,
    )

    numpy_state_restored = _maybe_restore_resume_checkpoint(
        memory_model,
        args.resume_checkpoint,
        resumed_training_state,
    )

    if is_tpu and args.compile_warmup and bucket_pools:
        _run_compile_warmup(
            model=memory_model,
            evaluate_fn=evaluate_fn,
            task_sampler=task_sampler,
            bucket_pools=bucket_pools,
            sampled_requests_per_task=args.mini_batch_size,
            is_master=is_master,
        )
        population_executor.barrier("compile_warmup_complete")

    trainer.startup_time_s = time.time() - startup_start

    if is_master:
        print()
        print("Starting ES training...")
    with torch.no_grad():
        trainer.train(
            resumed_history=resumed_history,
            numpy_state_restored=numpy_state_restored,
        )


def _tpu_multichip_worker(worker_rank, payload):
    args = argparse.Namespace(**payload["args"])
    worker_count = payload["worker_count"]

    gcs_client = None
    if args.gcs and worker_rank == 0:
        from es_finetuning.gcs import GCSClient
        gcs_client = GCSClient()

    preemption_handler = None
    if args.gcs:
        from es_finetuning.preemption import PreemptionHandler
        preemption_handler = PreemptionHandler()

    executor = ExactTpuPopulationExecutor(worker_rank, worker_count)
    _run_training_process(
        args,
        experiment_name=payload["experiment_name"],
        run_dir=payload["run_dir"],
        resumed_history=payload["resumed_history"],
        resumed_training_state=payload["resumed_training_state"],
        gcs_client=gcs_client,
        preemption_handler=preemption_handler,
        population_executor=executor,
        is_master=worker_rank == 0,
        worker_rank=worker_rank,
        worker_count=worker_count,
    )


def _run_tpu_multichip_exact(
    args,
    *,
    experiment_name,
    run_dir,
    resumed_history,
    resumed_training_state,
):
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError as exc:
        raise RuntimeError(
            "torch_xla is required for the TPU multichip execution backend."
        ) from exc

    worker_count = _resolve_tpu_worker_count(args.worker_count)
    args.worker_count = worker_count

    print(f"Launching TPU multichip exact backend with {worker_count} workers...")
    payload = {
        "args": vars(args).copy(),
        "experiment_name": experiment_name,
        "run_dir": run_dir,
        "resumed_history": resumed_history,
        "resumed_training_state": resumed_training_state,
        "worker_count": worker_count,
    }
    xmp.spawn(
        _tpu_multichip_worker,
        args=(payload,),
        nprocs=worker_count,
        start_method="spawn",
    )


def main():
    args = parse_args()
    _apply_runtime_defaults(args)

    # Auto-detect method
    if args.method is None:
        args.method = "es_namm" if args.namm_checkpoint else "es_only"

    # Setup GCS client if requested
    gcs = None
    if args.gcs:
        from es_finetuning.gcs import GCSClient
        gcs = GCSClient()

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

    if args.execution_backend == "tpu_multichip_exact":
        _run_tpu_multichip_exact(
            args,
            experiment_name=experiment_name,
            run_dir=run_dir,
            resumed_history=resumed_history,
            resumed_training_state=resumed_training_state,
        )
        return

    preemption_handler = None
    if args.gcs:
        from es_finetuning.preemption import PreemptionHandler
        preemption_handler = PreemptionHandler()

    _run_training_process(
        args,
        experiment_name=experiment_name,
        run_dir=run_dir,
        resumed_history=resumed_history,
        resumed_training_state=resumed_training_state,
        gcs_client=gcs,
        preemption_handler=preemption_handler,
        population_executor=SingleProcessPopulationExecutor(),
        is_master=True,
        worker_rank=0,
        worker_count=1,
    )


if __name__ == "__main__":
    main()
