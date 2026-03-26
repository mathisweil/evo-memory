"""Shared experiment-management helpers for run_es.py, run_lora.py, run_joint.py.

Extracted from run_es.py to break the run_lora → run_es import dependency.
"""

import argparse
import json
import os
import socket
from datetime import datetime

import yaml
from hydra import compose, initialize


# ── Constants (set by the importing script if different) ──────────────────────
EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
)
MANIFEST_PATH = os.path.join(EXPERIMENTS_DIR, "manifest.json")


# ── Local manifest management ─────────────────────────────────────────────────

def load_manifest() -> dict:
    """Load the local experiment manifest, returning an empty one if absent."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"experiments": {}}


def save_manifest(manifest: dict) -> None:
    """Persist the local experiment manifest to disk."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def normalize_name(name: str) -> str:
    """Convert a bare integer ID to the canonical experiment_N name.

    Args:
        name: Either "experiment_N" already, or a bare integer string like "3".

    Returns:
        Canonical experiment name, e.g. "experiment_3".
    """
    if name.isdigit():
        return f"experiment_{name}"
    return name


def _next_experiment_id(manifest: dict) -> int:
    existing = [int(k.split("_")[1]) for k in manifest["experiments"]
                if k.startswith("experiment_")]
    return max(existing, default=0) + 1


def get_or_create_experiment(experiment_id=None):
    """Get an existing active experiment or create a new one.

    If no experiment_id is given, uses the most recent active experiment.
    If no active experiments exist, creates a new one.

    Args:
        experiment_id: Integer ID of an existing experiment, or None.

    Returns:
        Tuple of (experiment_name, manifest).
    """
    manifest = load_manifest()

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
    save_manifest(manifest)
    print(f"Created new experiment: {name}")
    return name, manifest


# ── GCS experiment manifest management ───────────────────────────────────────

def get_or_create_experiment_gcs(gcs, experiment_id=None):
    """GCS-backed version of get_or_create_experiment.

    Args:
        gcs: GCSClient instance.
        experiment_id: Integer ID of an existing experiment, or None.

    Returns:
        Tuple of (experiment_name, manifest).
    """
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


def claim_run_gcs(gcs, experiment_name: str, method: str, run_name: str) -> None:
    """Atomically register a run as 'running' in the GCS manifest.

    Args:
        gcs: GCSClient instance.
        experiment_name: e.g. "experiment_3".
        method: Training method identifier, e.g. "es_namm".
        run_name: User-specified run name.

    Raises:
        RuntimeError: If the run is already claimed by another VM.
    """
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


# ── Hydra config loading ──────────────────────────────────────────────────────

def load_hydra_config(run_config: str, extra_overrides: list | None = None):
    """Load a Hydra experiment config by run_config name.

    Args:
        run_config: Name of the Hydra run config (e.g. "namm_bam_i1_llama32_1b").
        extra_overrides: Additional Hydra override strings.

    Returns:
        Composed Hydra DictConfig.
    """
    extra_overrides = extra_overrides or []
    with initialize(version_base=None, config_path="../config",
                    job_name="es_finetuning"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"run@_global_={run_config}",
                "wandb_log=false",
                "wandb_project=Experiments",
            ] + extra_overrides,
        )
    return cfg


# ── CLI config-file helper ────────────────────────────────────────────────────

def load_config_defaults(parser: argparse.ArgumentParser) -> None:
    """Inject defaults from a YAML config file into an argparse parser.

    Reads ``--config`` from sys.argv without triggering a full parse, then
    calls ``parser.set_defaults`` with the YAML content so that CLI flags
    still override the file values.

    Args:
        parser: The ArgumentParser to inject defaults into (mutated in place).
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**{k: v for k, v in cfg.items() if v is not None})


# ── Model utilities ───────────────────────────────────────────────────────────

def get_base_llm_param_names(model) -> list[str]:
    """Return the names of base LLM parameters (excluding memory policy params).

    Args:
        model: The memory model with optional ``base_model_param_keys`` attribute.

    Returns:
        List of parameter names to optimise with ES.
    """
    if hasattr(model, "base_model_param_keys"):
        model_params = dict(model.named_parameters())
        valid_keys = [k for k in model.base_model_param_keys if k in model_params]
        if valid_keys:
            return valid_keys
    return [
        name for name, _ in model.named_parameters()
        if not name.startswith("memory_policy.")
    ]


# ── Evaluation function factories ─────────────────────────────────────────────

def _gen_model_kwargs(temperature: float) -> dict:
    """Build model_kwargs dict for generation temperature.

    Args:
        temperature: Sampling temperature; 0 means greedy decoding.

    Returns:
        Dict passed as model_kwargs to the evaluator.
    """
    if temperature > 0:
        return {"temperature": temperature, "do_sample": True}
    return {}


def make_resample_fn(task_sampler, mini_batch_size: int, train: bool = True):
    """Return a callable that resamples the task mini-batch.

    Args:
        task_sampler: TaskSampler instance.
        mini_batch_size: Number of questions sampled per fitness evaluation.
        train: Whether to sample from the training split.

    Returns:
        Zero-argument callable suitable for ESTrainer.pre_step_fn.
    """
    def resample_fn():
        task_sampler.resample_requests(
            train=train,
            sampled_requests_per_task=mini_batch_size,
        )
    return resample_fn


def make_evaluate_fn(task_sampler, evaluator, mini_batch_size: int,
                     train: bool = True, temperature: float = 0.0,
                     num_samples: int = 1):
    """Return a callable(model) -> float for ES fitness evaluation.

    Args:
        task_sampler: TaskSampler instance.
        evaluator: MemoryHFEvaluator instance.
        mini_batch_size: Questions sampled per evaluation call.
        train: Whether to evaluate on the training split.
        temperature: Generation temperature (0 = greedy).
        num_samples: Number of generation samples averaged for the fitness score.

    Returns:
        Callable that takes a model and returns a scalar fitness in [0, 1].
    """
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


def make_full_eval_fn(task_sampler, evaluator, temperature: float = 0.0,
                      num_samples: int = 1):
    """Return a callable(model) -> dict for full test-set evaluation.

    Args:
        task_sampler: TaskSampler instance.
        evaluator: MemoryHFEvaluator instance.
        temperature: Generation temperature (0 = greedy).
        num_samples: Number of generation samples averaged for the reported score.

    Returns:
        Callable returning {"scores": dict, "num_samples": int}.
    """
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
        avg_scores = {
            k: sum(d[k] for d in all_score_dicts) / len(all_score_dicts)
            for k in all_score_dicts[0]
        }
        if task_sampler._test_idxs_per_task is not None:
            num_data_samples = sum(
                len(v) for v in task_sampler._test_idxs_per_task.values())
        else:
            num_data_samples = sum(task_sampler.num_prompts_per_lb_task.values())
        return {"scores": avg_scores, "num_samples": num_data_samples}
    return full_eval_fn
