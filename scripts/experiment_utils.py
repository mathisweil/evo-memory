"""Shared experiment-management helpers for run_es.py and run_lora.py.

Extracted from run_es.py to break the run_lora → run_es import dependency.
"""

import json
import os
import socket
from datetime import datetime

from hydra import compose, initialize


# ── Constants (set by the importing script if different) ──────────────────────
EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
)
MANIFEST_PATH = os.path.join(EXPERIMENTS_DIR, "manifest.json")


# ── Local manifest management ─────────────────────────────────────────────────

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


# ── GCS experiment manifest management ───────────────────────────────────────

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


# ── Hydra config loading ──────────────────────────────────────────────────────

def load_hydra_config(run_config, extra_overrides=None):
    extra_overrides = extra_overrides or []
    with initialize(version_base=None, config_path="../config",
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


# ── Model utilities ───────────────────────────────────────────────────────────

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
