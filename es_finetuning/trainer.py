"""Core Evolution Strategies training loop.

The ESTrainer does NOT perform inference or scoring — it accepts an
evaluate_fn callable that returns a scalar reward for the current model
state. This keeps all model/task-specific logic in the caller.
"""

import json
import os
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from .config import ESConfig
from .noise import apply_es_update, perturb_weights, restore_weights
from .utils import force_memory_cleanup, setup_tensorboard


def _save_results_json(path, config_dict, history, baseline_eval,
                       full_eval_result, total_time):
    """Write results.json summarizing the experiment."""
    results = {
        "config": config_dict,
        "training": {
            "total_time_s": round(total_time, 1),
            "total_time_h": round(total_time / 3600, 2),
            "iterations": len(history["mean"]),
            "reward_per_iteration": {
                "mean": history["mean"],
                "min": history["min"],
                "max": history["max"],
                "time_s": history["time"],
            },
        },
    }
    if baseline_eval is not None:
        results["baseline_eval"] = baseline_eval
    if full_eval_result is not None:
        results["full_eval"] = full_eval_result
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {path}")


def _clean_task_name(name):
    """Strip benchmark prefixes like 'lb/' for display."""
    for prefix in ("lb/", "ib/", "cb/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


class ESTrainer:
    """Evolution Strategies trainer for LLM weight optimization.

    Args:
        model: The model whose weights are perturbed and updated.
        param_names: List of parameter names to optimize (e.g. base LLM
            params only, excluding memory-policy params).
        evaluate_fn: Callable(model) -> float. Called after perturbation
            to get a scalar reward. Should run inference + scoring.
        config: ES hyperparameters.
        pre_step_fn: Optional callable() invoked once per iteration before
            evaluating population members. Use this to resample a shared
            data batch so all members are compared on the same inputs.
        full_eval_fn: Optional callable(model) -> dict for full evaluation
            at baseline and end of training. Returns {"scores": {...}, ...}.
        examples_fn: Optional callable(model) -> list[dict] to capture
            Q/A examples at end of training. Each dict has question,
            context_snippet, gold_answers, prediction.
        metadata: Optional dict of extra info to save in config.json.
    """

    def __init__(self, model, param_names: list[str],
                 evaluate_fn, config: ESConfig,
                 pre_step_fn=None,
                 full_eval_fn=None, examples_fn=None,
                 metadata=None, resume_from=None):
        self.model = model
        self.param_names = param_names
        self.evaluate_fn = evaluate_fn
        self.pre_step_fn = pre_step_fn
        self.full_eval_fn = full_eval_fn
        self.examples_fn = examples_fn
        self.config = config
        self.metadata = metadata or {}
        self.resume_from = resume_from

        # Verify all param_names exist in model
        model_param_names = {n for n, _ in model.named_parameters()}
        missing = set(param_names) - model_param_names
        if missing:
            raise ValueError(
                f"param_names contains {len(missing)} names not found in "
                f"model.named_parameters(). First 5: {list(missing)[:5]}"
            )

        # Snapshot initial parameter values for delta checkpointing.
        all_params = dict(model.named_parameters())
        self._initial_params = {
            name: all_params[name].detach().cpu().clone()
            for name in param_names
        }

    def train(self):
        """Run the ES optimization loop."""
        cfg = self.config
        np.random.seed(cfg.initial_seed)

        # Determine start iteration (for resuming)
        start_iteration = 0
        if self.resume_from is not None:
            import re
            match = re.search(r'iter(\d+)', os.path.basename(self.resume_from))
            if match:
                start_iteration = int(match.group(1))
            for i in range(start_iteration):
                np.random.randint(0, 2**30, size=cfg.population_size,
                                  dtype=np.int64)
            print(f"Resuming from iteration {start_iteration} "
                  f"(checkpoint: {self.resume_from})")

        # Setup logging — use log_dir directly (caller sets up hierarchy)
        log_dir = cfg.log_dir
        writer = setup_tensorboard(log_dir)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save full config
        config_dict = asdict(cfg)
        config_dict.update(self.metadata)
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Config saved to {config_path}")

        print(f"ES Training: pop={cfg.population_size}, iter={cfg.num_iterations}, "
              f"sigma={cfg.sigma}, alpha={cfg.alpha}, mode={cfg.noise_mode}")
        print(f"Optimizing {len(self.param_names)} parameters")
        print(f"Logging to {log_dir}")

        # Baseline full eval (skip if resuming)
        baseline_eval = None
        if self.full_eval_fn and start_iteration == 0:
            print("\nRunning baseline evaluation (before training)...")
            eval_start = time.time()
            baseline_eval = self.full_eval_fn(self.model)
            eval_time = time.time() - eval_start
            baseline_eval["time_s"] = round(eval_time, 1)
            print(f"Baseline eval complete in {eval_time:.1f}s")
            for k, v in baseline_eval.get("scores", {}).items():
                print(f"  {_clean_task_name(k)}: {v:.2f}")

        # Training loop
        history = {"mean": [], "min": [], "max": [], "time": []}
        training_start = time.time()

        for iteration in range(start_iteration, cfg.num_iterations):
            iter_start = time.time()
            force_memory_cleanup()

            seeds = np.random.randint(
                0, 2**30, size=cfg.population_size, dtype=np.int64
            ).tolist()

            if self.pre_step_fn is not None:
                self.pre_step_fn()

            rewards = []
            for member_idx, seed in enumerate(seeds):
                perturb_weights(
                    self.model, seed, cfg.sigma,
                    self.param_names, cfg.noise_mode
                )

                reward = self.evaluate_fn(self.model)
                rewards.append(reward)

                restore_weights(
                    self.model, seed, cfg.sigma,
                    self.param_names, cfg.noise_mode
                )

                force_memory_cleanup()

            rewards_arr = np.array(rewards, dtype=np.float32)
            normalized = (
                (rewards_arr - rewards_arr.mean())
                / (rewards_arr.std() + 1e-8)
            )

            apply_es_update(
                self.model, seeds, normalized, cfg.sigma, cfg.alpha,
                self.param_names, cfg.population_size, cfg.noise_mode
            )

            iter_time = time.time() - iter_start

            mean_r = rewards_arr.mean().item()
            min_r = rewards_arr.min().item()
            max_r = rewards_arr.max().item()

            history["mean"].append(round(mean_r, 6))
            history["min"].append(round(min_r, 6))
            history["max"].append(round(max_r, 6))
            history["time"].append(round(iter_time, 1))

            writer.add_scalar("reward/mean", mean_r, iteration)
            writer.add_scalar("reward/min", min_r, iteration)
            writer.add_scalar("reward/max", max_r, iteration)
            writer.add_scalar("time/iteration_sec", iter_time, iteration)

            if torch.cuda.is_available():
                mem_mb = torch.cuda.memory_allocated() / 1024**2
                writer.add_scalar("device/memory_mb", mem_mb, iteration)

            print(f"[{iteration + 1}/{cfg.num_iterations}] "
                  f"mean={mean_r:.4f} min={min_r:.4f} max={max_r:.4f} "
                  f"time={iter_time:.1f}s")

        total_time = time.time() - training_start
        print(f"Training complete in {total_time:.1f}s ({total_time / 3600:.1f}h)")

        # Final checkpoint only
        self._save_checkpoint(checkpoint_dir)
        writer.close()

        # Full evaluation on fine-tuned weights
        full_eval_result = None
        if self.full_eval_fn:
            print("\nRunning full evaluation (after training)...")
            eval_start = time.time()
            full_eval_result = self.full_eval_fn(self.model)
            eval_time = time.time() - eval_start
            full_eval_result["time_s"] = round(eval_time, 1)
            print(f"Full eval complete in {eval_time:.1f}s")
            for k, v in full_eval_result.get("scores", {}).items():
                print(f"  {_clean_task_name(k)}: {v:.2f}")

        # Save results
        results_path = os.path.join(log_dir, "results.json")
        _save_results_json(results_path, config_dict, history, baseline_eval,
                           full_eval_result, total_time)

        # Capture Q/A examples
        if self.examples_fn:
            print("\nCapturing Q/A examples...")
            examples = self.examples_fn(self.model)
            examples_path = os.path.join(log_dir, "examples.json")
            with open(examples_path, "w") as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(examples)} examples to {examples_path}")

        print("Done.")

    def _save_checkpoint(self, checkpoint_dir):
        """Save delta checkpoint (current - initial weights)."""
        path = os.path.join(checkpoint_dir, "es_checkpoint_final.pt")

        state = {"__format__": "delta"}
        all_params = dict(self.model.named_parameters())
        for name in self.param_names:
            delta = all_params[name].detach().cpu() - self._initial_params[name]
            state[name] = delta.to(torch.float16)

        torch.save(state, path)
        size_mb = os.path.getsize(path) / 1024**2
        print(f"  Checkpoint saved: {path} ({size_mb:.1f} MB)")
