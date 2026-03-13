"""Core Evolution Strategies training loop.

The ESTrainer does NOT perform inference or scoring — it accepts an
evaluate_fn callable that returns a scalar reward for the current model
state. This keeps all model/task-specific logic in the caller.
"""

import datetime
import glob
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from .config import ESConfig
from .population import (
    SingleProcessPopulationExecutor,
    summarize_phase_history,
)
from .utils import force_memory_cleanup, setup_tensorboard


def _save_results_json(path, config_dict, history, baseline_eval,
                       full_eval_result, total_time,
                       phase_history=None, benchmark_summary=None):
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
    if phase_history:
        results["training"]["phase_timing_s"] = phase_history
    if benchmark_summary is not None:
        results["training"]["benchmark"] = benchmark_summary
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
        resume_from: Path to a checkpoint to resume from.
        gcs_client: Optional GCSClient for cloud checkpointing.
        preemption_handler: Optional PreemptionHandler for spot VMs.
        experiment_name: Experiment name (needed for GCS paths).
        method: Method name (needed for GCS paths).
        run_name: Run name (needed for GCS paths).
    """

    def __init__(self, model, param_names: list[str],
                 evaluate_fn, config: ESConfig,
                 pre_step_fn=None,
                 full_eval_fn=None, examples_fn=None,
                 metadata=None, resume_from=None,
                 gcs_client=None, preemption_handler=None,
                 experiment_name=None, method=None, run_name=None,
                 population_executor=None, is_master=True,
                 startup_time_s: float = 0.0):
        self.model = model
        self.param_names = param_names
        self.evaluate_fn = evaluate_fn
        self.pre_step_fn = pre_step_fn
        self.full_eval_fn = full_eval_fn
        self.examples_fn = examples_fn
        self.config = config
        self.metadata = metadata or {}
        self.resume_from = resume_from
        self.gcs_client = gcs_client
        self.preemption_handler = preemption_handler
        self.experiment_name = experiment_name
        self.method = method
        self.run_name = run_name
        self.population_executor = (
            population_executor or SingleProcessPopulationExecutor()
        )
        self.is_master = bool(is_master and self.population_executor.is_master)
        self.startup_time_s = float(startup_time_s)

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

    def _print(self, *args, **kwargs):
        if self.is_master:
            print(*args, **kwargs)

    def train(self, resumed_history=None, numpy_state_restored=False):
        """Run the ES optimization loop.

        Args:
            resumed_history: If resuming, prepopulate with prior history
                dict (keys: mean, min, max, time).
            numpy_state_restored: If True, numpy RNG state was already
                restored from a training_state JSON — skip seed init
                and replay.
        """
        cfg = self.config

        # Determine start iteration (for resuming)
        start_iteration = 0
        if self.resume_from is not None:
            import re
            match = re.search(r'iter(\d+)', os.path.basename(self.resume_from))
            if match:
                start_iteration = int(match.group(1))

            if numpy_state_restored:
                # Numpy RNG was restored exactly from training state —
                # no need to seed or replay.
                self._print(
                    f"Resuming from iteration {start_iteration} "
                    f"(checkpoint: {self.resume_from}, "
                    f"numpy state restored)"
                )
            else:
                # Fallback: seed and replay RNG to approximate state.
                np.random.seed(cfg.initial_seed)
                for i in range(start_iteration):
                    np.random.randint(0, 2**30, size=cfg.population_size,
                                      dtype=np.int64)
                self._print(
                    f"Resuming from iteration {start_iteration} "
                    f"(checkpoint: {self.resume_from}, "
                    f"seed replay)"
                )
        else:
            np.random.seed(cfg.initial_seed)

        # Setup logging — use log_dir directly (caller sets up hierarchy)
        log_dir = cfg.log_dir
        writer = setup_tensorboard(log_dir) if self.is_master else None
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        if self.is_master:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Save full config
        config_dict = asdict(cfg)
        config_dict.update(self.metadata)
        if self.is_master:
            config_path = os.path.join(log_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)
            self._print(f"Config saved to {config_path}")

        # Upload config to GCS
        if self.is_master and self.gcs_client and self.experiment_name:
            self.gcs_client.upload_run_file(
                config_path, self.experiment_name, self.method, self.run_name,
                "config.json")

        self._print(
            f"ES Training: pop={cfg.population_size}, iter={cfg.num_iterations}, "
            f"sigma={cfg.sigma}, alpha={cfg.alpha}, mode={cfg.noise_mode}"
        )
        self._print(f"Optimizing {len(self.param_names)} parameters")
        self._print(f"Logging to {log_dir}")
        if cfg.checkpoint_every > 0:
            self._print(
                f"Periodic checkpoints every {cfg.checkpoint_every} iterations"
            )
        if self.gcs_client:
            self._print(f"GCS sync enabled: gs://{self.gcs_client.bucket_name}/")
        if self.preemption_handler:
            self._print("Preemption handler active (SIGTERM)")

        # Baseline full eval (skip if resuming)
        baseline_eval = None
        if self.is_master and self.full_eval_fn and start_iteration == 0:
            self._print("\nRunning baseline evaluation (before training)...")
            eval_start = time.time()
            baseline_eval = self.full_eval_fn(self.model)
            eval_time = time.time() - eval_start
            baseline_eval["time_s"] = round(eval_time, 1)
            self._print(f"Baseline eval complete in {eval_time:.1f}s")
            for k, v in baseline_eval.get("scores", {}).items():
                self._print(f"  {_clean_task_name(k)}: {v:.2f}")

        # Training loop
        if resumed_history:
            history = resumed_history
        else:
            history = {"mean": [], "min": [], "max": [], "time": []}
        phase_history = {}
        training_start = time.time()
        preempted = False

        for iteration in range(start_iteration, cfg.num_iterations):
            force_memory_cleanup()

            seeds = None
            if self.is_master:
                seeds = np.random.randint(
                    0,
                    2**30,
                    size=cfg.population_size,
                    dtype=np.int64,
                ).tolist()
            seeds = self.population_executor.broadcast_object(
                f"es_seeds_iter_{iteration}",
                seeds,
            )

            if self.pre_step_fn is not None:
                self.pre_step_fn()

            iter_result = self.population_executor.run_iteration(
                model=self.model,
                param_names=self.param_names,
                evaluate_fn=self.evaluate_fn,
                seeds=seeds,
                sigma=cfg.sigma,
                alpha=cfg.alpha,
                noise_mode=cfg.noise_mode,
                population_size=cfg.population_size,
                iteration=iteration,
            )
            rewards_arr = iter_result.rewards
            iter_time = iter_result.phase_times.get("iteration_s", 0.0)

            mean_r = rewards_arr.mean().item()
            min_r = rewards_arr.min().item()
            max_r = rewards_arr.max().item()

            history["mean"].append(round(mean_r, 6))
            history["min"].append(round(min_r, 6))
            history["max"].append(round(max_r, 6))
            history["time"].append(round(iter_time, 1))

            for phase_name, phase_value in iter_result.phase_times.items():
                phase_history.setdefault(phase_name, []).append(
                    round(float(phase_value), 6)
                )

            if writer is not None:
                writer.add_scalar("reward/mean", mean_r, iteration)
                writer.add_scalar("reward/min", min_r, iteration)
                writer.add_scalar("reward/max", max_r, iteration)
                writer.add_scalar("time/iteration_sec", iter_time, iteration)
                for phase_name, phase_value in iter_result.phase_times.items():
                    writer.add_scalar(
                        f"time/{phase_name}",
                        phase_value,
                        iteration,
                    )

            if writer is not None and torch.cuda.is_available():
                mem_mb = torch.cuda.memory_allocated() / 1024**2
                writer.add_scalar("device/memory_mb", mem_mb, iteration)

            self._print(
                f"[{iteration + 1}/{cfg.num_iterations}] "
                f"mean={mean_r:.4f} min={min_r:.4f} max={max_r:.4f} "
                f"time={iter_time:.1f}s"
            )

            # Check for preemption
            if (self.preemption_handler
                    and self.preemption_handler.check()):
                self._print(
                    f"Preemption detected at iteration {iteration + 1}. "
                    f"Saving emergency checkpoint..."
                )
                if self.is_master:
                    self._save_periodic_checkpoint(
                        checkpoint_dir, iteration + 1, history, training_start)
                self.population_executor.barrier(
                    f"preempted_iter_{iteration + 1}"
                )
                self._update_run_status("preempted", iteration + 1)
                self._print("Emergency checkpoint saved. Exiting.")
                if writer is not None:
                    writer.close()
                preempted = True
                break

            # Periodic checkpoint
            if (cfg.checkpoint_every > 0
                    and (iteration + 1) % cfg.checkpoint_every == 0
                    and (iteration + 1) < cfg.num_iterations):
                if self.is_master:
                    self._save_periodic_checkpoint(
                        checkpoint_dir, iteration + 1, history, training_start)
                self.population_executor.barrier(
                    f"checkpoint_iter_{iteration + 1}"
                )

        total_time = time.time() - training_start

        if preempted:
            sys.exit(130)  # 128 + SIGTERM(2) — distinct from normal exit

        self._print(
            f"Training complete in {total_time:.1f}s ({total_time / 3600:.1f}h)"
        )

        # Final checkpoint
        if self.is_master:
            self._save_checkpoint(checkpoint_dir)
        if self.is_master and self.gcs_client and self.experiment_name:
            final_path = os.path.join(checkpoint_dir, "es_checkpoint_final.pt")
            self.gcs_client.upload_checkpoint(
                final_path, self.experiment_name, self.method,
                self.run_name, "es_checkpoint_final.pt")
        if writer is not None:
            writer.close()
        self.population_executor.barrier("final_checkpoint")

        # Full evaluation on fine-tuned weights
        full_eval_result = None
        if self.is_master and self.full_eval_fn:
            self._print("\nRunning full evaluation (after training)...")
            eval_start = time.time()
            full_eval_result = self.full_eval_fn(self.model)
            eval_time = time.time() - eval_start
            full_eval_result["time_s"] = round(eval_time, 1)
            self._print(f"Full eval complete in {eval_time:.1f}s")
            for k, v in full_eval_result.get("scores", {}).items():
                self._print(f"  {_clean_task_name(k)}: {v:.2f}")

        # Save results
        benchmark_summary = None
        if cfg.benchmark_mode != "off":
            benchmark_summary = summarize_phase_history(
                phase_history,
                startup_time_s=self.startup_time_s,
                warmup_iterations=cfg.benchmark_warmup_iterations,
            )
            if self.is_master and benchmark_summary is not None:
                self._print("Benchmark summary:")
                self._print(
                    f"  startup={benchmark_summary['startup_time_s']:.2f}s "
                    f"warmup_iters={benchmark_summary['warmup_iterations']}"
                )
                for phase_name, median_value in sorted(
                        benchmark_summary["phase_median_s"].items()):
                    self._print(f"  median {phase_name}={median_value:.2f}s")
        if self.is_master:
            results_path = os.path.join(log_dir, "results.json")
            _save_results_json(
                results_path,
                config_dict,
                history,
                baseline_eval,
                full_eval_result,
                total_time,
                phase_history=phase_history,
                benchmark_summary=benchmark_summary,
            )

        # Capture Q/A examples
        if self.is_master and self.examples_fn:
            self._print("\nCapturing Q/A examples...")
            examples = self.examples_fn(self.model)
            examples_path = os.path.join(log_dir, "examples.json")
            with open(examples_path, "w") as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            self._print(f"  Saved {len(examples)} examples to {examples_path}")

        # Upload all artifacts to GCS
        if self.is_master and self.gcs_client and self.experiment_name:
            self._print("Uploading artifacts to GCS...")
            self.gcs_client.upload_run_artifacts(
                log_dir, self.experiment_name, self.method, self.run_name)
            self._update_run_status("completed")
            self._print("  GCS upload complete.")

        self._print("Done.")

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
        self._print(f"  Checkpoint saved: {path} ({size_mb:.1f} MB)")

    def _save_periodic_checkpoint(self, checkpoint_dir, iteration,
                                  history, training_start):
        """Save mid-training checkpoint locally and to GCS."""
        filename = f"es_checkpoint_iter{iteration:03d}.pt"
        local_path = os.path.join(checkpoint_dir, filename)

        # Save delta weights
        state = {"__format__": "delta"}
        all_params = dict(self.model.named_parameters())
        for name in self.param_names:
            delta = all_params[name].detach().cpu() - self._initial_params[name]
            state[name] = delta.to(torch.float16)
        torch.save(state, local_path)
        size_mb = os.path.getsize(local_path) / 1024**2

        # Save training state
        training_state = {
            "iteration": iteration,
            "history": history,
            "numpy_random_state": _serialize_numpy_state(
                np.random.get_state()),
            "wall_time_s": round(time.time() - training_start, 1),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        state_filename = f"training_state_iter{iteration:03d}.json"
        state_path = os.path.join(checkpoint_dir, state_filename)
        with open(state_path, "w") as f:
            json.dump(training_state, f)

        # Upload to GCS
        if self.gcs_client and self.experiment_name:
            try:
                self.gcs_client.upload_checkpoint(
                    local_path, self.experiment_name, self.method,
                    self.run_name, filename)
                self.gcs_client.upload_training_state(
                    training_state, self.experiment_name, self.method,
                    self.run_name, state_filename)
                self.gcs_client.cleanup_old_checkpoints(
                    self.experiment_name, self.method, self.run_name,
                    keep=2)
                # Upload TensorBoard events too
                log_dir = self.config.log_dir
                for f in glob.glob(
                        os.path.join(log_dir, "events.out.tfevents.*")):
                    self.gcs_client.upload_run_file(
                        f, self.experiment_name, self.method,
                        self.run_name, os.path.basename(f))
                self._print(
                    f"  Checkpoint iter {iteration} saved to GCS "
                    f"({size_mb:.1f} MB)"
                )
            except Exception as e:
                self._print(
                    f"  WARNING: GCS upload failed for iter {iteration}: "
                    f"{e}"
                )
                self._print(f"  Checkpoint saved locally: {local_path}")
        else:
            self._print(
                f"  Checkpoint iter {iteration} saved: {local_path} "
                f"({size_mb:.1f} MB)"
            )

        # Clean up local periodic checkpoints (keep only latest)
        for old in glob.glob(
                os.path.join(checkpoint_dir, "es_checkpoint_iter*.pt")):
            if old != local_path:
                os.remove(old)
        for old in glob.glob(
                os.path.join(checkpoint_dir, "training_state_iter*.json")):
            if old != state_path:
                os.remove(old)

    def _update_run_status(self, status, last_iter=None):
        """Update run status in GCS manifest."""
        if not self.gcs_client or not self.experiment_name:
            return

        run_key = f"{self.method}/{self.run_name}"

        def updater(manifest):
            exp = manifest["experiments"].get(self.experiment_name, {})
            runs = exp.setdefault("runs", {})
            run_info = runs.setdefault(run_key, {})
            run_info["status"] = status
            run_info[status] = datetime.datetime.now().isoformat()
            if last_iter is not None:
                run_info["last_checkpoint_iter"] = last_iter
            return manifest

        try:
            self.gcs_client.update_manifest(updater)
        except Exception as e:
            self._print(f"  WARNING: Failed to update manifest: {e}")


def _serialize_numpy_state(state):
    """Convert numpy random state tuple to JSON-serializable form."""
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def restore_numpy_state(state_dict):
    """Restore numpy random state from serialized form."""
    np.random.set_state((
        state_dict["name"],
        np.array(state_dict["keys"], dtype=np.uint32),
        state_dict["pos"],
        state_dict["has_gauss"],
        state_dict["cached_gaussian"],
    ))
