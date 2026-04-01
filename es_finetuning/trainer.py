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

import numpy as np
import torch
import wandb

from .config import ESConfig
from .noise import apply_es_update, perturb_weights, restore_weights
from .utils import force_memory_cleanup


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
                 experiment_name=None, method=None, run_name=None):
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
                print(f"Resuming from iteration {start_iteration} "
                      f"(checkpoint: {self.resume_from}, "
                      f"numpy state restored)")
            else:
                # Fallback: seed and replay RNG to approximate state.
                np.random.seed(cfg.initial_seed)
                for i in range(start_iteration):
                    np.random.randint(0, 2**30, size=cfg.population_size,
                                      dtype=np.int64)
                print(f"Resuming from iteration {start_iteration} "
                      f"(checkpoint: {self.resume_from}, "
                      f"seed replay)")
        else:
            np.random.seed(cfg.initial_seed)

        # Setup directories
        log_dir = cfg.log_dir
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save full config
        config_dict = asdict(cfg)
        config_dict.update(self.metadata)
        config_path = os.path.join(log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Config saved to {config_path}")

        # Upload config to GCS
        if self.gcs_client and self.experiment_name:
            self.gcs_client.upload_run_file(
                config_path, self.experiment_name, self.method, self.run_name,
                "config.json")

        # Init wandb
        if cfg.wandb_log:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name or self.run_name,
                group=cfg.wandb_group_name,
                config=config_dict,
                resume="allow",
            )

        total_evals = cfg.population_size * cfg.num_iterations
        meta = self.metadata or {}
        print()
        print("=" * 60)
        print("FAIR-01 Training Summary  [ES]")
        print("-" * 60)
        print(f"  method            : {self.method}")
        print(f"  run_name          : {self.run_name}")
        print(f"  device            : {next(self.model.parameters()).device}")
        print("  -- Compute --")
        print(f"  population_size   : {cfg.population_size}")
        print(f"  num_iterations    : {cfg.num_iterations}")
        print(f"  total_evaluations : {total_evals}")
        print(f"  sigma             : {cfg.sigma}")
        print(f"  alpha             : {cfg.alpha}")
        print(f"  noise_mode        : {cfg.noise_mode}")
        print(f"  mini_batch_size   : {cfg.mini_batch_size}")
        print(f"  num_parameters    : {len(self.param_names)}")
        print("  -- Data --")
        print(f"  train_split       : {meta.get('train_split', 'N/A')}")
        print(f"  split_seed        : {meta.get('split_seed', 'N/A')}")
        print("  -- Decoding --")
        print(f"  temperature       : {cfg.temperature}")
        print(f"  eval_temperature  : {cfg.eval_temperature}")
        print(f"  num_samples       : {cfg.num_samples}")
        print(f"  eval_num_samples  : {cfg.eval_num_samples}")
        print("  -- Logging --")
        print(f"  log_dir           : {log_dir}")
        if cfg.checkpoint_every > 0:
            print(f"  checkpoint_every  : {cfg.checkpoint_every}")
        if cfg.save_every > 0:
            print(f"  save_every        : {cfg.save_every}")
        if self.gcs_client:
            print(f"  gcs               : gs://{self.gcs_client.bucket_name}/")
        print("=" * 60)
        print()

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
            if cfg.wandb_log and wandb.run is not None:
                wandb.log(
                    {f"eval/baseline_{_clean_task_name(k)}": v
                     for k, v in baseline_eval.get("scores", {}).items()},
                    step=0,
                )

        # Training loop
        if resumed_history:
            history = resumed_history
        else:
            history = {"mean": [], "min": [], "max": [], "time": []}
        training_start = time.time()
        preempted = False

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

            if cfg.wandb_log and wandb.run is not None:
                log_dict = {
                    "reward/mean": mean_r,
                    "reward/min": min_r,
                    "reward/max": max_r,
                    "reward/std": rewards_arr.std().item(),
                    "reward/range": (max_r - min_r),
                    "time/iteration_sec": iter_time,
                    "time/evals_per_sec": cfg.population_size / iter_time,
                }
                if torch.cuda.is_available():
                    log_dict["device/memory_mb"] = (
                        torch.cuda.memory_allocated() / 1024**2
                    )
                wandb.log(log_dict, step=iteration)

            print(f"[{iteration + 1}/{cfg.num_iterations}] "
                  f"mean={mean_r:.4f} min={min_r:.4f} max={max_r:.4f} "
                  f"time={iter_time:.1f}s")

            # Check for preemption
            if (self.preemption_handler
                    and self.preemption_handler.check()):
                print(f"Preemption detected at iteration {iteration + 1}. "
                      f"Saving emergency checkpoint...")
                self._save_periodic_checkpoint(
                    checkpoint_dir, iteration + 1, history, training_start)
                self._update_run_status("preempted", iteration + 1)
                print("Emergency checkpoint saved. Exiting.")
                if cfg.wandb_log and wandb.run is not None:
                    wandb.finish(exit_code=1)
                preempted = True
                break

            # Permanent save (kept forever, never cleaned up)
            if (cfg.save_every > 0
                    and (iteration + 1) % cfg.save_every == 0
                    and (iteration + 1) < cfg.num_iterations):
                self._save_permanent_checkpoint(
                    checkpoint_dir, iteration + 1)

            # Periodic checkpoint (rolling, keep last 2)
            if (cfg.checkpoint_every > 0
                    and (iteration + 1) % cfg.checkpoint_every == 0
                    and (iteration + 1) < cfg.num_iterations):
                self._save_periodic_checkpoint(
                    checkpoint_dir, iteration + 1, history, training_start)

        total_time = time.time() - training_start

        if preempted:
            sys.exit(130)  # 128 + SIGTERM(2) — distinct from normal exit

        print(f"Training complete in {total_time:.1f}s ({total_time / 3600:.1f}h)")

        # Final checkpoint
        self._save_checkpoint(checkpoint_dir, config_path)
        if self.gcs_client and self.experiment_name:
            final_path = os.path.join(checkpoint_dir, "es_checkpoint_final.pt")
            self.gcs_client.upload_checkpoint(
                final_path, self.experiment_name, self.method,
                self.run_name, "es_checkpoint_final.pt")

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
            if cfg.wandb_log and wandb.run is not None:
                wandb.log(
                    {f"eval/final_{_clean_task_name(k)}": v
                     for k, v in full_eval_result.get("scores", {}).items()},
                    step=cfg.num_iterations,
                )

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

        # Upload all artifacts to GCS
        if self.gcs_client and self.experiment_name:
            print("Uploading artifacts to GCS...")
            self.gcs_client.upload_run_artifacts(
                log_dir, self.experiment_name, self.method, self.run_name)
            self._update_run_status("completed")
            print("  GCS upload complete.")

        if cfg.wandb_log and wandb.run is not None:
            wandb.finish()

        print("Done.")

    def _save_permanent_checkpoint(self, checkpoint_dir, iteration):
        """Save a permanent checkpoint that is never cleaned up.

        Stored in checkpoints/saved/ to keep them separate from the
        rolling periodic checkpoints.
        """
        saved_dir = os.path.join(checkpoint_dir, "saved")
        os.makedirs(saved_dir, exist_ok=True)
        filename = f"es_checkpoint_iter{iteration:03d}.pt"
        local_path = os.path.join(saved_dir, filename)

        state = {"__format__": "delta"}
        all_params = dict(self.model.named_parameters())
        for name in self.param_names:
            delta = all_params[name].detach().cpu() - self._initial_params[name]
            state[name] = delta.to(torch.float16)
        torch.save(state, local_path)
        size_mb = os.path.getsize(local_path) / 1024**2

        if self.gcs_client and self.experiment_name:
            gcs_path = (
                f"{self.gcs_client._run_prefix(self.experiment_name, self.method, self.run_name)}"
                f"/checkpoints/saved/{filename}")
            try:
                self.gcs_client.upload_file(local_path, gcs_path)
                print(f"  Permanent save iter {iteration} -> GCS ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  WARNING: GCS upload failed for permanent save "
                      f"iter {iteration}: {e}")
                print(f"  Saved locally: {local_path}")
        else:
            print(f"  Permanent save iter {iteration}: {local_path} "
                  f"({size_mb:.1f} MB)")

    def _save_checkpoint(self, checkpoint_dir, config_path=None):
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

        if self.config.wandb_log and wandb.run is not None:
            print("  Uploading final checkpoint and config as wandb artifact...")
            artifact = wandb.Artifact(
                name=f"run-{wandb.run.id}-es-final",
                type="model",
                description="Final ES fine-tuned delta weights",
                metadata={"num_iterations": self.config.num_iterations},
            )
            artifact.add_file(path)
            if config_path and os.path.exists(config_path):
                artifact.add_file(config_path)
            wandb.log_artifact(artifact, aliases=["final"])

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
                print(f"  Checkpoint iter {iteration} saved to GCS "
                      f"({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  WARNING: GCS upload failed for iter {iteration}: "
                      f"{e}")
                print(f"  Checkpoint saved locally: {local_path}")
        else:
            print(f"  Checkpoint iter {iteration} saved: {local_path} "
                  f"({size_mb:.1f} MB)")

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
            print(f"  WARNING: Failed to update manifest: {e}")


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
