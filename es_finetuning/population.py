"""Population execution helpers for ES fine-tuning."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .noise import apply_es_update, perturb_weights, restore_weights


@dataclass
class IterationRunResult:
    """Outputs captured from one ES iteration."""

    rewards: np.ndarray
    phase_times: Dict[str, float]


def shard_population_indices(
    population_size: int,
    worker_count: int,
    worker_rank: int,
) -> List[int]:
    """Return the population-member indices assigned to one worker."""
    if population_size <= 0:
        return []
    if worker_count <= 0:
        raise ValueError(f"worker_count must be > 0, received {worker_count}")
    if worker_rank < 0 or worker_rank >= worker_count:
        raise ValueError(
            f"worker_rank must be in [0, {worker_count}), received {worker_rank}"
        )
    shards = np.array_split(np.arange(population_size, dtype=np.int64), worker_count)
    return shards[worker_rank].tolist()


def summarize_phase_history(
    phase_history: Dict[str, List[float]],
    *,
    startup_time_s: float = 0.0,
    warmup_iterations: int = 0,
) -> Dict[str, object] | None:
    """Summarize per-iteration timing history for benchmarking."""
    if not phase_history:
        return None

    iteration_times = phase_history.get("iteration_s", [])
    if not iteration_times:
        return None

    start_idx = min(max(int(warmup_iterations), 0), len(iteration_times))
    steady_history = {
        key: values[start_idx:]
        for key, values in phase_history.items()
        if values
    }

    summary = {
        "startup_time_s": round(float(startup_time_s), 6),
        "warmup_iterations": start_idx,
        "measured_iterations": len(iteration_times),
        "steady_state_iterations": len(iteration_times) - start_idx,
        "phase_median_s": {},
        "phase_mean_s": {},
    }

    for key, values in steady_history.items():
        if not values:
            continue
        summary["phase_median_s"][key] = round(float(np.median(values)), 6)
        summary["phase_mean_s"][key] = round(float(np.mean(values)), 6)

    return summary


class SingleProcessPopulationExecutor:
    """Default sequential ES population evaluator."""

    is_master = True

    def barrier(self, tag: str) -> None:
        del tag

    def broadcast_object(self, tag: str, value):
        del tag
        return value

    def run_iteration(
        self,
        *,
        model,
        param_names: Sequence[str],
        evaluate_fn,
        seeds: Sequence[int],
        sigma: float,
        alpha: float,
        noise_mode: str,
        population_size: int,
        iteration: int,
    ) -> IterationRunResult:
        del iteration
        iter_start = time.perf_counter()
        phase_times = {
            "perturb_s": 0.0,
            "evaluate_s": 0.0,
            "restore_s": 0.0,
            "sync_s": 0.0,
            "update_s": 0.0,
            "iteration_s": 0.0,
        }

        rewards = []
        for seed in seeds:
            start = time.perf_counter()
            perturb_weights(model, seed, sigma, param_names, noise_mode)
            phase_times["perturb_s"] += time.perf_counter() - start

            start = time.perf_counter()
            reward = evaluate_fn(model)
            phase_times["evaluate_s"] += time.perf_counter() - start
            rewards.append(reward)

            start = time.perf_counter()
            restore_weights(model, seed, sigma, param_names, noise_mode)
            phase_times["restore_s"] += time.perf_counter() - start

        rewards_arr = np.array(rewards, dtype=np.float32)
        normalized = (
            (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)
        )

        start = time.perf_counter()
        apply_es_update(
            model,
            seeds,
            normalized,
            sigma,
            alpha,
            param_names,
            population_size,
            noise_mode,
        )
        phase_times["update_s"] = time.perf_counter() - start
        phase_times["iteration_s"] = time.perf_counter() - iter_start

        return IterationRunResult(rewards=rewards_arr, phase_times=phase_times)


def _merge_indexed_rewards(reward_dicts):
    merged = {}
    for reward_dict in reward_dicts:
        merged.update(reward_dict)
    return merged


def _reduce_max_phase_times(phase_dicts):
    reduced = {}
    for phase_dict in phase_dicts:
        for key, value in phase_dict.items():
            reduced[key] = max(reduced.get(key, 0.0), float(value))
    return reduced


class ExactTpuPopulationExecutor:
    """Exact synchronous ES population execution across TPU workers."""

    def __init__(self, worker_rank: int, worker_count: int):
        if worker_count <= 1:
            raise ValueError(
                "ExactTpuPopulationExecutor requires worker_count > 1."
            )
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:  # pragma: no cover - TPU runtime only
            raise RuntimeError(
                "torch_xla is required for the TPU multichip executor."
            ) from exc

        self.worker_rank = worker_rank
        self.worker_count = worker_count
        self._xm = xm
        self.is_master = worker_rank == 0

    def barrier(self, tag: str) -> None:
        self._xm.rendezvous(tag)

    def broadcast_object(self, tag: str, value):
        return self._xm.mesh_reduce(
            tag,
            value,
            lambda values: next((item for item in values if item is not None), None),
        )

    def run_iteration(
        self,
        *,
        model,
        param_names: Sequence[str],
        evaluate_fn,
        seeds: Sequence[int],
        sigma: float,
        alpha: float,
        noise_mode: str,
        population_size: int,
        iteration: int,
    ) -> IterationRunResult:
        iter_start = time.perf_counter()
        phase_times = {
            "perturb_s": 0.0,
            "evaluate_s": 0.0,
            "restore_s": 0.0,
            "sync_s": 0.0,
            "update_s": 0.0,
            "iteration_s": 0.0,
        }

        local_rewards = {}
        local_indices = shard_population_indices(
            population_size,
            self.worker_count,
            self.worker_rank,
        )

        for member_idx in local_indices:
            seed = seeds[member_idx]

            start = time.perf_counter()
            perturb_weights(model, seed, sigma, param_names, noise_mode)
            phase_times["perturb_s"] += time.perf_counter() - start

            start = time.perf_counter()
            local_rewards[member_idx] = float(evaluate_fn(model))
            phase_times["evaluate_s"] += time.perf_counter() - start

            start = time.perf_counter()
            restore_weights(model, seed, sigma, param_names, noise_mode)
            phase_times["restore_s"] += time.perf_counter() - start

        start = time.perf_counter()
        merged_rewards = self._xm.mesh_reduce(
            f"es_rewards_iter_{iteration}",
            local_rewards,
            _merge_indexed_rewards,
        )
        phase_times["sync_s"] += time.perf_counter() - start

        rewards_arr = np.array(
            [merged_rewards[idx] for idx in range(population_size)],
            dtype=np.float32,
        )
        normalized = (
            (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)
        )

        start = time.perf_counter()
        apply_es_update(
            model,
            seeds,
            normalized,
            sigma,
            alpha,
            param_names,
            population_size,
            noise_mode,
        )
        phase_times["update_s"] = time.perf_counter() - start
        phase_times["iteration_s"] = time.perf_counter() - iter_start

        start = time.perf_counter()
        reduced_phase_times = self._xm.mesh_reduce(
            f"es_phase_iter_{iteration}",
            phase_times,
            _reduce_max_phase_times,
        )
        reduced_phase_times["sync_s"] = (
            float(reduced_phase_times.get("sync_s", 0.0))
            + (time.perf_counter() - start)
        )

        return IterationRunResult(
            rewards=rewards_arr,
            phase_times=reduced_phase_times,
        )
