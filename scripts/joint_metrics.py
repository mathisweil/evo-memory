"""Metrics tracking for joint NAMM + adapter training (``run_joint.py``).

Collects NAMM, adapter, and cross-component statistics at each stage
boundary of the alternating training loop, writes them to
``results_detailed.json``, optionally forwards them to WandB, and renders
diagnostic figures to ``{run_dir}/figures/``.

Design goals:
  * Minimal invasion in ``run_joint.py`` — the tracker only needs the
    already-available objects (CMA-ES state, LoRA trainer output dir,
    ``memory_policy``, ``memory_model``).
  * Cheap defaults — per-iteration CMA-ES fitness is captured by
    wrapping ``MemoryTrainer._train_step`` for the duration of one stage,
    and the post-stage retention probe runs on <=8 samples.
  * Expensive cross-component diagnostics (LoRA adapter-disabled eval
    via PEFT's ``disable_adapter()`` context) are opt-in via
    ``detailed_diagnostics``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_DIAGNOSTIC_SAMPLES: int = 8
_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)


# ── Plot styling (kept in sync with generate_paper_figures.py) ────────────────


def _apply_style() -> None:
    """Apply the paper-figure matplotlib style."""
    if not _HAS_MPL:
        return
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _save_fig(fig: "plt.Figure", stem: str, out_dir: str) -> None:
    """Save ``fig`` as PNG and PDF under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── Data containers ───────────────────────────────────────────────────────────


@dataclass
class NammStageMetrics:
    """Metrics collected after one NAMM (Stage A) run."""

    outer_loop: int
    time_s: float
    best_fitness: float
    sigma: float
    param_l2: float
    param_size: int
    fitness_trajectory: List[Dict[str, float]] = field(default_factory=list)
    evo_stats: Dict[str, float] = field(default_factory=dict)
    retention: Dict[str, float] = field(default_factory=dict)
    retention_per_layer: List[float] = field(default_factory=list)
    eviction_rate: Optional[float] = None
    token_score_percentiles: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdapterStageMetrics:
    """Metrics collected after one adapter (Stage B) run."""

    outer_loop: int
    adapter_type: str
    time_s: float
    loss_history: List[Dict[str, float]] = field(default_factory=list)
    final_loss: Optional[float] = None
    final_grad_norm: Optional[float] = None
    val_f1: Optional[float] = None
    lora_weight_norms: Dict[str, float] = field(default_factory=dict)
    es_fitness_history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class OuterLoopMetrics:
    """Aggregated metrics for a single outer-loop iteration."""

    outer_loop: int
    namm: Optional[NammStageMetrics] = None
    adapter: Optional[AdapterStageMetrics] = None
    eval_scores: Dict[str, float] = field(default_factory=dict)
    eval_scores_adapter_only: Optional[Dict[str, float]] = None
    eval_time_s: float = 0.0
    total_time_s: float = 0.0


# ── Helpers for pulling metrics from runtime objects ──────────────────────────


def _cma_es_best_fitness(evolution_algorithm) -> float:
    """Return the best fitness recorded by the CMA-ES state, as a float."""
    bf = getattr(evolution_algorithm, "best_fitness", None)
    if bf is None:
        return float("nan")
    try:
        return float(bf.item())
    except AttributeError:
        return float(bf)


def _cma_es_sigma(evolution_algorithm) -> float:
    """Return the CMA-ES step size ``sigma``."""
    sigma = getattr(evolution_algorithm, "sigma", None)
    if sigma is None:
        return float("nan")
    try:
        return float(sigma.item())
    except AttributeError:
        return float(sigma)


def _cma_es_param_l2(evolution_algorithm) -> tuple[float, int]:
    """Return (L2 norm of best_member, param_size)."""
    best = getattr(evolution_algorithm, "best_member", None)
    if best is None:
        return float("nan"), 0
    with torch.no_grad():
        return float(torch.linalg.vector_norm(best).item()), int(best.numel())


def _collect_lora_weight_norms(memory_model) -> Dict[str, float]:
    """Compute L2 weight norms of LoRA parameters, grouped by target module.

    Groups by the PEFT target module identifier embedded in the parameter
    name, e.g. ``q_proj``/``v_proj``. Returns an empty dict if LoRA is not
    applied.
    """
    inner = memory_model.model if hasattr(memory_model, "model") else memory_model
    norms: Dict[str, List[float]] = {}
    with torch.no_grad():
        for name, p in inner.named_parameters():
            if "lora_" not in name:
                continue
            module_tag = "other"
            for cand in getattr(memory_model, "_lora_target_modules", []) or []:
                if cand in name:
                    module_tag = cand
                    break
            norms.setdefault(module_tag, []).append(
                float(torch.linalg.vector_norm(p.detach()).item())
            )
    return {k: float(np.sqrt(np.sum(np.square(v)))) for k, v in norms.items()}


def _parse_csv(path: str) -> List[Dict[str, float]]:
    """Parse a numeric-valued CSV into a list of dicts."""
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            parsed: Dict[str, float] = {}
            for k, v in raw.items():
                if k is None or v is None or v == "":
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


# ── Fitness capture (CMA-ES per-iteration population trajectory) ──────────────


class _FitnessCapture:
    """Temporary wrapper for ``MemoryTrainer._train_step`` that records
    per-iteration population statistics without touching MemoryTrainer."""

    def __init__(self, namm_trainer):
        self.trainer = namm_trainer
        self.original = namm_trainer._train_step
        self.records: List[Dict[str, float]] = []

    def __enter__(self) -> "_FitnessCapture":
        def wrapped():
            scores, score_dicts = self.original()
            arr = np.asarray(scores, dtype=np.float64)
            if arr.size > 0:
                self.records.append(
                    {
                        "mean": float(arr.mean()),
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        "std": float(arr.std()),
                    }
                )
            return scores, score_dicts

        self.trainer._train_step = wrapped
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.trainer._train_step = self.original


# ── Diagnostic eviction probe ─────────────────────────────────────────────────


def _probe_retention(
    memory_policy,
    memory_evaluator,
    task_sampler,
    num_samples: int = _DIAGNOSTIC_SAMPLES,
) -> Dict[str, Any]:
    """Run a small eval batch with stat recording enabled and return a
    summary of retained-token counts per layer.

    The probe flips ``record_eval_stats``/``record_mask_based_sparsity`` on
    the policy, calls ``initialize_stat_objects``, runs the evaluator on
    ``num_samples`` questions, then reads ``get_param_stats``. The
    recording flags are restored to their prior values on exit.
    """
    if memory_policy is None or memory_evaluator is None or task_sampler is None:
        return {}

    prev_eval = memory_policy.record_eval_stats
    prev_mask = getattr(memory_policy, "record_mask_based_sparsity", False)
    memory_policy.record_eval_stats = True
    if hasattr(memory_policy, "_record_mask_based_sparsity"):
        memory_policy.record_mask_based_sparsity = True
    try:
        memory_policy.initialize_stat_objects()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("initialize_stat_objects failed: %s", exc)
        return {}

    try:
        with torch.no_grad():
            task_sampler.evaluate(
                lm=memory_evaluator,
                train=False,
                evolved_model=False,
                pop_reps=1,
                resample_requests=True,
                sampled_requests_per_task=num_samples,
                model_kwargs={},
            )
    except Exception as exc:
        logger.warning("Retention probe eval failed: %s", exc)
        memory_policy.record_eval_stats = prev_eval
        if hasattr(memory_policy, "_record_mask_based_sparsity"):
            memory_policy.record_mask_based_sparsity = prev_mask
        return {}

    stats = memory_policy.get_param_stats(reset=False)
    memory_policy.record_eval_stats = prev_eval
    if hasattr(memory_policy, "_record_mask_based_sparsity"):
        memory_policy.record_mask_based_sparsity = prev_mask

    summary: Dict[str, Any] = {}
    num_layers = int(getattr(memory_policy, "num_memory_layers", 0) or 0)
    per_layer: List[float] = []
    cache_size = getattr(memory_policy, "cache_size", None)
    for i in range(num_layers):
        prefix = f"mem_stats/layer_id_{i}/"
        key = prefix + "unmasked_samples"
        if key in stats:
            per_layer.append(float(stats[key]))
    overall_unmasked = stats.get("mem_stats/overall/unmasked_sample_final")
    summary["per_layer_unmasked"] = per_layer
    summary["overall_unmasked"] = (
        float(overall_unmasked) if overall_unmasked is not None else None
    )
    if cache_size and cache_size > 0 and per_layer:
        ratios = [v / float(cache_size) for v in per_layer]
        summary["per_layer_retention_ratio"] = ratios
        summary["mean_retention_ratio"] = float(np.mean(ratios))
    elif per_layer:
        summary["per_layer_retention_ratio"] = []
        summary["mean_retention_ratio"] = None

    if overall_unmasked is not None and cache_size and cache_size > 0:
        summary["eviction_rate"] = max(
            0.0, 1.0 - float(overall_unmasked) / float(cache_size)
        )
    return summary


# ── Disable adapter context (for co-adaptation eval) ──────────────────────────


@contextmanager
def _disable_lora_adapter(memory_model):
    """Yield a context where the LoRA adapter is bypassed.

    Uses PEFT's ``disable_adapter`` context manager when available. Falls
    back to a no-op (with a warning) if PEFT has no such attribute.
    """
    inner = memory_model.model if hasattr(memory_model, "model") else memory_model
    disable = getattr(inner, "disable_adapter", None)
    if disable is None:
        logger.warning("disable_adapter not available on model; running no-op")
        yield
        return
    with disable():
        yield


# ── Tracker ───────────────────────────────────────────────────────────────────


class JointMetricsTracker:
    """Collect and persist joint-training diagnostics.

    Usage (from ``run_joint.py``):

        tracker = JointMetricsTracker(
            output_dir=run_dir,
            wandb_run=None,
            detailed_diagnostics=args.detailed_diagnostics,
        )
        for k in range(num_outer_loops):
            with tracker.capture_namm_fitness(namm_trainer):
                namm_trainer.train()
            tracker.log_namm_stage_end(
                outer_loop=k,
                evolution_algorithm=evolution_algorithm,
                memory_policy=memory_policy,
                memory_evaluator=memory_evaluator,
                task_sampler=task_sampler,
                stage_time_s=namm_time,
            )
            ...
            tracker.log_adapter_stage_end(
                outer_loop=k,
                adapter_type=args.adapter_type,
                memory_model=memory_model,
                stage_dir=adapter_stage_dir,
                stage_time_s=adapter_time,
            )
            tracker.log_outer_loop_end(
                outer_loop=k,
                eval_scores=eval_scores,
                eval_time_s=eval_time,
                adapter_disabled_scores=None,
                total_time_s=total_time,
            )
        tracker.finalize()
    """

    def __init__(
        self,
        output_dir: str,
        wandb_run: Optional[Any] = None,
        detailed_diagnostics: bool = False,
        retention_probe_samples: int = _DIAGNOSTIC_SAMPLES,
    ) -> None:
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.wandb_run = wandb_run if _HAS_WANDB else None
        self.detailed_diagnostics = detailed_diagnostics
        self.retention_probe_samples = retention_probe_samples

        self.history: List[OuterLoopMetrics] = []
        self._current: Optional[OuterLoopMetrics] = None
        self._pending_fitness_capture: Optional[_FitnessCapture] = None
        self._start_time = time.time()

    # -- Public API --------------------------------------------------------

    def capture_namm_fitness(self, namm_trainer) -> _FitnessCapture:
        """Return a context manager that records per-iteration CMA-ES
        population fitness while ``namm_trainer.train()`` runs."""
        self._pending_fitness_capture = _FitnessCapture(namm_trainer)
        return self._pending_fitness_capture

    def log_namm_stage_end(
        self,
        outer_loop: int,
        evolution_algorithm,
        memory_policy,
        memory_evaluator,
        task_sampler,
        stage_time_s: float,
    ) -> NammStageMetrics:
        """Pull metrics from the CMA-ES state + policy and stash them."""
        param_l2, param_size = _cma_es_param_l2(evolution_algorithm)
        evo_stats = {}
        try:
            evo_stats = {
                k: float(v) for k, v in evolution_algorithm.get_stats().items()
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("evolution_algorithm.get_stats() failed: %s", exc)

        trajectory: List[Dict[str, float]] = []
        if self._pending_fitness_capture is not None:
            trajectory = self._pending_fitness_capture.records
            self._pending_fitness_capture = None

        retention_summary = _probe_retention(
            memory_policy=memory_policy,
            memory_evaluator=memory_evaluator,
            task_sampler=task_sampler,
            num_samples=self.retention_probe_samples,
        )

        metrics = NammStageMetrics(
            outer_loop=outer_loop,
            time_s=round(stage_time_s, 1),
            best_fitness=_cma_es_best_fitness(evolution_algorithm),
            sigma=_cma_es_sigma(evolution_algorithm),
            param_l2=param_l2,
            param_size=param_size,
            fitness_trajectory=trajectory,
            evo_stats=evo_stats,
            retention={
                "mean_retention_ratio": retention_summary.get("mean_retention_ratio"),
                "overall_unmasked": retention_summary.get("overall_unmasked"),
                "eviction_rate": retention_summary.get("eviction_rate"),
            },
            retention_per_layer=retention_summary.get("per_layer_retention_ratio", []),
            eviction_rate=retention_summary.get("eviction_rate"),
        )

        self._current = OuterLoopMetrics(outer_loop=outer_loop, namm=metrics)

        self._log_wandb(
            {
                "namm/best_fitness": metrics.best_fitness,
                "namm/sigma": metrics.sigma,
                "namm/param_l2": metrics.param_l2,
                "namm/time_s": metrics.time_s,
                "namm/mean_retention_ratio": metrics.retention.get(
                    "mean_retention_ratio"
                ),
                "namm/eviction_rate": metrics.eviction_rate,
                "joint/outer_loop": outer_loop,
            }
        )
        if trajectory:
            tail = trajectory[-1]
            self._log_wandb(
                {
                    "namm/pop_fitness_mean": tail["mean"],
                    "namm/pop_fitness_min": tail["min"],
                    "namm/pop_fitness_max": tail["max"],
                    "namm/pop_fitness_std": tail["std"],
                }
            )
        return metrics

    def log_adapter_stage_end(
        self,
        outer_loop: int,
        adapter_type: str,
        memory_model,
        stage_dir: str,
        stage_time_s: float,
    ) -> AdapterStageMetrics:
        """Collect per-step loss/grad history + weight norms."""
        loss_history: List[Dict[str, float]] = []
        final_loss: Optional[float] = None
        final_grad_norm: Optional[float] = None
        val_f1: Optional[float] = None
        es_history: List[Dict[str, float]] = []
        lora_norms: Dict[str, float] = {}

        if adapter_type == "lora":
            metrics_csv = os.path.join(stage_dir, "metrics.csv")
            rows = _parse_csv(metrics_csv)
            loss_history = rows
            if rows:
                final_loss = rows[-1].get("loss")
                final_grad_norm = rows[-1].get("grad_norm")

            val_csv = os.path.join(stage_dir, "val_metrics.csv")
            val_rows = _parse_csv(val_csv)
            if val_rows:
                f1_keys = [k for k in val_rows[-1] if k.startswith("val_lb_avg_f1")]
                if f1_keys:
                    val_f1 = val_rows[-1][f1_keys[0]]

            lora_norms = _collect_lora_weight_norms(memory_model)

        elif adapter_type == "es":
            results_path = os.path.join(stage_dir, "results.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                    reward = data.get("training", {}).get("reward_per_iteration", {})
                    means = reward.get("mean", [])
                    mins = reward.get("min", [])
                    maxs = reward.get("max", [])
                    for i, m in enumerate(means):
                        es_history.append(
                            {
                                "iter": i,
                                "mean": float(m),
                                "min": float(mins[i]) if i < len(mins) else float("nan"),
                                "max": float(maxs[i]) if i < len(maxs) else float("nan"),
                            }
                        )
                    if means:
                        final_loss = float(means[-1])
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to read ES results %s: %s", results_path, exc)
        else:
            logger.warning("Unknown adapter_type %r", adapter_type)

        metrics = AdapterStageMetrics(
            outer_loop=outer_loop,
            adapter_type=adapter_type,
            time_s=round(stage_time_s, 1),
            loss_history=loss_history,
            final_loss=final_loss,
            final_grad_norm=final_grad_norm,
            val_f1=val_f1,
            lora_weight_norms=lora_norms,
            es_fitness_history=es_history,
        )

        if self._current is None:
            self._current = OuterLoopMetrics(outer_loop=outer_loop, adapter=metrics)
        else:
            self._current.adapter = metrics

        wandb_payload: Dict[str, Any] = {"adapter/time_s": metrics.time_s}
        if final_loss is not None:
            wandb_payload["adapter/final_loss"] = final_loss
        if final_grad_norm is not None:
            wandb_payload["adapter/final_grad_norm"] = final_grad_norm
        if val_f1 is not None:
            wandb_payload["adapter/val_f1"] = val_f1
        for tag, val in lora_norms.items():
            wandb_payload[f"adapter/lora_norm_{tag}"] = val
        self._log_wandb(wandb_payload)
        return metrics

    def log_outer_loop_end(
        self,
        outer_loop: int,
        eval_scores: Dict[str, float],
        eval_time_s: float,
        adapter_disabled_scores: Optional[Dict[str, float]] = None,
        total_time_s: float = 0.0,
    ) -> None:
        """Append the finished outer-loop entry and persist the history."""
        if self._current is None:
            self._current = OuterLoopMetrics(outer_loop=outer_loop)

        self._current.eval_scores = {k: float(v) for k, v in eval_scores.items()}
        self._current.eval_scores_adapter_only = (
            {k: float(v) for k, v in adapter_disabled_scores.items()}
            if adapter_disabled_scores
            else None
        )
        self._current.eval_time_s = round(eval_time_s, 1)
        self._current.total_time_s = round(total_time_s, 1)

        self.history.append(self._current)
        self._current = None

        payload: Dict[str, Any] = {"joint/eval_time_s": eval_time_s}
        for k, v in eval_scores.items():
            payload[f"joint/eval_{k.replace('/', '_')}"] = float(v)
        if adapter_disabled_scores:
            for k, v in adapter_disabled_scores.items():
                payload[f"joint/adapter_off_{k.replace('/', '_')}"] = float(v)
        self._log_wandb(payload)

        self._dump_json()

    def evaluate_adapter_disabled(
        self,
        memory_model,
        full_eval_fn: Callable,
    ) -> Optional[Dict[str, float]]:
        """Run ``full_eval_fn`` with LoRA adapters bypassed via PEFT.

        Only meaningful when LoRA is applied; for the ES path we simply
        return ``None``. Gated by ``detailed_diagnostics`` in the caller.
        """
        if not self.detailed_diagnostics:
            return None
        if not getattr(memory_model, "has_lora_adapters", lambda: False)():
            return None
        try:
            with _disable_lora_adapter(memory_model), torch.no_grad():
                result = full_eval_fn(memory_model)
            return {k: float(v) for k, v in result.get("scores", {}).items()}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("adapter-disabled eval failed: %s", exc)
            return None

    def finalize(self) -> None:
        """Write the final JSON and render all figures."""
        self._dump_json()
        if _HAS_MPL:
            _apply_style()
            self._render_figures()
        else:
            logger.warning("matplotlib not available; skipping figure generation")

    # -- Internals ---------------------------------------------------------

    def _log_wandb(self, payload: Dict[str, Any]) -> None:
        if self.wandb_run is None or not _HAS_WANDB:
            return
        cleaned = {k: v for k, v in payload.items() if v is not None}
        if cleaned:
            try:
                wandb.log(cleaned)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("wandb.log failed: %s", exc)

    def _dump_json(self) -> None:
        path = os.path.join(self.output_dir, "results_detailed.json")
        serializable = {
            "history": [self._entry_to_dict(e) for e in self.history],
            "wall_time_s": round(time.time() - self._start_time, 1),
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    @staticmethod
    def _entry_to_dict(entry: OuterLoopMetrics) -> Dict[str, Any]:
        def dc_to_dict(obj):
            if obj is None:
                return None
            if hasattr(obj, "__dict__"):
                return {k: v for k, v in obj.__dict__.items()}
            return obj

        return {
            "outer_loop": entry.outer_loop,
            "namm": dc_to_dict(entry.namm),
            "adapter": dc_to_dict(entry.adapter),
            "eval_scores": entry.eval_scores,
            "eval_scores_adapter_only": entry.eval_scores_adapter_only,
            "eval_time_s": entry.eval_time_s,
            "total_time_s": entry.total_time_s,
        }

    # -- Figures -----------------------------------------------------------

    def _render_figures(self) -> None:
        if not self.history:
            logger.warning("No history recorded; skipping figures")
            return
        self._fig_learning_curve()
        self._fig_retention()
        self._fig_cma_convergence()
        self._fig_namm_param_norm()
        self._fig_lora_norms()
        self._fig_loss_curve()
        self._fig_co_adaptation()

    def _outer_loops(self) -> List[int]:
        return [e.outer_loop + 1 for e in self.history]

    def _fig_learning_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        loops = self._outer_loops()
        task_names: List[str] = sorted(
            {k for e in self.history for k in e.eval_scores}
        )
        if not task_names:
            plt.close(fig)
            return
        for task in task_names:
            ys = [e.eval_scores.get(task, np.nan) for e in self.history]
            ax.plot(loops, ys, marker="o", label=task.split("/")[-1])
        ax.set_xlabel("Outer loop")
        ax.set_ylabel("F1 / accuracy")
        ax.set_title("Evaluation performance across outer loops")
        ax.legend(loc="best")
        _save_fig(fig, "learning_curve", self.figures_dir)

    def _fig_retention(self) -> None:
        loops = self._outer_loops()
        mean_ret = [
            (e.namm.retention.get("mean_retention_ratio")
             if e.namm and e.namm.retention else None)
            for e in self.history
        ]
        if all(v is None for v in mean_ret):
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ys = [np.nan if v is None else v for v in mean_ret]
        ax.plot(loops, ys, marker="o", color="#1a9850")
        ax.set_xlabel("Outer loop")
        ax.set_ylabel("Mean retention ratio (unmasked / cache_size)")
        ax.set_title("NAMM cache retention across outer loops")
        ax.set_ylim(0, None)
        _save_fig(fig, "retention", self.figures_dir)

    def _fig_cma_convergence(self) -> None:
        flat_means: List[float] = []
        flat_stds: List[float] = []
        boundaries: List[int] = []
        cursor = 0
        for e in self.history:
            if e.namm is None:
                continue
            for rec in e.namm.fitness_trajectory:
                flat_means.append(rec["mean"])
                flat_stds.append(rec["std"])
            cursor += len(e.namm.fitness_trajectory)
            boundaries.append(cursor)
        if not flat_means:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = np.arange(1, len(flat_means) + 1)
        means = np.asarray(flat_means)
        stds = np.asarray(flat_stds)
        ax.plot(xs, means, color="#2166ac", label="population mean")
        ax.fill_between(xs, means - stds, means + stds,
                        color="#2166ac", alpha=0.2, label="±1 std")
        for b in boundaries[:-1]:
            ax.axvline(b + 0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("CMA-ES generation (concatenated across outer loops)")
        ax.set_ylabel("Population fitness")
        ax.set_title("CMA-ES population fitness across outer loops")
        ax.legend(loc="best")
        _save_fig(fig, "cma_convergence", self.figures_dir)

    def _fig_namm_param_norm(self) -> None:
        loops = self._outer_loops()
        norms = [
            e.namm.param_l2 if e.namm else np.nan for e in self.history
        ]
        sigmas = [
            e.namm.sigma if e.namm else np.nan for e in self.history
        ]
        if all(np.isnan(norms)):
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(loops, norms, marker="o", color="#762a83", label="NAMM best_member L2")
        ax.set_xlabel("Outer loop")
        ax.set_ylabel("L2 norm", color="#762a83")
        ax.tick_params(axis="y", labelcolor="#762a83")

        ax2 = ax.twinx()
        ax2.plot(loops, sigmas, marker="s", color="#f46d43", label="CMA-ES sigma")
        ax2.set_ylabel("sigma", color="#f46d43")
        ax2.tick_params(axis="y", labelcolor="#f46d43")

        ax.set_title("NAMM parameter norm & CMA-ES sigma over outer loops")
        _save_fig(fig, "namm_param_trajectory", self.figures_dir)

    def _fig_lora_norms(self) -> None:
        loops = self._outer_loops()
        all_tags: List[str] = sorted(
            {
                tag
                for e in self.history
                if e.adapter is not None
                for tag in e.adapter.lora_weight_norms
            }
        )
        if not all_tags:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        for tag in all_tags:
            ys = [
                (e.adapter.lora_weight_norms.get(tag, np.nan)
                 if e.adapter else np.nan)
                for e in self.history
            ]
            ax.plot(loops, ys, marker="o", label=tag)
        ax.set_xlabel("Outer loop")
        ax.set_ylabel("LoRA weight L2 norm")
        ax.set_title("LoRA weight norms over outer loops")
        ax.legend(loc="best")
        _save_fig(fig, "lora_norms", self.figures_dir)

    def _fig_loss_curve(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        cursor = 0
        has_data = False
        for e in self.history:
            if e.adapter is None:
                continue
            rows = e.adapter.loss_history
            if not rows:
                continue
            if "loss" in rows[0]:
                ys = [r.get("loss", np.nan) for r in rows]
                label = (
                    f"Loop {e.outer_loop + 1}" if e.outer_loop == 0 else None
                )
                xs = np.arange(cursor + 1, cursor + 1 + len(ys))
                ax.plot(xs, ys, color="#2166ac", alpha=0.8)
                cursor += len(ys)
                has_data = True
            elif "mean" in rows[0]:
                ys = [r.get("mean", np.nan) for r in rows]
                xs = np.arange(cursor + 1, cursor + 1 + len(ys))
                ax.plot(xs, ys, color="#f46d43", alpha=0.8)
                cursor += len(ys)
                has_data = True
            if cursor > 0:
                ax.axvline(cursor + 0.5, color="gray", linestyle="--", alpha=0.4)
        if not has_data:
            plt.close(fig)
            return
        ax.set_xlabel("Step (concatenated across adapter stages)")
        ax.set_ylabel("Training loss / ES fitness")
        ax.set_title("Adapter training trajectory across outer loops")
        _save_fig(fig, "adapter_loss_curve", self.figures_dir)

    def _fig_co_adaptation(self) -> None:
        loops = self._outer_loops()
        full_scores: List[float] = []
        adapter_off_scores: List[float] = []
        for e in self.history:
            full_scores.append(
                float(np.mean(list(e.eval_scores.values()))) if e.eval_scores else np.nan
            )
            if e.eval_scores_adapter_only:
                adapter_off_scores.append(
                    float(np.mean(list(e.eval_scores_adapter_only.values())))
                )
            else:
                adapter_off_scores.append(np.nan)
        if all(np.isnan(adapter_off_scores)):
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(loops, full_scores, marker="o", color="#d73027", label="full system")
        ax.plot(
            loops,
            adapter_off_scores,
            marker="s",
            color="#01665e",
            label="LoRA disabled (NAMM only)",
        )
        ax.set_xlabel("Outer loop")
        ax.set_ylabel("Mean task score")
        ax.set_title("Co-adaptation diagnostic")
        ax.legend(loc="best")
        _save_fig(fig, "co_adaptation", self.figures_dir)
