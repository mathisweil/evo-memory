#!/usr/bin/env python3
"""Analysis 8 -- Probing for Residual Knowledge of Evicted Content (plots).

Loads pre-computed probe data from maskfix_probe_data.npz and generates:
  probe_accuracy.png        -- Per-layer probe accuracy for M1, M3, and random baseline
  entity_survival.png       -- Per-task answer-token survival fraction after NAMM eviction
  layer_wise_information.png -- Per-layer accuracy gap (M1 - M3)

Only M1 vs M3 data is plotted (mask-fixed run). No buggy data is used.

Runnable without a GPU:
    PYTHONPATH=. .venv/bin/python analysis/report_8/generate_plots.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG -- how to regenerate the data
# ---------------------------------------------------------------------------
# Data file produced by analysis/run_maskfix_namm_analyses.py on a GPU node.
# That script loads the following checkpoints:
#   M1 LoRA   : experiment_artifacts/gcs/M1/best_ckpt.pt
#   M3 LoRA   : experiment_artifacts/gcs/M3_cs1024/best_ckpt.pt
#   M2 NAMM   : experiment_artifacts/gcs/M2_cs1024/ckpt.pt
#                (frozen NAMM scoring network, shared between M2 and M3)
# and trains logistic regression probes at each layer to predict whether
# gold-answer entities were evicted.
#
# To regenerate the data:
#   PYTHONPATH=. .venv/bin/python analysis/run_maskfix_namm_analyses.py
#
# NPZ keys (all from the mask-fixed run):
#   m1_accuracies          (17,)   -- M1 probe accuracy per layer
#   m3_accuracies          (17,)   -- M3 probe accuracy per layer
#   m1_stds                (17,)   -- M1 accuracy std per layer
#   m3_stds                (17,)   -- M3 accuracy std per layer
#   random_accuracy        (1,)    -- majority-class baseline accuracy
#   n_samples              (1,)    -- total number of probe samples
#   n_positive             (1,)    -- number of "answer evicted" samples
#   labels                 (40,)   -- binary label per sample
#   answer_survival_fracs  (40,)   -- approx survival fraction per sample
#   retained_counts        (40,)   -- tokens retained after eviction
#   total_counts           (40,)   -- original sequence lengths
#   task_names             (40,)   -- task name per sample (object array)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "maskfix_probe_data.npz"
OUT_DIR = SCRIPT_DIR

NUM_LAYERS = 16  # LLaMA 3.2-1B decoder layers


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _scalar(arr: np.ndarray | float) -> float:
    """Extract a scalar from a possibly-array value."""
    if hasattr(arr, "__len__") and len(arr) == 1:
        return float(arr[0])
    return float(arr)


def plot_probe_accuracy(
    m1_acc: np.ndarray,
    m3_acc: np.ndarray,
    m1_std: np.ndarray,
    m3_std: np.ndarray,
    random_acc: float,
    n_samples: int,
    n_positive: int,
    out_dir: Path,
) -> None:
    """Per-layer probe accuracy for M1, M3, and the majority-class baseline."""
    n_layers_plus_1 = len(m1_acc)
    layers = np.arange(n_layers_plus_1)
    layer_labels = ["emb"] + [f"{i}" for i in range(n_layers_plus_1 - 1)]

    colors = {"M1": "#d62728", "M3": "#1f77b4", "random": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, m1_acc, "-o", color=colors["M1"],
            label="M1 (full context)", linewidth=2, markersize=5)
    ax.fill_between(layers, m1_acc - m1_std, m1_acc + m1_std,
                    alpha=0.15, color=colors["M1"])

    ax.plot(layers, m3_acc, "-s", color=colors["M3"],
            label="M3 cs1024 (with eviction)", linewidth=2, markersize=5)
    ax.fill_between(layers, m3_acc - m3_std, m3_acc + m3_std,
                    alpha=0.15, color=colors["M3"])

    ax.axhline(random_acc, color=colors["random"], linestyle="--",
               linewidth=1.5, label=f"Majority-class baseline ({random_acc:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy (CV)")
    ax.set_title(
        "Entity Presence Probe: Per-Layer Accuracy\n"
        "Can a linear probe on retained hidden states detect evicted answer entities?"
    )
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.02, 0.02,
        f"n={n_samples}, {n_positive} positive / {n_samples - n_positive} negative",
        transform=ax.transAxes, fontsize=9, color="grey",
        verticalalignment="bottom",
    )

    fig.tight_layout()
    path = out_dir / "probe_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_entity_survival(
    task_names: np.ndarray,
    answer_survival_fracs: np.ndarray,
    out_dir: Path,
) -> None:
    """Per-task bar chart of answer-token survival fraction after eviction."""
    # Group survival fractions by task
    task_survival: dict[str, list[float]] = {}
    for task_raw, frac in zip(task_names, answer_survival_fracs):
        task = str(task_raw).replace("/", "_")
        fval = float(frac)
        if not np.isnan(fval):
            task_survival.setdefault(task, []).append(fval)

    display_names = {
        "lb_qasper": "Qasper",
        "lb_2wikimqa": "2WikiMQA",
        "lb_qasper_e": "Qasper-E",
        "lb_hotpotqa_e": "HotpotQA-E",
        "lb_2wikimqa_e": "2WikiMQA-E",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    if task_survival:
        sorted_tasks = sorted(task_survival.keys())
        x = np.arange(len(sorted_tasks))
        means = [np.mean(task_survival[t]) for t in sorted_tasks]
        stds = [np.std(task_survival[t]) for t in sorted_tasks]
        counts = [len(task_survival[t]) for t in sorted_tasks]

        ax.bar(x, means, yerr=stds, capsize=4,
               color="#2ca02c", edgecolor="white", alpha=0.85)

        for xi, cnt in zip(x, counts):
            ax.text(xi, 0.02, f"n={cnt}", ha="center", fontsize=8,
                    color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_names.get(t, t) for t in sorted_tasks],
            rotation=15, ha="right",
        )
    else:
        ax.text(0.5, 0.5, "No survival data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="grey")

    ax.set_ylabel("Answer Token Survival Fraction")
    ax.set_title(
        "Entity Survival: Fraction of Answer Tokens Retained After Eviction\n"
        "(per task, NAMM cs1024)"
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.5, label="No eviction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = out_dir / "entity_survival.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_layer_wise_information(
    m1_acc: np.ndarray,
    m3_acc: np.ndarray,
    m1_std: np.ndarray,
    m3_std: np.ndarray,
    out_dir: Path,
) -> None:
    """Per-layer accuracy gap (M1 - M3): positive means info lost to eviction."""
    n_layers_plus_1 = len(m1_acc)
    layers = np.arange(n_layers_plus_1)
    layer_labels = ["emb"] + [f"{i}" for i in range(n_layers_plus_1 - 1)]

    diff = m1_acc - m3_acc
    diff_std = np.sqrt(m1_std**2 + m3_std**2)  # propagated uncertainty

    bar_colors = ["#e74c3c" if d > 0 else "#27ae60" for d in diff]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(layers, diff, yerr=diff_std, capsize=3,
           color=bar_colors, edgecolor="white", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy Difference (M1 - M3)")
    ax.set_title(
        "Information Loss per Layer: Probe Accuracy Gap\n"
        "Positive = M1 retains more entity info than M3 (information lost to eviction)"
    )
    ax.set_xticks(layers)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate the layer with maximum gap
    max_layer = int(np.argmax(diff))
    if diff[max_layer] > 0:
        ax.annotate(
            f"Max gap: layer {layer_labels[max_layer]}\n({diff[max_layer]:.3f})",
            xy=(max_layer, diff[max_layer]),
            xytext=(max_layer + 1.5, diff[max_layer] + 0.02),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=9, color="grey",
        )

    fig.tight_layout()
    path = out_dir / "layer_wise_information.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            "Generate it on a GPU node with:\n"
            "  PYTHONPATH=. .venv/bin/python analysis/run_maskfix_namm_analyses.py"
        )

    loaded = np.load(DATA_FILE, allow_pickle=True)
    print(f"Loaded {DATA_FILE}")
    print(f"  Keys: {sorted(loaded.keys())}")

    m1_acc = loaded["m1_accuracies"]
    m3_acc = loaded["m3_accuracies"]
    m1_std = loaded["m1_stds"]
    m3_std = loaded["m3_stds"]
    random_acc = _scalar(loaded["random_accuracy"])
    n_samples = int(_scalar(loaded["n_samples"]))
    n_positive = int(_scalar(loaded["n_positive"]))

    print(f"  m1_accuracies range: [{m1_acc.min():.3f}, {m1_acc.max():.3f}]")
    print(f"  m3_accuracies range: [{m3_acc.min():.3f}, {m3_acc.max():.3f}]")
    print(f"  random_accuracy: {random_acc:.3f}")
    print(f"  n_samples={n_samples}, n_positive={n_positive}")

    os.makedirs(OUT_DIR, exist_ok=True)

    plot_probe_accuracy(
        m1_acc, m3_acc, m1_std, m3_std,
        random_acc, n_samples, n_positive, OUT_DIR,
    )
    plot_entity_survival(
        loaded["task_names"],
        loaded["answer_survival_fracs"],
        OUT_DIR,
    )
    plot_layer_wise_information(m1_acc, m3_acc, m1_std, m3_std, OUT_DIR)

    # Print summary
    diff = m1_acc - m3_acc
    print(f"\nSummary:")
    print(f"  M1 mean probe accuracy: {np.mean(m1_acc):.3f}")
    print(f"  M3 mean probe accuracy: {np.mean(m3_acc):.3f}")
    print(f"  Random baseline:        {random_acc:.3f}")
    print(f"  Mean accuracy gap:      {np.mean(diff):.3f}")
    print(f"  Max gap at layer {int(np.argmax(diff))}: {diff.max():.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
