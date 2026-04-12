#!/usr/bin/env python3
"""
Analysis 1 -- Per-Task Eviction Sensitivity (Test-Set Evaluation)
=================================================================
Reads test-set results from results/main_table_5t/all_results.json and
generates four plots:
  1. test_f1_comparison.png          -- grouped bar chart, test split
  2. extended_test_f1_comparison.png  -- grouped bar chart, extended_test split
  3. sensitivity_test.png            -- eviction sensitivity per task, test split
  4. recovery_ratio_test.png         -- recovery ratio per task, test split

Also writes results_test.json with the extracted numbers.

Run from repo root:
    python analysis/report_1/generate_plots_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -- Paths (relative to repo root) -------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "results" / "main_table_5t" / "all_results.json"
OUT_DIR = REPO_ROOT / "analysis" / "report_1"

# -- Task configuration -------------------------------------------------------

TASK_KEYS = [
    "lb/qasper",
    "lb/2wikimqa",
    "lb/qasper_e",
    "lb/hotpotqa_e",
    "lb/2wikimqa_e",
]
TASK_DISPLAY = {
    "lb/qasper":     "Qasper",
    "lb/2wikimqa":   "2WikiMQA",
    "lb/qasper_e":   "Qasper-E",
    "lb/hotpotqa_e": "HotpotQA-E",
    "lb/2wikimqa_e": "2WikiMQA-E",
}

# Condition ordering for bar charts
CONDITION_ORDER = [
    "B0",
    "B1/cs1024",
    "B1/cs2048",
    "M1",
    "M2/cs1024",
    "M2/cs2048",
    "M4/cs1024",
    "M4/cs2048",
    "A4/cs1024_no_namm",
    "A4/cs2048_no_namm",
]

CONDITION_COLORS = {
    "B0":                "#95a5a6",
    "B1/cs1024":         "#bdc3c7",
    "B1/cs2048":         "#7f8c8d",
    "M1":                "#f39c12",
    "M2/cs1024":         "#5dade2",
    "M2/cs2048":         "#2e86c1",
    "M4/cs1024":         "#e74c3c",
    "M4/cs2048":         "#c0392b",
    "A4/cs1024_no_namm": "#27ae60",
    "A4/cs2048_no_namm": "#1e8449",
}

CONDITION_LABELS = {
    "B0":                "B0 (no train)",
    "B1/cs1024":         "B1 cs1024",
    "B1/cs2048":         "B1 cs2048",
    "M1":                "M1 (LoRA)",
    "M2/cs1024":         "M2 cs1024",
    "M2/cs2048":         "M2 cs2048",
    "M4/cs1024":         "M4 cs1024",
    "M4/cs2048":         "M4 cs2048",
    "A4/cs1024_no_namm": "A4 cs1024",
    "A4/cs2048_no_namm": "A4 cs2048",
}


# -- Helpers -------------------------------------------------------------------

def load_data() -> dict:
    """Load all_results.json."""
    with open(DATA_PATH) as f:
        return json.load(f)


def plot_grouped_bar(
    data: dict,
    split: str,
    out_name: str,
    title: str,
) -> None:
    """Grouped bar chart of all conditions, per-task + micro mean."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
    })

    task_labels = [TASK_DISPLAY[t] for t in TASK_KEYS] + ["Micro F1"]
    keys = TASK_KEYS + ["micro"]

    conditions = [c for c in CONDITION_ORDER if c in data]
    n_cond = len(conditions)
    n_groups = len(keys)
    width = 0.8 / n_cond
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, cond in enumerate(conditions):
        vals = [data[cond][split].get(k, 0.0) for k in keys]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLORS.get(cond, "#333333"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x + width * n_cond / 2 - width / 2)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("F1 (%)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    max_val = max(
        data[c][split].get(k, 0.0)
        for c in conditions
        for k in keys
    )
    ax.set_ylim(0, max_val * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"Saved {out_name}")


def plot_sensitivity(data: dict) -> None:
    """Eviction sensitivity: (M1 - M4) / M1, test split, cs1024 and cs2048."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    m1 = data["M1"]["test"]
    cache_sizes = [1024, 2048]
    cs_colors = {1024: "#e74c3c", 2048: "#3498db"}
    task_labels = [TASK_DISPLAY[t] for t in TASK_KEYS]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASK_KEYS))
    width = 0.35

    for i, cs in enumerate(cache_sizes):
        m4 = data[f"M4/cs{cs}"]["test"]
        sens = []
        for t in TASK_KEYS:
            if abs(m1[t]) < 1e-9:
                sens.append(0.0)
            else:
                sens.append((m1[t] - m4[t]) / m1[t] * 100)

        bars = ax.bar(
            x + i * width,
            sens,
            width,
            label=f"cs={cs}",
            color=cs_colors[cs],
            edgecolor="white",
        )
        for bar, val in zip(bars, sens):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:+.1f}%",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Eviction Sensitivity  (M1 - M4) / M1  [%]")
    ax.set_title("Per-Task Eviction Sensitivity (positive = M4 worse than M1)")
    ax.legend(title="Cache Size")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_test.png", dpi=150)
    plt.close(fig)
    print("Saved sensitivity_test.png")


def plot_recovery_ratio(data: dict) -> None:
    """Recovery ratio: (M4 - M2) / (M1 - M2), test split, cs1024 and cs2048."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    m1 = data["M1"]["test"]
    cache_sizes = [1024, 2048]
    cs_colors = {1024: "#e74c3c", 2048: "#3498db"}
    task_labels = [TASK_DISPLAY[t] for t in TASK_KEYS] + ["Micro F1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASK_KEYS) + 1)  # tasks + micro
    width = 0.35

    for i, cs in enumerate(cache_sizes):
        m2 = data[f"M2/cs{cs}"]["test"]
        m4 = data[f"M4/cs{cs}"]["test"]
        rr = []
        for t in TASK_KEYS:
            denom = m1[t] - m2[t]
            if abs(denom) < 1e-9:
                rr.append(float("nan"))
            else:
                rr.append((m4[t] - m2[t]) / denom)
        # micro mean recovery
        denom_micro = m1["micro"] - m2["micro"]
        if abs(denom_micro) < 1e-9:
            rr.append(float("nan"))
        else:
            rr.append((m4["micro"] - m2["micro"]) / denom_micro)

        bars = ax.bar(
            x + i * width,
            rr,
            width,
            label=f"cs={cs}",
            color=cs_colors[cs],
            edgecolor="white",
        )
        for bar, val in zip(bars, rr):
            if np.isnan(val):
                continue
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.axhline(1.0, color="green", linewidth=1.0, linestyle="--", label="Full recovery")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Recovery Ratio  (M4 - M2) / (M1 - M2)")
    ax.set_title("Recovery Ratio: How Much LoRA+NAMM Recovers vs LoRA-only")
    ax.legend(title="Cache Size")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recovery_ratio_test.png", dpi=150)
    plt.close(fig)
    print("Saved recovery_ratio_test.png")


def save_results_json(data: dict) -> None:
    """Save extracted test/extended_test numbers to results_test.json."""
    out = {}
    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        out[cond] = {}
        for split in ["test", "extended_test"]:
            if split not in data[cond]:
                continue
            split_data = data[cond][split]
            out[cond][split] = {
                TASK_DISPLAY[t]: round(split_data[t], 2) for t in TASK_KEYS
            }
            out[cond][split]["macro"] = round(split_data["mean"], 2)
            out[cond][split]["micro"] = round(split_data["micro"], 2)
            out[cond][split]["n"] = split_data["n"]

    with open(OUT_DIR / "results_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results_test.json")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    data = load_data()
    print(f"Loaded {len(data)} conditions from {DATA_PATH}")

    # Print summary
    print("\n--- Test Split (Micro F1) ---")
    for cond in CONDITION_ORDER:
        if cond in data:
            micro = data[cond]["test"]["micro"]
            print(f"  {cond:<22s}  {micro:.2f}")

    print("\n--- Extended Test Split (Micro F1) ---")
    for cond in CONDITION_ORDER:
        if cond in data:
            micro = data[cond]["extended_test"]["micro"]
            print(f"  {cond:<22s}  {micro:.2f}")

    # Generate plots
    plot_grouped_bar(
        data,
        split="test",
        out_name="test_f1_comparison.png",
        title="Test-Set F1 Comparison (10 Conditions, 5 Tasks)",
    )
    plot_grouped_bar(
        data,
        split="extended_test",
        out_name="extended_test_f1_comparison.png",
        title="Extended-Test F1 Comparison (10 Conditions, 5 Tasks)",
    )
    plot_sensitivity(data)
    plot_recovery_ratio(data)
    save_results_json(data)

    print("\nDone.")


if __name__ == "__main__":
    main()
