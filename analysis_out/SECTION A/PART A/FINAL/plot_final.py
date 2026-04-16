"""Paper-ready plots: 4 conditions, test set only.

1. mean_f1_test.png — Mean F1 bar chart
2. per_task_f1_test.png — Per-task grouped bars
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
EVAL_DIR = os.path.join(REPO_ROOT, "analysis_out", "latest_analysis_matched",
                        "00_eval_results")
OUTPUT_DIR = os.path.dirname(__file__)

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_LABELS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]

COND_FILES = {
    "B0": "b0_base.json",
    "M1-matched\n(full cache)": "m1_matched_full_cache.json",
    "M1-matched\nunder NAMM": "m1_matched_under_namm_cs1024.json",
    "M4 LoRA\n+NAMM": "m4_lora_namm_cs1024.json",
}

COND_ORDER = list(COND_FILES.keys())

# Display labels for plots
DISPLAY_LABELS = {
    "B0": "Base",
    "M1-matched\n(full cache)": "Fine-Tuned\n(Standard)",
    "M1-matched\nunder NAMM": "Fine-Tuned (Standard)\n+ Post-Hoc NAMM",
    "M4 LoRA\n+NAMM": "Fine-Tuned\n(NAMM Active)",
}

COLORS = {
    "B0": "#888888",
    "M1-matched\n(full cache)": "#4C72B0",
    "M1-matched\nunder NAMM": "#DD8452",
    "M4 LoRA\n+NAMM": "#55A868",
}


def load_data():
    data = {}
    for label, fname in COND_FILES.items():
        with open(os.path.join(EVAL_DIR, fname)) as f:
            d = json.load(f)
        sd = d["scores_per_split"]["test"]
        data[label] = {
            "micro": sd["micro_mean_f1"],
            "tasks": {t: sd[t] for t in TASKS},
        }
    return data


def plot_mean_f1(data, output_path):
    fig, ax = plt.subplots(figsize=(5.5, 4))

    vals = [data[c]["micro"] for c in COND_ORDER]
    x = np.arange(len(COND_ORDER))
    bars = ax.bar(x, vals, width=0.6,
                  color=[COLORS[c] for c in COND_ORDER],
                  edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[c] for c in COND_ORDER], fontsize=7)
    ax.set_ylabel("Micro Mean F1 (%)", fontsize=9)
    ax.set_title("Held-out Test Set Mean F1",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_per_task_f1(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_conds = len(COND_ORDER)
    n_tasks = len(TASKS)
    bar_width = 0.18
    group_gap = 0.15

    for ci, cond in enumerate(COND_ORDER):
        task_vals = [data[cond]["tasks"][t] for t in TASKS]
        x = np.arange(n_tasks) * (n_conds * bar_width + group_gap) + ci * bar_width
        bars = ax.bar(x, task_vals, width=bar_width,
                      color=COLORS[cond], edgecolor="white", linewidth=0.5,
                      label=DISPLAY_LABELS[cond].replace("\n", " "))

    group_centers = np.arange(n_tasks) * (n_conds * bar_width + group_gap) + \
                    (n_conds - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(TASK_LABELS, fontsize=8)

    ax.set_ylabel("F1 (%)", fontsize=9)
    ax.set_title("Held-out Test Set Per-Task F1",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    data = load_data()
    plot_mean_f1(data, os.path.join(OUTPUT_DIR, "mean_f1_test.png"))
    plot_per_task_f1(data, os.path.join(OUTPUT_DIR, "per_task_f1_test.png"))


if __name__ == "__main__":
    main()
