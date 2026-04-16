"""Generate paper-ready F1 bar charts for Section A, Part A.

Plots:
  1. Mean F1 — Test set (4 conditions)
  2. Mean F1 — Extended test set (4 conditions)
  3. Per-task F1 — Test set (4 conditions x 5 tasks)
  4. Per-task F1 — Extended test set (4 conditions x 5 tasks)

All data read from analysis_out/latest_analysis_matched/00_eval_results/*.json.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
EVAL_DIR = os.path.join(REPO_ROOT, "analysis_out", "latest_analysis_matched",
                        "00_eval_results")
OUTPUT_DIR = os.path.dirname(__file__)

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_LABELS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]

# Condition label -> JSON filename in 00_eval_results
COND_FILES = {
    "B0": "b0_base.json",
    "Base+NAMM\n(M2)": "m2_base_namm_cs1024.json",
    "M1-matched\n(full cache)": "m1_matched_full_cache.json",
    "M1-matched\nunder NAMM": "m1_matched_under_namm_cs1024.json",
    "M1-matched\ntruncated": "m1_matched_trunc1024.json",
    "M4 LoRA\n+NAMM": "m4_lora_namm_cs1024.json",
    "A4: M4 LoRA\nno NAMM": "a4_m4_lora_no_namm.json",
}

COND_ORDER = list(COND_FILES.keys())

COLORS = {
    "B0": "#888888",
    "Base+NAMM\n(M2)": "#937DC2",
    "M1-matched\n(full cache)": "#4C72B0",
    "M1-matched\nunder NAMM": "#DD8452",
    "M1-matched\ntruncated": "#55A868",
    "M4 LoRA\n+NAMM": "#C44E52",
    "A4: M4 LoRA\nno NAMM": "#CCB974",
}


# ── Load data ────────────────────────────────────────────────────────

def load_data():
    data = {}
    for split in ["test", "extended_test"]:
        data[split] = {}
        for label, fname in COND_FILES.items():
            with open(os.path.join(EVAL_DIR, fname)) as f:
                d = json.load(f)
            sd = d["scores_per_split"][split]
            data[split][label] = {
                "micro": sd["micro_mean_f1"],
                "tasks": {t: sd[t] for t in TASKS},
            }
    return data


# ── Plotting ─────────────────────────────────────────────────────────

def plot_mean_f1(data, split, output_path):
    fig, ax = plt.subplots(figsize=(7, 4))

    vals = [data[split][c]["micro"] for c in COND_ORDER]
    x = np.arange(len(COND_ORDER))
    bars = ax.bar(x, vals, width=0.6,
                  color=[COLORS[c] for c in COND_ORDER],
                  edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(COND_ORDER, fontsize=9)
    split_label = "Test" if split == "test" else "Extended Test"
    ax.set_ylabel("Micro Mean F1 (%)", fontsize=11)
    ax.set_title(f"Mean F1 — {split_label} (cs1024, lr=1e-4)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_per_task_f1(data, split, output_path):
    fig, ax = plt.subplots(figsize=(13, 5))

    n_conds = len(COND_ORDER)
    n_tasks = len(TASKS)
    bar_width = 0.13
    group_gap = 0.18

    for ci, cond in enumerate(COND_ORDER):
        task_vals = [data[split][cond]["tasks"][t] for t in TASKS]
        x = np.arange(n_tasks) * (n_conds * bar_width + group_gap) + ci * bar_width
        bars = ax.bar(x, task_vals, width=bar_width,
                      color=COLORS[cond], edgecolor="white", linewidth=0.5,
                      label=cond.replace("\n", " "))

        for bar, val in zip(bars, task_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                    rotation=45)

    group_centers = np.arange(n_tasks) * (n_conds * bar_width + group_gap) + \
                    (n_conds - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(TASK_LABELS, fontsize=10)

    split_label = "Test" if split == "test" else "Extended Test"
    ax.set_ylabel("F1 (%)", fontsize=11)
    ax.set_title(f"Per-Task F1 — {split_label} (cs1024, lr=1e-4)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    data = load_data()

    plot_mean_f1(data, "test",
                 os.path.join(OUTPUT_DIR, "mean_f1_test.png"))
    plot_mean_f1(data, "extended_test",
                 os.path.join(OUTPUT_DIR, "mean_f1_extended_test.png"))
    plot_per_task_f1(data, "test",
                     os.path.join(OUTPUT_DIR, "per_task_f1_test.png"))
    plot_per_task_f1(data, "extended_test",
                     os.path.join(OUTPUT_DIR, "per_task_f1_extended_test.png"))


if __name__ == "__main__":
    main()
