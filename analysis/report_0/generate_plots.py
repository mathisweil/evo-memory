"""Report 0 -- Dataset Characterisation: Relevant Token Analysis.

Loads pre-extracted per-sample relevant-token statistics from
relevant_tokens_data.json and generates:
  1. relevant_tokens.png          -- 3-panel bar: mean tokens, fraction, regions
  2. relevant_tokens_boxplot.png  -- boxplot of relevant tokens per task
  3. answer_positions.png         -- histogram of answer position in context
  4. eviction_survival.png        -- estimated fraction of relevant tokens
                                     surviving eviction at different cache sizes

The remaining report_0 plots (dataset_characteristics.png, prompt_templates.png,
length_distributions.png, answer_types.png, eviction_analysis.png) are generated
by analyse_datasets.py which requires loading the raw LongBench dataset.

Run:
    PYTHONPATH=. .venv/bin/python analysis/report_0/generate_plots.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "relevant_tokens_data.json"
OUT_DIR = Path(__file__).parent

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_LABELS = {
    "qasper": "Qasper",
    "2wikimqa": "2WikiMQA",
    "qasper_e": "Qasper-E",
    "hotpotqa_e": "HotpotQA-E",
    "2wikimqa_e": "2WikiMQA-E",
}
TASK_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

CACHE_SIZES = [1024]


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_data() -> dict[str, list[dict]]:
    with open(DATA_PATH) as f:
        return json.load(f)


def _per_task_means(
    data: dict[str, list[dict]], field: str
) -> dict[str, float]:
    out: dict[str, float] = {}
    for task in TASKS:
        vals = [s[field] for s in data[task] if s[field] is not None]
        out[task] = float(np.mean(vals)) if vals else 0.0
    return out


# ── Plot 1: Relevant tokens 3-panel bar chart ──────────────────────────────


def plot_relevant_tokens(data: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(TASKS))
    labels = [TASK_LABELS[t] for t in TASKS]
    width = 0.6

    panels = [
        ("relevant_tokens", "Tokens", "Mean Relevant Tokens per Prompt"),
        ("relevant_fraction", "% of Context", "Mean Relevant Fraction of Context"),
        ("n_relevant_regions", "Regions", "Mean # Distinct Relevant Regions"),
    ]

    for ax, (field, ylabel, title) in zip(axes, panels):
        means = _per_task_means(data, field)
        vals = [means[t] for t in TASKS]
        if field == "relevant_fraction":
            vals = [v * 100 for v in vals]

        bars = ax.bar(x, vals, width, color=TASK_COLORS, edgecolor="white",
                      linewidth=0.5)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.1f}%" if field == "relevant_fraction" else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("Relevant Token Analysis by Task", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / "relevant_tokens.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 2: Relevant tokens boxplot ────────────────────────────────────────


def plot_relevant_tokens_boxplot(data: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    bp_data = []
    for task in TASKS:
        bp_data.append([s["relevant_tokens"] for s in data[task]])

    bp = ax.boxplot(bp_data, patch_artist=True, showfliers=True,
                    flierprops=dict(markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], TASK_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for cs in CACHE_SIZES:
        ax.axhline(y=cs, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(len(TASKS) + 0.4, cs, f"cache={cs}", fontsize=8, va="center",
                color="grey")

    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10)
    ax.set_ylabel("Estimated Relevant Tokens", fontsize=11)
    ax.set_title("Distribution of Relevant Tokens per Prompt", fontsize=13,
                 fontweight="bold")
    ax.legend(
        [plt.Line2D([0], [0], color="grey", linestyle="--", linewidth=0.8)],
        ["Cache size thresholds"],
        fontsize=9, loc="upper left",
    )
    fig.tight_layout()
    path = OUT_DIR / "relevant_tokens_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 3: Answer position histogram ──────────────────────────────────────


def plot_answer_positions(data: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for task, color in zip(TASKS, TASK_COLORS):
        positions = [
            s["answer_position_relative"]
            for s in data[task]
            if s["answer_position_relative"] is not None
        ]
        if positions:
            ax.hist(positions, bins=20, alpha=0.4, color=color,
                    label=TASK_LABELS[task], edgecolor="white", linewidth=0.5,
                    density=True)

    ax.set_xlabel("Relative Position in Context (0=start, 1=end)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of First Answer Occurrence Position", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "answer_positions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 4: Eviction survival estimate ─────────────────────────────────────


def plot_eviction_survival(data: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(TASKS))
    width = 0.25
    cs_colors = {1024: "#3498db", 2048: "#ff7f0e", 3072: "#2ecc71"}

    for i, cs in enumerate(CACHE_SIZES):
        survival_pcts = []
        for task in TASKS:
            samples = data[task]
            fracs = []
            for s in samples:
                ctx = s["context_tokens"]
                rel = s["relevant_tokens"]
                if ctx > 0 and rel > 0:
                    tokens_evicted = max(0, ctx - cs)
                    if tokens_evicted == 0:
                        fracs.append(100.0)
                    else:
                        survival = min(cs, rel) / rel * 100
                        fracs.append(min(survival, 100.0))
            survival_pcts.append(float(np.mean(fracs)) if fracs else 100.0)

        bars = ax.bar(x + i * width, survival_pcts, width, label=f"cache={cs}",
                      color=cs_colors[cs], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, survival_pcts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=10)
    ax.set_ylabel("Estimated Relevant Token Survival (%)", fontsize=11)
    ax.set_title(
        "Estimated Fraction of Relevant Tokens Fitting in Cache\n"
        "(assuming ideal eviction policy)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    path = OUT_DIR / "eviction_survival.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    data = load_data()
    print(f"Loaded data for {len(data)} tasks from {DATA_PATH}")
    for task in TASKS:
        print(f"  {task}: {len(data[task])} samples")

    plot_relevant_tokens(data)
    plot_relevant_tokens_boxplot(data)
    plot_answer_positions(data)
    plot_eviction_survival(data)
    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
