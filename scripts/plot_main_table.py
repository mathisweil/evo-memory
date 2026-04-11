"""Generate comparison plots from results/main_table_5t/all_results.json.

Produces (under results/main_table_5t/plots/):
  - mean_f1_test.png             — micro mean F1 per condition (test split)
  - mean_f1_extended.png         — micro mean F1 per condition (extended_test)
  - per_task_test.png            — per-task grouped bars (test)
  - per_task_extended.png        — per-task grouped bars (extended_test)
  - cs_sweep.png                 — cache_size 1024 vs 2048 for B1/M2/M4

**Headline metric is the MICRO mean F1** (prompt-count-weighted, matches the
val_lb_avg_f1 reported during LoRA training). The macro mean (1/5 per task)
is shown as a side annotation on the per-condition bars for transparency.
Per-task plots are unchanged — those are within-task means.

Usage:
    /cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python \
        scripts/plot_main_table.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path("/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo")
ROOT = PROJ / "results" / "main_table_5t"
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_LABELS = [t.replace("lb/", "") for t in TASKS]

# Display order + colors. Conditions absent from all_results.json are skipped.
CONDITIONS = [
    ("B0",                "B0 plain",            "#7f7f7f"),
    ("B1/cs1024",         "B1 recency cs1024",   "#bcbd22"),
    ("B1/cs2048",         "B1 recency cs2048",   "#9d9c20"),
    ("M1",                "M1 LoRA only",        "#1f77b4"),
    ("M2/cs1024",         "M2 NAMM cs1024",      "#2ca02c"),
    ("M2/cs2048",         "M2 NAMM cs2048",      "#1d6a1d"),
    ("M4/cs1024",         "M4 LoRA+NAMM cs1024", "#d62728"),
    ("M4/cs2048",         "M4 LoRA+NAMM cs2048", "#8c1a1b"),
    ("A4/cs1024_no_namm", "A4 (M4-cs1024 no NAMM)", "#e377c2"),
    ("A4/cs2048_no_namm", "A4 (M4-cs2048 no NAMM)", "#9b3b8c"),
]


def load_results():
    with open(ROOT / "all_results.json") as f:
        return json.load(f)


def plot_mean_f1(results, split, out_path, title, n_examples=None):
    """Bar plot of micro mean F1 per condition. Annotates each bar with the
    macro F1 in parentheses underneath for transparency.
    """
    rows = []
    for key, name, color in CONDITIONS:
        s = results.get(key, {}).get(split, {})
        micro = s.get("micro")
        macro = s.get("mean")
        if micro is None:
            continue
        rows.append((name, color, micro, macro))
    if not rows:
        return False
    names = [r[0] for r in rows]
    colors = [r[1] for r in rows]
    vals = [r[2] for r in rows]
    macros = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(vals)), vals, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Micro mean F1 (%) — prompt-count-weighted")
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, micro, macro in zip(bars, vals, macros):
        x = b.get_x() + b.get_width() / 2
        ax.text(x, micro + max(vals) * 0.01,
                f"{micro:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        if macro is not None:
            ax.text(x, micro + max(vals) * 0.055,
                    f"(macro {macro:.1f})", ha="center", va="bottom",
                    fontsize=7, color="#444")

    # Footer caption — explicit about the averaging
    caption = ("Headline = micro (prompt-weighted) mean F1, matches "
               "LoRA training val_lb_avg_f1.\n"
               "Italic numbers in parentheses = macro mean over 5 tasks "
               "(each task = 1/5).")
    if n_examples is not None:
        caption = f"n = {n_examples} prompts. " + caption
    fig.text(0.5, 0.005, caption, ha="center", va="bottom",
             fontsize=8, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def plot_per_task(results, split, out_path, title):
    present = [(key, name, color) for key, name, color in CONDITIONS
               if results.get(key, {}).get(split)]
    if not present:
        return False

    n_cond = len(present)
    n_task = len(TASKS)
    width = 0.8 / n_cond
    x = np.arange(n_task)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (key, name, color) in enumerate(present):
        scores = results[key][split]
        vals = [scores.get(t, np.nan) for t in TASKS]
        offsets = x + (i - (n_cond - 1) / 2) * width
        ax.bar(offsets, vals, width=width, label=name, color=color,
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, rotation=15)
    ax.set_ylabel("Per-task F1 (%) — within-task mean over prompts")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    fig.text(0.5, 0.01,
             "Each bar = mean F1 over the prompts of that task in this "
             "split (per-task macro). Aggregation across tasks is shown in "
             "the mean_f1_*.png plots.",
             ha="center", va="bottom", fontsize=7, color="#444")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def plot_cs_sweep(results, out_path, split="test"):
    """For each family that has both cs1024 and cs2048, show both side-by-side
    on the requested split's MICRO mean F1.
    """
    families = [
        ("B1 recency",            "B1/cs1024",          "B1/cs2048"),
        ("M2 standalone NAMM",    "M2/cs1024",          "M2/cs2048"),
        ("M4 LoRA+NAMM",          "M4/cs1024",          "M4/cs2048"),
        ("A4 ablation (no NAMM)", "A4/cs1024_no_namm",  "A4/cs2048_no_namm"),
    ]
    rows = []
    for fam, k1, k2 in families:
        v1 = results.get(k1, {}).get(split, {}).get("micro")
        v2 = results.get(k2, {}).get(split, {}).get("micro")
        if v1 is not None or v2 is not None:
            rows.append((fam, v1, v2))
    if not rows:
        return False

    n = len(rows)
    width = 0.36
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    cs1024 = [r[1] if r[1] is not None else 0 for r in rows]
    cs2048 = [r[2] if r[2] is not None else 0 for r in rows]
    b1 = ax.bar(x - width / 2, cs1024, width, label="cs=1024",
                color="#4c72b0", edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + width / 2, cs2048, width, label="cs=2048",
                color="#dd8452", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=15)
    ax.set_ylabel("Micro mean F1 (%) — prompt-count-weighted")
    ax.set_title(f"Cache-size sweep — {split} split")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()

    # Plain baseline horizontal line (also in micro)
    b0 = results.get("B0", {}).get(split, {}).get("micro")
    if b0 is not None:
        ax.axhline(b0, color="gray", linestyle="--", linewidth=1)
        ax.text(n - 0.5, b0 + 0.3, f"B0 plain = {b0:.1f}", color="gray",
                ha="right", fontsize=8)

    # M1 LoRA-only line for reference
    m1 = results.get("M1", {}).get(split, {}).get("micro")
    if m1 is not None:
        ax.axhline(m1, color="#1f77b4", linestyle=":", linewidth=1)
        ax.text(n - 0.5, m1 + 0.3, f"M1 LoRA only = {m1:.1f}",
                color="#1f77b4", ha="right", fontsize=8)

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.text(0.5, 0.005,
             "Micro mean = prompt-count-weighted (matches LoRA "
             "training val_lb_avg_f1).",
             ha="center", va="bottom", fontsize=8, color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def _detect_n(results, split):
    """Find n_prompts_total from any condition that has it for this split."""
    for cond in results.values():
        n = cond.get(split, {}).get("n")
        if n is not None:
            return n
    return None


def main():
    results = load_results()
    plots = []

    n_test = _detect_n(results, "test")
    n_ext = _detect_n(results, "extended_test")

    if plot_mean_f1(results, "test", PLOTS / "mean_f1_test.png",
                    "5-task QA — Micro mean F1 (test split)",
                    n_examples=n_test):
        plots.append("mean_f1_test.png")
    if plot_mean_f1(results, "extended_test", PLOTS / "mean_f1_extended.png",
                    "5-task QA — Micro mean F1 (extended_test, 6500–8192 tok)",
                    n_examples=n_ext):
        plots.append("mean_f1_extended.png")
    if plot_per_task(results, "test", PLOTS / "per_task_test.png",
                     "Per-task F1 — test split"):
        plots.append("per_task_test.png")
    if plot_per_task(results, "extended_test", PLOTS / "per_task_extended.png",
                     "Per-task F1 — extended_test split"):
        plots.append("per_task_extended.png")
    if plot_cs_sweep(results, PLOTS / "cs_sweep.png", split="test"):
        plots.append("cs_sweep.png")
    if plot_cs_sweep(results, PLOTS / "cs_sweep_extended.png",
                     split="extended_test"):
        plots.append("cs_sweep_extended.png")

    print(f"Wrote {len(plots)} plots to {PLOTS}:")
    for p in plots:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
