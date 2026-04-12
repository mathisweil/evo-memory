"""Generate comparison bar charts from test-set F1 data.

Reads results/main_table_5t/all_results.json and produces:

  results/M1/test_comparison_bar.png
      M1 vs B0 (baseline), per-task + micro mean.

  results/M2/test_comparison_bar.png
      M2 vs B0, B1 (recency), Trunc (truncation), per-task + micro mean.

  results/M3/test_comparison_bar.png
      M3/cs1024, M3/cs2048 vs M1, Trunc/lora, A4 ablation, per-task + micro.
      NOTE: results dir uses "M4" label but this is actually experiment-spec M3
      (LoRA + frozen NAMM). See experiment_specification.md naming warning.

  results/main_table_5t/plots/test_overview.png
      All conditions, per-task grouped bars, test split.

Usage:
    .venv/bin/python scripts/generate_test_comparison_plots.py
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "main_table_5t" / "all_results.json"
RESULTS_DIR = ROOT / "results"

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_LABELS = ["Qasper", "2WikiMQA", "Qasper-E", "HotpotQA-E", "2WikiMQA-E"]

# Conditions to skip (broken runs)
SKIP_CONDITIONS = {"M1_recency/cs1024"}

# Consistent color palette matching plot_main_table.py
COLORS = {
    "B0": "#7f7f7f",
    "B1/cs1024": "#bcbd22",
    "B1/cs2048": "#9d9c20",
    "M1": "#1f77b4",
    "M2/cs1024": "#2ca02c",
    "M2/cs2048": "#1d6a1d",
    "M4/cs1024": "#d62728",
    "M4/cs2048": "#8c1a1b",
    "A4/cs1024_no_namm": "#e377c2",
    "A4/cs2048_no_namm": "#9b3b8c",
    "Trunc/plain_1024": "#c5b0d5",
    "Trunc/plain_2048": "#9467bd",
    "Trunc/lora_m1_1024": "#aec7e8",
    "Trunc/lora_m1_2048": "#5d8aa8",
}

# Display labels — NOTE: "M4" in results dir = experiment-spec M3 (frozen NAMM)
LABELS = {
    "B0": "B0 (baseline)",
    "B1/cs1024": "B1 recency cs1024",
    "B1/cs2048": "B1 recency cs2048",
    "M1": "M1 (LoRA only)",
    "M2/cs1024": "M2 NAMM cs1024",
    "M2/cs2048": "M2 NAMM cs2048",
    "M4/cs1024": "M3 LoRA+NAMM cs1024",
    "M4/cs2048": "M3 LoRA+NAMM cs2048",
    "A4/cs1024_no_namm": "A4 (no NAMM) cs1024",
    "A4/cs2048_no_namm": "A4 (no NAMM) cs2048",
    "Trunc/plain_1024": "Trunc plain 1024",
    "Trunc/plain_2048": "Trunc plain 2048",
    "Trunc/lora_m1_1024": "Trunc LoRA-M1 1024",
    "Trunc/lora_m1_2048": "Trunc LoRA-M1 2048",
}


def load_results() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


def get_values(results: dict, condition: str, split: str = "test") -> list[float]:
    """Return per-task F1 values + micro mean for a condition."""
    data = results[condition][split]
    per_task = [data[t] for t in TASKS]
    return per_task + [data["micro"]]


def _annotate_bars(ax, bars, fontsize: float = 6.5) -> None:
    """Add value annotations above bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


def _bar_chart(results, conditions, out_path, title, figsize=(14, 6), fontsize=6):
    """Generic grouped bar chart for a set of conditions."""
    # Filter to conditions present in results
    conditions = [c for c in conditions if c in results and c not in SKIP_CONDITIONS]
    group_labels = TASK_LABELS + ["Micro Mean"]
    x = np.arange(len(group_labels))
    n_bars = len(conditions)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(conditions):
        vals = get_values(results, cond)
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=LABELS.get(cond, cond),
            color=COLORS.get(cond, "#333333"),
            edgecolor="black", linewidth=0.3,
        )
        _annotate_bars(ax, bars, fontsize=fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel("F1 (%)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    all_vals = [v for c in conditions for v in get_values(results, c)]
    ax.set_ylim(0, max(all_vals) * 1.22 if all_vals else 50)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_m1_comparison(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return _bar_chart(
        results, ["B0", "M1"],
        out_dir / "test_comparison_bar.png",
        "M1 (LoRA only) \u2014 Test F1 vs Baseline  (n=70)",
        figsize=(12, 6), fontsize=7,
    )


def plot_m2_comparison(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return _bar_chart(
        results,
        ["B0", "B1/cs1024", "B1/cs2048",
         "Trunc/plain_1024", "Trunc/plain_2048",
         "M2/cs1024", "M2/cs2048"],
        out_dir / "test_comparison_bar.png",
        "M2 (NAMM only) vs Recency & Truncation \u2014 Test F1  (n=70)",
        figsize=(16, 6), fontsize=5.5,
    )


def plot_m3_comparison(results: dict, out_dir: Path) -> Path:
    """M3 (frozen NAMM + LoRA) vs M1, truncation, A4 ablation.

    NOTE: results dir labels this "M4" but it is experiment-spec M3.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return _bar_chart(
        results,
        ["B0", "M1",
         "Trunc/lora_m1_1024", "Trunc/lora_m1_2048",
         "A4/cs1024_no_namm", "A4/cs2048_no_namm",
         "M4/cs1024", "M4/cs2048"],
        out_dir / "test_comparison_bar.png",
        "M3 (LoRA + frozen NAMM) vs M1, Truncation & A4 Ablation \u2014 Test F1  (n=70)",
        figsize=(16, 6), fontsize=5.5,
    )


def plot_overview(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_overview.png"

    condition_order = [
        "B0",
        "B1/cs1024", "B1/cs2048",
        "Trunc/plain_1024", "Trunc/plain_2048",
        "M1",
        "Trunc/lora_m1_1024", "Trunc/lora_m1_2048",
        "M2/cs1024", "M2/cs2048",
        "M4/cs1024", "M4/cs2048",
        "A4/cs1024_no_namm", "A4/cs2048_no_namm",
    ]
    conditions = [c for c in condition_order
                  if c in results and c not in SKIP_CONDITIONS]

    group_labels = TASK_LABELS + ["Micro Mean"]
    x = np.arange(len(group_labels))
    n_bars = len(conditions)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(20, 7))
    for i, cond in enumerate(conditions):
        vals = get_values(results, cond)
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=LABELS.get(cond, cond),
            color=COLORS.get(cond, "#333333"),
            edgecolor="black", linewidth=0.25,
        )
        _annotate_bars(ax, bars, fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel("F1 (%)", fontsize=12)
    ax.set_title("All Conditions \u2014 Per-Task Test F1 (n=70)", fontsize=14)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    all_vals = [v for c in conditions for v in get_values(results, c)]
    ax.set_ylim(0, max(all_vals) * 1.20 if all_vals else 50)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    results = load_results()
    created: list[Path] = []

    logger.info("Generating M1 test comparison bar chart...")
    created.append(plot_m1_comparison(results, RESULTS_DIR / "M1"))

    logger.info("Generating M2 test comparison bar chart...")
    created.append(plot_m2_comparison(results, RESULTS_DIR / "M2"))

    logger.info("Generating M3 (results dir: M3/) test comparison bar chart...")
    created.append(plot_m3_comparison(results, RESULTS_DIR / "M3"))

    logger.info("Generating overview test bar chart...")
    created.append(plot_overview(results, RESULTS_DIR / "main_table_5t" / "plots"))

    logger.info("Created %d plots:", len(created))
    for p in created:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
