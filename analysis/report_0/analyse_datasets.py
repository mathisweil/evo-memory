#!/usr/bin/env python3
"""
Dataset characteristics analysis for LongBench 5-task QA subset.

Produces:
  - dataset_characteristics.png: summary table-as-figure
  - prompt_templates.png: prompt templates for each task
  - length_distributions.png: context and answer length distributions
  - Printed statistics used in _report.md

Tasks analysed:
  lb/qasper, lb/2wikimqa, lb/qasper_e, lb/hotpotqa_e, lb/2wikimqa_e

Filtering applied (matching experiment configs):
  - max_conditioning_length word filter: length < 6500/1.3 = 5000 words
  - max_answer_tokens word filter: shortest answer <= 64/1.3 ~ 49 words
  - min_conditioning_length: >= 4096 tokens (approximated as >= 3151 words)
  - train_frac=0.7, val_frac=0.15, split_seed=42
"""

import os
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import Counter

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_DISPLAY = {
    "qasper": "Qasper",
    "2wikimqa": "2WikiMQA",
    "qasper_e": "Qasper-E",
    "hotpotqa_e": "HotpotQA-E",
    "2wikimqa_e": "2WikiMQA-E",
}
MAX_COND_LENGTH = 6500
MIN_COND_LENGTH = 4096
MAX_ANSWER_TOKENS = 64
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
SPLIT_SEED = 42

# Approximate token/word ratios used in the codebase
TOKEN_WORD_RATIO = 1.3

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prompt templates (from data/longbench/dataset2prompt.json)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROMPT_FILE = os.path.join(REPO_ROOT, "data", "longbench", "dataset2prompt.json")

with open(PROMPT_FILE) as f:
    PROMPT_TEMPLATES = json.load(f)


# ---------------------------------------------------------------------------
# Load and filter datasets
# ---------------------------------------------------------------------------
def load_and_filter(task_name):
    """Load a LongBench task, apply word-based filters matching the codebase."""
    ds = load_dataset("THUDM/LongBench", task_name, split="test",
                      trust_remote_code=True)
    all_examples = list(ds)

    max_words = MAX_COND_LENGTH / TOKEN_WORD_RATIO
    max_ans_words = MAX_ANSWER_TOKENS / TOKEN_WORD_RATIO
    min_words = MIN_COND_LENGTH / TOKEN_WORD_RATIO

    # Step 1: filter by context length (upper bound)
    filtered = [ex for ex in all_examples if ex["length"] < max_words]

    # Step 2: filter by answer length
    filtered2 = []
    for ex in filtered:
        answers = ex["answers"]
        if isinstance(answers, str):
            answers = [answers]
        shortest_ans_words = min(len(a.split()) for a in answers)
        if shortest_ans_words <= max_ans_words:
            filtered2.append(ex)

    # Step 3: filter by min context length
    eligible = [ex for ex in filtered2 if ex["length"] >= min_words]

    return all_examples, filtered2, eligible


def compute_splits(n_eligible):
    """Compute train/val/test sizes matching the codebase logic."""
    n_train = int(n_eligible * TRAIN_FRAC)
    n_val = int(n_eligible * VAL_FRAC)
    n_test = n_eligible - n_train - n_val
    return n_train, n_val, n_test


def classify_answer(answer_text):
    """Classify answer type."""
    ans = answer_text.strip().lower()
    words = ans.split()
    n_words = len(words)
    if ans in ("yes", "no"):
        return "yes/no"
    if ans == "unanswerable":
        return "unanswerable"
    if n_words <= 3:
        return "short factoid"
    if n_words <= 10:
        return "phrase"
    return "sentence+"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyse():
    stats = {}

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"  Task: {task}")
        print(f"{'='*60}")

        all_ex, filtered, eligible = load_and_filter(task)

        # Context lengths (word-count field) for eligible samples
        ctx_lengths = np.array([ex["length"] for ex in eligible])
        # Context lengths in characters
        ctx_chars = np.array([len(ex["context"]) for ex in eligible])

        # Answer analysis
        ans_word_counts = []
        ans_types = []
        for ex in eligible:
            answers = ex["answers"]
            if isinstance(answers, str):
                answers = [answers]
            shortest = min(answers, key=lambda a: len(a.split()))
            ans_word_counts.append(len(shortest.split()))
            ans_types.append(classify_answer(shortest))

        ans_word_counts = np.array(ans_word_counts)
        type_counter = Counter(ans_types)

        # Splits
        n_train, n_val, n_test = compute_splits(len(eligible))

        stats[task] = {
            "total_raw": len(all_ex),
            "after_filter": len(filtered),
            "eligible": len(eligible),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "ctx_length_mean": float(np.mean(ctx_lengths)) if len(ctx_lengths) > 0 else 0,
            "ctx_length_median": float(np.median(ctx_lengths)) if len(ctx_lengths) > 0 else 0,
            "ctx_length_min": float(np.min(ctx_lengths)) if len(ctx_lengths) > 0 else 0,
            "ctx_length_max": float(np.max(ctx_lengths)) if len(ctx_lengths) > 0 else 0,
            "ctx_length_std": float(np.std(ctx_lengths)) if len(ctx_lengths) > 0 else 0,
            "ctx_chars_mean": float(np.mean(ctx_chars)) if len(ctx_chars) > 0 else 0,
            "ans_words_mean": float(np.mean(ans_word_counts)) if len(ans_word_counts) > 0 else 0,
            "ans_words_median": float(np.median(ans_word_counts)) if len(ans_word_counts) > 0 else 0,
            "ans_words_max": float(np.max(ans_word_counts)) if len(ans_word_counts) > 0 else 0,
            "ans_type_counts": dict(type_counter),
            "ctx_lengths": ctx_lengths,
            "ans_word_counts": ans_word_counts,
        }

        print(f"  Raw samples:     {len(all_ex)}")
        print(f"  After filtering: {len(filtered)}")
        print(f"  Eligible:        {len(eligible)}")
        print(f"  Split: {n_train} train / {n_val} val / {n_test} test")
        print(f"  Context length (words): mean={np.mean(ctx_lengths):.0f}, "
              f"median={np.median(ctx_lengths):.0f}, "
              f"range=[{np.min(ctx_lengths)}, {np.max(ctx_lengths)}]")
        print(f"  Answer length (words):  mean={np.mean(ans_word_counts):.1f}, "
              f"median={np.median(ans_word_counts):.0f}, "
              f"max={np.max(ans_word_counts)}")
        print(f"  Answer types: {dict(type_counter)}")

    # Print totals
    total_train = sum(s["n_train"] for s in stats.values())
    total_val = sum(s["n_val"] for s in stats.values())
    total_test = sum(s["n_test"] for s in stats.values())
    total_eligible = sum(s["eligible"] for s in stats.values())
    print(f"\n{'='*60}")
    print(f"  TOTALS")
    print(f"{'='*60}")
    print(f"  Total eligible: {total_eligible}")
    print(f"  Total split: {total_train} train / {total_val} val / {total_test} test")
    print(f"  (Expected: 306 train / 64 val / 69 test = 439)")
    print(f"  Note: Exact counts depend on tokenizer-based filtering;")
    print(f"        word-based approximation gives slightly different numbers.")

    return stats


# ---------------------------------------------------------------------------
# Plot 1: Dataset characteristics summary table
# ---------------------------------------------------------------------------
def plot_characteristics_table(stats):
    """Create a summary table-as-figure showing per-task characteristics."""

    # Task metadata (manually annotated based on task analysis)
    task_meta = {
        "qasper": {
            "source": "Qasper (Dasigi et al., 2021)",
            "domain": "Scientific papers",
            "question_type": "Varied (factoid, yes/no,\nunanswerable)",
            "answer_type": "Phrase / sentence /\nyes-no / unanswerable",
            "info_locality": "Localised\n(specific section)",
            "eviction_sensitivity": "MEDIUM",
            "reasoning": "Single-passage\nretrieval",
        },
        "2wikimqa": {
            "source": "2WikiMultihopQA\n(Ho et al., 2020)",
            "domain": "Wikipedia passages",
            "question_type": "Multi-hop\n(bridge/comparison)",
            "answer_type": "Short factoid\n(1-3 words)",
            "info_locality": "Distributed\n(2+ passages)",
            "eviction_sensitivity": "HIGH",
            "reasoning": "Multi-hop\nreasoning",
        },
        "qasper_e": {
            "source": "Qasper-E\n(LongBench-E)",
            "domain": "Scientific papers",
            "question_type": "Varied (factoid, yes/no,\nunanswerable)",
            "answer_type": "Phrase / sentence /\nyes-no / unanswerable",
            "info_locality": "Localised\n(specific section)",
            "eviction_sensitivity": "MEDIUM",
            "reasoning": "Single-passage\nretrieval",
        },
        "hotpotqa_e": {
            "source": "HotpotQA-E\n(Yang et al., 2018)",
            "domain": "Wikipedia passages",
            "question_type": "Multi-hop\n(bridge/comparison)",
            "answer_type": "Short factoid\n(1-3 words)",
            "info_locality": "Distributed\n(2 passages)",
            "eviction_sensitivity": "HIGH",
            "reasoning": "Multi-hop\nreasoning",
        },
        "2wikimqa_e": {
            "source": "2WikiMQA-E\n(LongBench-E)",
            "domain": "Wikipedia passages",
            "question_type": "Multi-hop\n(bridge/comparison)",
            "answer_type": "Short factoid\n(1-3 words)",
            "info_locality": "Distributed\n(2+ passages)",
            "eviction_sensitivity": "HIGH",
            "reasoning": "Multi-hop\nreasoning",
        },
    }

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_axis_off()

    # Column headers
    columns = [
        "Task", "Source", "Domain", "Question\nType", "Answer\nType",
        "Eligible\nSamples", "Train/Val/\nTest",
        "Ctx Length\n(words, mean)",
        "Ans Length\n(words, mean)",
        "Information\nLocality",
        "Eviction\nSensitivity",
    ]

    n_cols = len(columns)
    n_rows = len(TASKS) + 1  # +1 for header

    # Column widths (proportional)
    col_widths = [0.07, 0.11, 0.08, 0.10, 0.10, 0.06, 0.08, 0.08, 0.07, 0.10, 0.08]
    col_widths = [w / sum(col_widths) for w in col_widths]  # normalise

    # Colours
    header_color = "#2c3e50"
    header_text_color = "white"
    row_colors = ["#ecf0f1", "#ffffff"]
    sensitivity_colors = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}

    cell_height = 0.14
    start_y = 0.92

    # Draw header
    x_pos = 0.02
    for j, (col, w) in enumerate(zip(columns, col_widths)):
        rect = FancyBboxPatch(
            (x_pos, start_y - cell_height), w - 0.005, cell_height,
            boxstyle="round,pad=0.003", facecolor=header_color,
            edgecolor="white", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x_pos + w / 2, start_y - cell_height / 2, col,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=header_text_color, family="monospace")
        x_pos += w

    # Draw data rows
    for i, task in enumerate(TASKS):
        s = stats[task]
        m = task_meta[task]
        y = start_y - (i + 1) * cell_height - cell_height

        row_data = [
            TASK_DISPLAY[task],
            m["source"],
            m["domain"],
            m["question_type"],
            m["answer_type"],
            str(s["eligible"]),
            f"{s['n_train']}/{s['n_val']}/{s['n_test']}",
            f"{s['ctx_length_mean']:.0f}",
            f"{s['ans_words_mean']:.1f}",
            m["info_locality"],
            m["eviction_sensitivity"],
        ]

        x_pos = 0.02
        for j, (val, w) in enumerate(zip(row_data, col_widths)):
            if j == n_cols - 1:  # eviction sensitivity column
                fc = sensitivity_colors.get(val, row_colors[i % 2])
                tc = "white" if val in sensitivity_colors else "black"
            else:
                fc = row_colors[i % 2]
                tc = "black"

            rect = FancyBboxPatch(
                (x_pos, y), w - 0.005, cell_height,
                boxstyle="round,pad=0.003", facecolor=fc,
                edgecolor="#bdc3c7", linewidth=0.8
            )
            ax.add_patch(rect)

            fontsize = 7 if len(val) > 25 else 8
            ax.text(x_pos + w / 2, y + cell_height / 2, val,
                    ha="center", va="center", fontsize=fontsize,
                    color=tc, family="monospace")
            x_pos += w

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    fig.suptitle("Dataset Characteristics: LongBench 5-Task QA Subset",
                 fontsize=14, fontweight="bold", y=0.98)

    # Add footnotes
    footnote = (
        "Filtering: min_conditioning_length=4096 tokens, max_conditioning_length=6500 tokens, "
        "max_answer_tokens=64. Word-based approximation (x1.3 token/word ratio).\n"
        "Eviction sensitivity: expected degradation when ~75% of KV-cache tokens are evicted "
        "(cache_size=1024, context=4096-6500 tokens)."
    )
    ax.text(0.5, 0.04, footnote, ha="center", va="center", fontsize=7,
            style="italic", color="#7f8c8d", transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "dataset_characteristics.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Prompt templates
# ---------------------------------------------------------------------------
def plot_prompt_templates():
    """Show prompt templates for each task as a clean figure."""

    fig, axes = plt.subplots(5, 1, figsize=(16, 22))
    fig.suptitle("Prompt Templates for Each Task",
                 fontsize=16, fontweight="bold", y=0.995)

    for i, task in enumerate(TASKS):
        ax = axes[i]
        ax.set_axis_off()

        template = PROMPT_TEMPLATES[task]
        # Replace placeholders for readability
        template_display = template.replace("{context}", "[CONTEXT]")
        template_display = template_display.replace("{input}", "[QUESTION]")

        # Wrap text
        wrapped = textwrap.fill(template_display, width=110)

        # Task header
        ax.text(0.02, 0.95, f"{TASK_DISPLAY[task]}",
                fontsize=12, fontweight="bold", va="top",
                transform=ax.transAxes, color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#3498db",
                          edgecolor="none", alpha=0.2))

        # Template text
        ax.text(0.02, 0.78, wrapped,
                fontsize=8.5, va="top", transform=ax.transAxes,
                family="monospace", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1",
                          edgecolor="#bdc3c7", linewidth=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.5)
    out_path = os.path.join(OUT_DIR, "prompt_templates.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Length distributions
# ---------------------------------------------------------------------------
def plot_length_distributions(stats):
    """Plot context and answer length distributions per task."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    # Context length distributions
    ax = axes[0]
    for i, task in enumerate(TASKS):
        ctx = stats[task]["ctx_lengths"]
        ax.hist(ctx, bins=30, alpha=0.5, label=TASK_DISPLAY[task],
                color=colors[i], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Context Length (words)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Context Length Distribution (Eligible Samples)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.axvline(x=MIN_COND_LENGTH / TOKEN_WORD_RATIO, color="gray",
               linestyle="--", alpha=0.7, label=f"Min filter (~{MIN_COND_LENGTH/TOKEN_WORD_RATIO:.0f} words)")
    ax.axvline(x=MAX_COND_LENGTH / TOKEN_WORD_RATIO, color="gray",
               linestyle=":", alpha=0.7, label=f"Max filter (~{MAX_COND_LENGTH/TOKEN_WORD_RATIO:.0f} words)")
    ax.legend(fontsize=9)

    # Answer length distributions
    ax = axes[1]
    positions = np.arange(len(TASKS))
    bp_data = [stats[t]["ans_word_counts"] for t in TASKS]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in TASKS], fontsize=10)
    ax.set_ylabel("Answer Length (words)", fontsize=11)
    ax.set_title("Answer Length Distribution (Eligible Samples)", fontsize=13,
                 fontweight="bold")

    # Add mean annotations
    for i, task in enumerate(TASKS):
        mean_val = stats[task]["ans_words_mean"]
        ax.text(i, mean_val + 0.5, f"mean={mean_val:.1f}",
                ha="center", fontsize=8, color=colors[i], fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "length_distributions.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Answer type breakdown
# ---------------------------------------------------------------------------
def plot_answer_types(stats):
    """Stacked bar chart of answer types per task."""
    fig, ax = plt.subplots(figsize=(12, 6))

    answer_categories = ["short factoid", "phrase", "sentence+", "yes/no", "unanswerable"]
    colors_map = {
        "short factoid": "#3498db",
        "phrase": "#2ecc71",
        "sentence+": "#f39c12",
        "yes/no": "#e74c3c",
        "unanswerable": "#95a5a6",
    }

    x = np.arange(len(TASKS))
    width = 0.6
    bottoms = np.zeros(len(TASKS))

    for cat in answer_categories:
        values = []
        for task in TASKS:
            type_counts = stats[task]["ans_type_counts"]
            n = stats[task]["eligible"]
            count = type_counts.get(cat, 0)
            values.append(100 * count / n if n > 0 else 0)
        values = np.array(values)
        ax.bar(x, values, width, bottom=bottoms, label=cat,
               color=colors_map[cat], edgecolor="white", linewidth=0.5)

        # Add percentage labels for non-trivial segments
        for j, (v, b) in enumerate(zip(values, bottoms)):
            if v > 5:
                ax.text(j, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY[t] for t in TASKS], fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Answer Type Distribution by Task (Eligible Samples)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "answer_types.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Eviction impact analysis
# ---------------------------------------------------------------------------
def plot_eviction_analysis(stats):
    """Visualise how much context is evicted at different cache sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cache_sizes = [1024, 2048, 3072]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    # Panel 1: % of tokens evicted per task at each cache size
    ax = axes[0]
    x = np.arange(len(cache_sizes))
    width = 0.15
    offsets = np.linspace(-width * 2, width * 2, len(TASKS))

    for i, task in enumerate(TASKS):
        # Mean context in tokens (approx)
        mean_ctx_tokens = stats[task]["ctx_length_mean"] * TOKEN_WORD_RATIO
        eviction_pcts = []
        for cs in cache_sizes:
            pct_evicted = max(0, (mean_ctx_tokens - cs) / mean_ctx_tokens * 100)
            eviction_pcts.append(pct_evicted)
        ax.bar(x + offsets[i], eviction_pcts, width, label=TASK_DISPLAY[task],
               color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"cs={cs}" for cs in cache_sizes], fontsize=10)
    ax.set_ylabel("Tokens Evicted (%)", fontsize=11)
    ax.set_title("Mean % Tokens Evicted by Cache Size", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # Panel 2: tokens retained vs total context
    ax = axes[1]
    for i, task in enumerate(TASKS):
        ctx_tokens = stats[task]["ctx_lengths"] * TOKEN_WORD_RATIO
        sorted_ctx = np.sort(ctx_tokens)

        for cs_idx, cs in enumerate(cache_sizes):
            retained_pct = np.minimum(cs / sorted_ctx * 100, 100)
            if cs_idx == 0:  # only label once
                label = TASK_DISPLAY[task]
            else:
                label = None
            linestyle = ["-", "--", ":"][cs_idx]
            if i == 0:  # Only show cache-size labels for first task
                ax.plot([], [], linestyle=linestyle, color="gray",
                        label=f"cs={cs}")

    # Simplified: show distribution of eviction rates at cs=1024
    for i, task in enumerate(TASKS):
        ctx_tokens = stats[task]["ctx_lengths"] * TOKEN_WORD_RATIO
        retained_pct = np.minimum(1024 / ctx_tokens * 100, 100)
        ax.hist(retained_pct, bins=20, alpha=0.5, label=TASK_DISPLAY[task],
                color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("% of Context Retained (cache_size=1024)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Context Retention at cs=1024",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "eviction_analysis.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  LongBench 5-Task Dataset Characteristics Analysis")
    print("=" * 60)

    stats = analyse()

    print("\n\nGenerating plots...")
    plot_characteristics_table(stats)
    plot_prompt_templates()
    plot_length_distributions(stats)
    plot_answer_types(stats)
    plot_eviction_analysis(stats)

    print("\nDone. All outputs saved to:", OUT_DIR)
