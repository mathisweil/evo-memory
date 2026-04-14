"""
D1: Generation-level error analysis across experimental conditions.
Classifies predictions and builds per-sample win/loss matrices.
Output: analysis_giacomo/report_error_analysis/
"""

import json
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "main_table_5t"
OUT = Path(__file__).resolve().parent

# ── conditions to load ─────────────────────────────────────────────────
CONDITION_PATHS = {
    "B0":                RESULTS / "B0" / "generations.json",
    "M1":                RESULTS / "M1" / "generations.json",
    "M4/cs1024":         RESULTS / "M4" / "cs1024" / "generations.json",
    "M4/cs2048":         RESULTS / "M4" / "cs2048" / "generations.json",
    "M2/cs1024":         RESULTS / "M2" / "cs1024" / "generations.json",
    "M2/cs2048":         RESULTS / "M2" / "cs2048" / "generations.json",
    "B1/cs1024":         RESULTS / "B1" / "cs1024" / "generations.json",
    "B1/cs2048":         RESULTS / "B1" / "cs2048" / "generations.json",
    "A4/cs1024":         RESULTS / "A4" / "cs1024_no_namm" / "generations.json",
    "A4/cs2048":         RESULTS / "A4" / "cs2048_no_namm" / "generations.json",
    "Trunc/plain_1024":  RESULTS / "Trunc" / "plain_1024" / "generations.json",
    "Trunc/plain_2048":  RESULTS / "Trunc" / "plain_2048" / "generations.json",
    "Trunc/lora_1024":   RESULTS / "Trunc" / "lora_m1_1024" / "generations.json",
    "Trunc/lora_2048":   RESULTS / "Trunc" / "lora_m1_2048" / "generations.json",
}

# Key conditions for deep comparison
KEY_CONDS = ["B0", "M1", "M4/cs1024", "M4/cs2048", "A4/cs1024", "Trunc/lora_2048"]

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_SHORT = {
    "lb/qasper": "Qasper",
    "lb/2wikimqa": "2WikiMQA",
    "lb/qasper_e": "Qasper-E",
    "lb/hotpotqa_e": "HotpotQA-E",
    "lb/2wikimqa_e": "2WikiMQA-E",
}


# ── token F1 (matches the eval code) ──────────────────────────────────
def normalize_answer(s: str) -> str:
    """Lower text, remove articles/punctuation, collapse whitespace."""
    s = s.lower()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    s = " ".join(s.split())
    return s


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(pred_toks)
    rec = n_common / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def best_f1(pred: str, answers: list[str]) -> float:
    """Max token-F1 over all reference answers."""
    return max(token_f1(pred, a) for a in answers) if answers else 0.0


# ── error classification ──────────────────────────────────────────────
def classify_prediction(pred: str, answers: list[str], f1: float) -> str:
    """Classify a single prediction into error categories."""
    pred_norm = normalize_answer(pred)
    ans_norms = [normalize_answer(a) for a in answers]

    is_unanswerable_gold = any("unanswerable" in a for a in ans_norms)
    is_unanswerable_pred = "unanswerable" in pred_norm

    if f1 >= 0.8:
        return "correct"
    elif f1 >= 0.4:
        return "partial"
    elif is_unanswerable_gold and is_unanswerable_pred:
        return "correct"  # both say unanswerable
    elif is_unanswerable_gold and not is_unanswerable_pred:
        return "hallucination"  # gold says unanswerable, model fabricates
    elif not is_unanswerable_gold and is_unanswerable_pred:
        return "abstention"  # model refuses when answer exists
    elif f1 > 0:
        return "partial"
    else:
        return "wrong"


# ── load all generations ──────────────────────────────────────────────
def load_generations(split: str = "test") -> dict:
    """Returns {condition: {task: [{prompt_idx, pred, answers, length, f1, error_type}, ...]}}"""
    data = {}
    for cond, path in CONDITION_PATHS.items():
        if not path.exists():
            print(f"  SKIP {cond}: {path} not found")
            continue
        with open(path) as f:
            raw = json.load(f)
        if split not in raw:
            print(f"  SKIP {cond}: split '{split}' not in file")
            continue
        cond_data = {}
        for task in TASKS:
            if task not in raw[split]:
                continue
            items = []
            for item in raw[split][task]:
                f1 = best_f1(item["pred"], item["answers"])
                etype = classify_prediction(item["pred"], item["answers"], f1)
                items.append({
                    "prompt_idx": item["prompt_idx"],
                    "pred": item["pred"],
                    "answers": item["answers"],
                    "length": item.get("length", 0),
                    "f1": f1,
                    "error_type": etype,
                })
            cond_data[task] = items
        data[cond] = cond_data
    return data


# ── analysis functions ─────────────────────────────────────────────────

def error_type_breakdown(data: dict, split: str) -> pd.DataFrame:
    """Per-condition, per-task error type counts."""
    rows = []
    for cond, tasks in data.items():
        for task, items in tasks.items():
            counts = Counter(it["error_type"] for it in items)
            n = len(items)
            for etype in ["correct", "partial", "wrong", "hallucination", "abstention"]:
                rows.append({
                    "condition": cond,
                    "task": TASK_SHORT.get(task, task),
                    "error_type": etype,
                    "count": counts.get(etype, 0),
                    "pct": 100 * counts.get(etype, 0) / n if n > 0 else 0,
                })
    return pd.DataFrame(rows)


def per_sample_matrix(data: dict, task: str, conditions: list[str]) -> pd.DataFrame:
    """Build prompt_idx x condition matrix of F1 scores."""
    # Collect all prompt_idxs across conditions
    all_idxs = set()
    for cond in conditions:
        if cond in data and task in data[cond]:
            for item in data[cond][task]:
                all_idxs.add(item["prompt_idx"])

    rows = []
    for idx in sorted(all_idxs):
        row = {"prompt_idx": idx}
        for cond in conditions:
            if cond in data and task in data[cond]:
                match = [it for it in data[cond][task] if it["prompt_idx"] == idx]
                row[cond] = match[0]["f1"] if match else np.nan
            else:
                row[cond] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("prompt_idx")


def find_divergent_samples(data: dict, task: str, cond_a: str, cond_b: str, top_k: int = 10):
    """Find samples where cond_a and cond_b differ most."""
    mat = per_sample_matrix(data, task, [cond_a, cond_b])
    mat["delta"] = mat[cond_a] - mat[cond_b]
    mat = mat.dropna()

    # Samples where A wins
    a_wins = mat.nlargest(top_k, "delta")
    # Samples where B wins
    b_wins = mat.nsmallest(top_k, "delta")
    return a_wins, b_wins


def prediction_length_stats(data: dict) -> pd.DataFrame:
    """Average prediction length (in tokens) per condition per task."""
    rows = []
    for cond, tasks in data.items():
        for task, items in tasks.items():
            pred_lens = [len(normalize_answer(it["pred"]).split()) for it in items]
            rows.append({
                "condition": cond,
                "task": TASK_SHORT.get(task, task),
                "mean_pred_tokens": np.mean(pred_lens) if pred_lens else 0,
                "median_pred_tokens": np.median(pred_lens) if pred_lens else 0,
                "mean_f1": np.mean([it["f1"] for it in items]) if items else 0,
            })
    return pd.DataFrame(rows)


# ── plotting functions ─────────────────────────────────────────────────

def plot_error_type_heatmap(breakdown: pd.DataFrame, conditions: list[str], filename: str):
    """Stacked bar chart of error types per condition, aggregated across tasks."""
    agg = breakdown[breakdown["condition"].isin(conditions)].groupby(
        ["condition", "error_type"]
    )["count"].sum().unstack(fill_value=0)

    # Reorder
    order = [c for c in conditions if c in agg.index]
    etypes = ["correct", "partial", "wrong", "hallucination", "abstention"]
    cols = [e for e in etypes if e in agg.columns]
    agg = agg.loc[order, cols]

    # Normalize to percentages
    agg_pct = agg.div(agg.sum(axis=1), axis=0) * 100

    colors = {
        "correct": "#2ecc71",
        "partial": "#f1c40f",
        "wrong": "#e74c3c",
        "hallucination": "#9b59b6",
        "abstention": "#3498db",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(agg_pct))
    for etype in cols:
        vals = agg_pct[etype].values
        ax.barh(range(len(agg_pct)), vals, left=bottom,
                color=colors.get(etype, "#95a5a6"), label=etype, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_yticks(range(len(agg_pct)))
    ax.set_yticklabels(agg_pct.index)
    ax.set_xlabel("% of predictions")
    ax.set_title(f"Error type breakdown (all tasks, {len(breakdown['task'].unique())} tasks)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_error_type_by_task(breakdown: pd.DataFrame, conditions: list[str], filename: str):
    """Per-task error type comparison for key conditions."""
    tasks_short = list(TASK_SHORT.values())
    n_tasks = len(tasks_short)
    n_conds = len(conditions)

    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 5), sharey=True)
    colors = {
        "correct": "#2ecc71",
        "partial": "#f1c40f",
        "wrong": "#e74c3c",
        "hallucination": "#9b59b6",
        "abstention": "#3498db",
    }
    etypes = ["correct", "partial", "wrong", "hallucination", "abstention"]

    for ti, task in enumerate(tasks_short):
        ax = axes[ti]
        sub = breakdown[(breakdown["task"] == task) & (breakdown["condition"].isin(conditions))]
        pivot = sub.pivot(index="condition", columns="error_type", values="pct").fillna(0)
        order = [c for c in conditions if c in pivot.index]
        cols = [e for e in etypes if e in pivot.columns]
        pivot = pivot.loc[order, cols]

        bottom = np.zeros(len(pivot))
        for etype in cols:
            vals = pivot[etype].values
            ax.barh(range(len(pivot)), vals, left=bottom,
                    color=colors.get(etype, "#95a5a6"), label=etype if ti == 0 else "",
                    edgecolor="white", linewidth=0.5)
            bottom += vals

        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index if ti == 0 else [])
        ax.set_xlabel("%")
        ax.set_title(task, fontsize=10)
        ax.set_xlim(0, 100)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[e]) for e in etypes]
    fig.legend(handles, etypes, loc="lower center", ncol=5, fontsize=8,
              bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("Error types by task and condition", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_win_loss_heatmap(data: dict, task: str, conditions: list[str], filename: str):
    """Pairwise win rate matrix for a single task."""
    mat = per_sample_matrix(data, task, conditions)
    n = len(conditions)
    winrate = np.zeros((n, n))

    for i, ca in enumerate(conditions):
        for j, cb in enumerate(conditions):
            if i == j:
                winrate[i, j] = 0.5
                continue
            valid = mat[[ca, cb]].dropna()
            if len(valid) == 0:
                winrate[i, j] = 0.5
                continue
            wins = (valid[ca] > valid[cb]).sum()
            ties = (valid[ca] == valid[cb]).sum()
            winrate[i, j] = (wins + 0.5 * ties) / len(valid)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(winrate, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(conditions, fontsize=8)
    ax.set_xlabel("Column condition")
    ax.set_ylabel("Row condition")
    ax.set_title(f"Pairwise win rate: {TASK_SHORT.get(task, task)}\n(row beats column)")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{winrate[i,j]:.2f}", ha="center", va="center", fontsize=7,
                   color="black" if 0.3 < winrate[i,j] < 0.7 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_f1_by_length(data: dict, task: str, conditions: list[str], filename: str):
    """Scatter: context length vs F1 for each condition on a task."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.tab10

    for ci, cond in enumerate(conditions):
        if cond not in data or task not in data[cond]:
            continue
        items = data[cond][task]
        lengths = [it["length"] for it in items]
        f1s = [it["f1"] for it in items]
        ax.scatter(lengths, f1s, alpha=0.5, s=20, label=cond, color=cmap(ci))

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Token F1")
    ax.set_title(f"F1 vs context length: {TASK_SHORT.get(task, task)}")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_pred_length_vs_f1(plstats: pd.DataFrame, conditions: list[str], filename: str):
    """Scatter: mean prediction length vs mean F1 per condition-task."""
    sub = plstats[plstats["condition"].isin(conditions)]
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.tab10

    for ci, cond in enumerate(conditions):
        csub = sub[sub["condition"] == cond]
        ax.scatter(csub["mean_pred_tokens"], csub["mean_f1"] * 100, s=40,
                  label=cond, color=cmap(ci), marker="o")
        for _, row in csub.iterrows():
            ax.annotate(row["task"], (row["mean_pred_tokens"], row["mean_f1"] * 100),
                       fontsize=6, alpha=0.7)

    ax.set_xlabel("Mean prediction length (tokens)")
    ax.set_ylabel("Mean F1 (%)")
    ax.set_title("Prediction verbosity vs accuracy")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def write_divergent_samples_report(data: dict, split: str):
    """Write markdown report of most divergent samples between M1 and M4."""
    lines = [
        f"# Divergent Samples Report ({split} split)\n",
        "Samples with the largest F1 difference between key condition pairs.\n",
    ]

    comparisons = [
        ("M1", "M4/cs1024", "lb/2wikimqa", "M4 massively beats M1 on 2WikiMQA"),
        ("M1", "M4/cs1024", "lb/qasper", "M1 beats M4 on Qasper"),
        ("B0", "M1", "lb/hotpotqa_e", "Both strong on HotpotQA-E"),
        ("M4/cs1024", "A4/cs1024", "lb/2wikimqa", "M4(NAMM+LoRA) vs A4(LoRA-only)"),
    ]

    for cond_a, cond_b, task, desc in comparisons:
        lines.append(f"\n## {cond_a} vs {cond_b} on {TASK_SHORT.get(task, task)}\n")
        lines.append(f"*{desc}*\n")

        if cond_a not in data or cond_b not in data:
            lines.append("(condition data not available)\n")
            continue
        if task not in data.get(cond_a, {}) or task not in data.get(cond_b, {}):
            lines.append("(task data not available)\n")
            continue

        a_wins, b_wins = find_divergent_samples(data, task, cond_a, cond_b, top_k=5)

        lines.append(f"### {cond_a} wins (top 5):\n")
        lines.append("| prompt_idx | {a} F1 | {b} F1 | delta |\n".format(a=cond_a, b=cond_b))
        lines.append("|---|---|---|---|\n")
        for idx, row in a_wins.iterrows():
            lines.append(f"| {idx} | {row[cond_a]:.3f} | {row[cond_b]:.3f} | {row['delta']:+.3f} |\n")

        lines.append(f"\n### {cond_b} wins (top 5):\n")
        lines.append("| prompt_idx | {a} F1 | {b} F1 | delta |\n".format(a=cond_a, b=cond_b))
        lines.append("|---|---|---|---|\n")
        for idx, row in b_wins.iterrows():
            lines.append(f"| {idx} | {row[cond_a]:.3f} | {row[cond_b]:.3f} | {row['delta']:+.3f} |\n")

        # Show actual predictions for most divergent sample
        most_div_idx = a_wins.index[0] if len(a_wins) > 0 else None
        if most_div_idx is not None:
            lines.append(f"\n**Example: prompt {most_div_idx} ({cond_a} wins)**\n")
            for cond in [cond_a, cond_b]:
                match = [it for it in data[cond][task] if it["prompt_idx"] == most_div_idx]
                if match:
                    it = match[0]
                    lines.append(f"\n*{cond}* (F1={it['f1']:.3f}, type={it['error_type']}):\n")
                    lines.append(f"- **Pred**: `{it['pred'][:200]}`\n")
                    golds = " | ".join(a[:100] for a in it["answers"])
                    lines.append(f"- **Gold**: `{golds}`\n")

        most_div_idx_b = b_wins.index[0] if len(b_wins) > 0 else None
        if most_div_idx_b is not None:
            lines.append(f"\n**Example: prompt {most_div_idx_b} ({cond_b} wins)**\n")
            for cond in [cond_a, cond_b]:
                match = [it for it in data[cond][task] if it["prompt_idx"] == most_div_idx_b]
                if match:
                    it = match[0]
                    lines.append(f"\n*{cond}* (F1={it['f1']:.3f}, type={it['error_type']}):\n")
                    lines.append(f"- **Pred**: `{it['pred'][:200]}`\n")
                    golds = " | ".join(a[:100] for a in it["answers"])
                    lines.append(f"- **Gold**: `{golds}`\n")

    outpath = OUT / f"divergent_samples_{split}.md"
    with open(outpath, "w") as f:
        f.writelines(lines)
    print(f"  Saved {outpath.name}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("D1: Generation-level error analysis")
    print("=" * 60)

    for split in ["test", "extended_test"]:
        print(f"\n--- Loading {split} split ---")
        data = load_generations(split)
        print(f"  Loaded {len(data)} conditions")

        # 1. Error type breakdown
        print("\n[1] Error type breakdown...")
        breakdown = error_type_breakdown(data, split)
        breakdown.to_csv(OUT / f"error_breakdown_{split}.csv", index=False)
        print(f"  Saved error_breakdown_{split}.csv")

        # Print summary table
        summary = breakdown.groupby(["condition", "error_type"])["count"].sum().unstack(fill_value=0)
        etypes = ["correct", "partial", "wrong", "hallucination", "abstention"]
        cols = [e for e in etypes if e in summary.columns]
        summary = summary[cols]
        summary["total"] = summary.sum(axis=1)
        summary["correct_pct"] = (summary["correct"] / summary["total"] * 100).round(1)
        print(f"\n  Summary ({split}):")
        print(summary.sort_values("correct_pct", ascending=False).to_string())

        # 2. Plots
        print("\n[2] Plotting error type heatmap...")
        plot_error_type_heatmap(breakdown, KEY_CONDS, f"error_types_overall_{split}.png")

        print("[3] Plotting per-task error types...")
        plot_error_type_by_task(breakdown, KEY_CONDS, f"error_types_by_task_{split}.png")

        # 3. Pairwise win rates for key tasks
        print("\n[4] Pairwise win rate heatmaps...")
        for task in TASKS:
            safe_name = TASK_SHORT[task].lower().replace("-", "").replace(" ", "_")
            plot_win_loss_heatmap(data, task, KEY_CONDS,
                                f"winrate_{safe_name}_{split}.png")

        # 4. F1 vs context length
        print("\n[5] F1 vs context length...")
        for task in TASKS:
            safe_name = TASK_SHORT[task].lower().replace("-", "").replace(" ", "_")
            plot_f1_by_length(data, task, KEY_CONDS,
                             f"f1_vs_length_{safe_name}_{split}.png")

        # 5. Prediction verbosity
        print("\n[6] Prediction length analysis...")
        plstats = prediction_length_stats(data)
        plstats.to_csv(OUT / f"pred_length_stats_{split}.csv", index=False)
        plot_pred_length_vs_f1(plstats, KEY_CONDS, f"verbosity_vs_f1_{split}.png")

        # 6. Divergent samples report
        print("\n[7] Divergent samples report...")
        write_divergent_samples_report(data, split)

        # 7. Per-sample cross-condition matrix for key tasks
        print("\n[8] Per-sample F1 matrices...")
        for task in ["lb/qasper", "lb/2wikimqa", "lb/hotpotqa_e"]:
            mat = per_sample_matrix(data, task, KEY_CONDS)
            safe_name = TASK_SHORT[task].lower().replace("-", "").replace(" ", "_")
            mat.to_csv(OUT / f"sample_matrix_{safe_name}_{split}.csv")
            print(f"  Saved sample_matrix_{safe_name}_{split}.csv ({len(mat)} samples)")

    # 8. Cross-split analysis: which samples break on longer contexts?
    print("\n\n--- Cross-split length analysis ---")
    test_data = load_generations("test")
    ext_data = load_generations("extended_test")

    lines = ["# Cross-split analysis: test vs extended_test\n\n"]
    lines.append("Samples appearing in both splits, showing F1 change with longer context.\n\n")

    for cond in KEY_CONDS:
        if cond not in test_data or cond not in ext_data:
            continue
        lines.append(f"\n## {cond}\n\n")
        for task in TASKS:
            if task not in test_data.get(cond, {}) or task not in ext_data.get(cond, {}):
                continue
            test_items = {it["prompt_idx"]: it for it in test_data[cond][task]}
            ext_items = {it["prompt_idx"]: it for it in ext_data[cond][task]}
            shared = set(test_items.keys()) & set(ext_items.keys())
            if not shared:
                continue
            deltas = []
            for idx in shared:
                d = ext_items[idx]["f1"] - test_items[idx]["f1"]
                deltas.append((idx, test_items[idx]["f1"], ext_items[idx]["f1"], d,
                              test_items[idx]["length"], ext_items[idx]["length"]))

            if deltas:
                mean_delta = np.mean([d[3] for d in deltas])
                degraded = sum(1 for d in deltas if d[3] < -0.1)
                improved = sum(1 for d in deltas if d[3] > 0.1)
                lines.append(f"**{TASK_SHORT[task]}**: {len(shared)} shared samples, "
                           f"mean F1 delta={mean_delta:+.3f}, "
                           f"{degraded} degraded, {improved} improved\n\n")

    with open(OUT / "cross_split_analysis.md", "w") as f:
        f.writelines(lines)
    print("  Saved cross_split_analysis.md")

    print("\n" + "=" * 60)
    print("D1 complete. All outputs in:")
    print(f"  {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
