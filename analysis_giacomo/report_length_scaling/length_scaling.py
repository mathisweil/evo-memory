"""
D2: Length scaling analysis — systematic test vs extended_test comparison.
Tests whether NAMM's advantage changes as context grows from ~5K to ~8K tokens.
Output: analysis_giacomo/report_length_scaling/
"""

import json
from collections import defaultdict
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

# ── load all_results.json ──────────────────────────────────────────────
with open(RESULTS / "all_results.json") as f:
    ALL_RESULTS = json.load(f)

# ── conditions and tasks ───────────────────────────────────────────────
CONDITIONS = list(ALL_RESULTS.keys())
TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_SHORT = {
    "lb/qasper": "Qasper",
    "lb/2wikimqa": "2WikiMQA",
    "lb/qasper_e": "Qasper-E",
    "lb/hotpotqa_e": "HotpotQA-E",
    "lb/2wikimqa_e": "2WikiMQA-E",
}

# Group conditions by type for cleaner analysis
COND_GROUPS = {
    "Baselines": ["B0", "B1/cs1024", "B1/cs2048"],
    "Truncation": ["Trunc/plain_1024", "Trunc/plain_2048"],
    "LoRA-only": ["Trunc/lora_m1_1024", "Trunc/lora_m1_2048", "A4/cs1024_no_namm", "A4/cs2048_no_namm"],
    "NAMM-only": ["M1", "M2/cs1024", "M2/cs2048"],
    "NAMM+LoRA": ["M4/cs1024", "M4/cs2048"],
}

KEY_CONDS = ["B0", "M1", "M4/cs1024", "M4/cs2048", "A4/cs1024_no_namm", "A4/cs2048_no_namm",
             "Trunc/lora_m1_2048", "B1/cs1024"]

# Nice display names
DISPLAY = {
    "B0": "B0 (vanilla)",
    "B1/cs1024": "B1/1K (rand NAMM)",
    "B1/cs2048": "B1/2K (rand NAMM)",
    "M1": "M1 (NAMM-only)",
    "M2/cs1024": "M2/1K (rand LoRA+NAMM)",
    "M2/cs2048": "M2/2K (rand LoRA+NAMM)",
    "M4/cs1024": "M4/1K (LoRA+NAMM)",
    "M4/cs2048": "M4/2K (LoRA+NAMM)",
    "A4/cs1024_no_namm": "A4/1K (LoRA-only)",
    "A4/cs2048_no_namm": "A4/2K (LoRA-only)",
    "Trunc/plain_1024": "Trunc/1K",
    "Trunc/plain_2048": "Trunc/2K",
    "Trunc/lora_m1_1024": "Trunc+LoRA/1K",
    "Trunc/lora_m1_2048": "Trunc+LoRA/2K",
    "M1_recency/cs1024": "M1rec/1K (broken)",
}


# ── analysis 1: test vs extended_test delta ────────────────────────────
def build_scaling_table() -> pd.DataFrame:
    """Build a table of test/ext_test F1 per condition per task + deltas."""
    rows = []
    for cond in CONDITIONS:
        if cond not in ALL_RESULTS:
            continue
        test = ALL_RESULTS[cond].get("test", {})
        ext = ALL_RESULTS[cond].get("extended_test", {})
        for task in TASKS:
            t_f1 = test.get(task, None)
            e_f1 = ext.get(task, None)
            if t_f1 is not None and e_f1 is not None:
                rows.append({
                    "condition": cond,
                    "display": DISPLAY.get(cond, cond),
                    "task": TASK_SHORT[task],
                    "test_f1": t_f1,
                    "ext_f1": e_f1,
                    "delta": e_f1 - t_f1,
                    "pct_change": ((e_f1 - t_f1) / t_f1 * 100) if t_f1 > 0 else float("nan"),
                })
        # Also add mean
        t_mean = test.get("mean", None)
        e_mean = ext.get("mean", None)
        if t_mean is not None and e_mean is not None:
            rows.append({
                "condition": cond,
                "display": DISPLAY.get(cond, cond),
                "task": "Mean",
                "test_f1": t_mean,
                "ext_f1": e_mean,
                "delta": e_mean - t_mean,
                "pct_change": ((e_mean - t_mean) / t_mean * 100) if t_mean > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


# ── analysis 2: per-sample length bins ─────────────────────────────────
def load_generations_with_length(cond_name: str, gen_path: Path, split: str):
    """Load generations and return list of (prompt_idx, f1, length, task)."""
    import re, string
    from collections import Counter

    def normalize(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = s.translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())

    def tok_f1(pred, gold):
        pt = normalize(pred).split()
        gt = normalize(gold).split()
        if not gt and not pt: return 1.0
        if not gt or not pt: return 0.0
        c = sum((Counter(pt) & Counter(gt)).values())
        if c == 0: return 0.0
        p, r = c/len(pt), c/len(gt)
        return 2*p*r/(p+r)

    if not gen_path.exists():
        return []
    with open(gen_path) as f:
        raw = json.load(f)
    if split not in raw:
        return []
    items = []
    for task in TASKS:
        if task not in raw[split]:
            continue
        for it in raw[split][task]:
            f1 = max(tok_f1(it["pred"], a) for a in it["answers"]) if it["answers"] else 0.0
            items.append({
                "condition": cond_name,
                "task": TASK_SHORT[task],
                "prompt_idx": it["prompt_idx"],
                "f1": f1,
                "length": it.get("length", 0),
            })
    return items


# ── generation paths ───────────────────────────────────────────────────
GEN_PATHS = {
    "B0":                RESULTS / "B0" / "generations.json",
    "M1":                RESULTS / "M1" / "generations.json",
    "M4/cs1024":         RESULTS / "M4" / "cs1024" / "generations.json",
    "M4/cs2048":         RESULTS / "M4" / "cs2048" / "generations.json",
    "M2/cs1024":         RESULTS / "M2" / "cs1024" / "generations.json",
    "B1/cs1024":         RESULTS / "B1" / "cs1024" / "generations.json",
    "A4/cs1024_no_namm": RESULTS / "A4" / "cs1024_no_namm" / "generations.json",
    "A4/cs2048_no_namm": RESULTS / "A4" / "cs2048_no_namm" / "generations.json",
    "Trunc/lora_m1_2048": RESULTS / "Trunc" / "lora_m1_2048" / "generations.json",
}


# ── plotting ───────────────────────────────────────────────────────────

def plot_test_vs_ext_bars(df: pd.DataFrame, filename: str):
    """Grouped bar chart: test vs ext F1 by condition (mean across tasks)."""
    mean_df = df[df["task"] == "Mean"].copy()
    mean_df = mean_df.sort_values("test_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(mean_df))
    h = 0.35
    ax.barh(y - h/2, mean_df["test_f1"], h, label="test (4K-6.5K)", color="#3498db", edgecolor="white")
    ax.barh(y + h/2, mean_df["ext_f1"], h, label="extended_test (up to 8K)", color="#e74c3c", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(mean_df["display"], fontsize=8)
    ax.set_xlabel("Mean F1 (%)")
    ax.set_title("Test vs Extended Test: Mean F1 across tasks")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_scaling_delta_heatmap(df: pd.DataFrame, filename: str):
    """Heatmap of F1 deltas (ext - test) per condition x task."""
    pivot = df.pivot(index="condition", columns="task", values="delta")
    # Order by mean delta
    mean_delta = pivot.mean(axis=1).sort_values()
    pivot = pivot.loc[mean_delta.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-15, vmax=15)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    disp = [DISPLAY.get(c, c) for c in pivot.index]
    ax.set_yticklabels(disp, fontsize=8)
    ax.set_title("F1 change: extended_test - test (green = improves at longer context)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                       color="black" if abs(v) < 8 else "white")

    plt.colorbar(im, ax=ax, shrink=0.7, label="F1 delta")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_scaling_slopes(df: pd.DataFrame, conditions: list[str], filename: str):
    """Slope chart: test → ext_test F1 for each condition (per task)."""
    tasks = [t for t in df["task"].unique() if t != "Mean"]
    n_tasks = len(tasks)

    fig, axes = plt.subplots(1, n_tasks, figsize=(3.5 * n_tasks, 5), sharey=False)
    cmap = plt.cm.tab10

    for ti, task in enumerate(tasks):
        ax = axes[ti]
        sub = df[(df["task"] == task) & (df["condition"].isin(conditions))]
        for ci, (_, row) in enumerate(sub.iterrows()):
            color = cmap(ci % 10)
            ax.plot([0, 1], [row["test_f1"], row["ext_f1"]], "o-",
                   color=color, label=DISPLAY.get(row["condition"], row["condition"]),
                   linewidth=1.5, markersize=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["test\n(4K-6.5K)", "ext_test\n(up to 8K)"], fontsize=8)
        ax.set_ylabel("F1 (%)")
        ax.set_title(task, fontsize=10)
        if ti == 0:
            ax.legend(fontsize=6, loc="best")

    plt.suptitle("F1 scaling: test → extended_test", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_length_binned_f1(all_items: list[dict], conditions: list[str], filename: str):
    """F1 vs length bin for key conditions (extended_test only, has range)."""
    df = pd.DataFrame(all_items)
    df = df[df["condition"].isin(conditions)]

    # Create length bins
    bins = [0, 4096, 5000, 6000, 7000, 8192]
    labels = ["<4K", "4-5K", "5-6K", "6-7K", "7-8K"]
    df["len_bin"] = pd.cut(df["length"], bins=bins, labels=labels, right=True)

    # Aggregate
    agg = df.groupby(["condition", "len_bin"], observed=True)["f1"].agg(["mean", "count"]).reset_index()
    agg["mean"] *= 100  # to percentage

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10
    for ci, cond in enumerate(conditions):
        sub = agg[agg["condition"] == cond]
        if len(sub) < 2:
            continue
        ax.plot(sub["len_bin"].astype(str), sub["mean"], "o-",
               label=DISPLAY.get(cond, cond), color=cmap(ci), linewidth=1.5)

    ax.set_xlabel("Context length bin")
    ax.set_ylabel("Mean F1 (%)")
    ax.set_title("F1 by context length bin (extended_test, all tasks)")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_degradation_rate(df: pd.DataFrame, filename: str):
    """Which conditions degrade most from test to extended_test?"""
    mean_df = df[df["task"] == "Mean"].copy()
    mean_df = mean_df.sort_values("delta")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in mean_df["delta"]]
    ax.barh(range(len(mean_df)), mean_df["delta"], color=colors, edgecolor="white")

    ax.set_yticks(range(len(mean_df)))
    ax.set_yticklabels([DISPLAY.get(c, c) for c in mean_df["condition"]], fontsize=8)
    ax.set_xlabel("Mean F1 change (extended_test - test)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Degradation at longer contexts (red = worse at longer context)")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def write_scaling_report(df: pd.DataFrame, all_items_ext: list[dict]):
    """Write markdown summary report."""
    lines = ["# D2: Length Scaling Analysis\n\n"]
    lines.append("Systematic comparison of test (4K-6.5K tokens) vs extended_test (up to 8K tokens).\n\n")

    # Summary table
    mean_df = df[df["task"] == "Mean"].sort_values("delta")
    lines.append("## Overall scaling (mean F1 across tasks)\n\n")
    lines.append("| Condition | Test F1 | Ext F1 | Delta | %Change |\n")
    lines.append("|---|---|---|---|---|\n")
    for _, row in mean_df.iterrows():
        lines.append(f"| {DISPLAY.get(row['condition'], row['condition'])} | "
                    f"{row['test_f1']:.1f} | {row['ext_f1']:.1f} | "
                    f"{row['delta']:+.1f} | {row['pct_change']:+.1f}% |\n")

    # Per-task breakdown for key conditions
    lines.append("\n## Per-task deltas (ext - test F1)\n\n")
    key = ["B0", "M1", "M4/cs1024", "A4/cs1024_no_namm", "Trunc/lora_m1_2048"]
    for cond in key:
        sub = df[(df["condition"] == cond) & (df["task"] != "Mean")]
        if sub.empty:
            continue
        lines.append(f"\n### {DISPLAY.get(cond, cond)}\n")
        lines.append("| Task | Test F1 | Ext F1 | Delta |\n")
        lines.append("|---|---|---|---|\n")
        for _, row in sub.iterrows():
            lines.append(f"| {row['task']} | {row['test_f1']:.1f} | {row['ext_f1']:.1f} | {row['delta']:+.1f} |\n")

    # Key findings
    lines.append("\n## Key findings\n\n")

    # Which conditions are most robust?
    robust = mean_df.nlargest(3, "delta")
    fragile = mean_df.nsmallest(3, "delta")
    lines.append("**Most robust to longer context** (smallest degradation or improvement):\n")
    for _, row in robust.iterrows():
        lines.append(f"- {DISPLAY.get(row['condition'], row['condition'])}: {row['delta']:+.1f}\n")
    lines.append("\n**Most fragile at longer context**:\n")
    for _, row in fragile.iterrows():
        lines.append(f"- {DISPLAY.get(row['condition'], row['condition'])}: {row['delta']:+.1f}\n")

    # Length bin analysis
    if all_items_ext:
        df_ext = pd.DataFrame(all_items_ext)
        bins = [0, 5000, 6500, 8192]
        labels = ["short (<5K)", "medium (5-6.5K)", "long (6.5-8K)"]
        df_ext["len_bin"] = pd.cut(df_ext["length"], bins=bins, labels=labels, right=True)

        lines.append("\n## F1 by length bin (extended_test)\n\n")
        for cond in key:
            sub = df_ext[df_ext["condition"] == cond]
            if sub.empty:
                continue
            agg = sub.groupby("len_bin", observed=True)["f1"].agg(["mean", "count"])
            lines.append(f"\n**{DISPLAY.get(cond, cond)}**:\n")
            for bin_label in labels:
                if bin_label in agg.index:
                    lines.append(f"- {bin_label}: F1={agg.loc[bin_label, 'mean']*100:.1f}% (n={int(agg.loc[bin_label, 'count'])})\n")

    outpath = OUT / "scaling_report.md"
    with open(outpath, "w") as f:
        f.writelines(lines)
    print(f"  Saved scaling_report.md")


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("D2: Length scaling analysis")
    print("=" * 60)

    # 1. Build scaling table from all_results.json
    print("\n[1] Building scaling table...")
    df = build_scaling_table()
    df.to_csv(OUT / "scaling_table.csv", index=False)
    print(f"  {len(df)} rows saved to scaling_table.csv")

    # 2. Plots from aggregate data
    print("\n[2] Plotting test vs ext bars...")
    plot_test_vs_ext_bars(df, "test_vs_ext_bars.png")

    print("[3] Plotting delta heatmap...")
    plot_scaling_delta_heatmap(df, "scaling_delta_heatmap.png")

    print("[4] Plotting scaling slopes...")
    plot_scaling_slopes(df, KEY_CONDS, "scaling_slopes.png")

    print("[5] Plotting degradation rate...")
    plot_degradation_rate(df, "degradation_rate.png")

    # 3. Per-sample length binned analysis (extended_test only — wider range)
    print("\n[6] Loading generations for length-binned analysis...")
    all_items = []
    for cond, path in GEN_PATHS.items():
        items = load_generations_with_length(cond, path, "extended_test")
        all_items.extend(items)
        if items:
            print(f"  {cond}: {len(items)} samples")

    if all_items:
        plot_length_binned_f1(all_items, list(GEN_PATHS.keys()), "f1_by_length_bin.png")

    # 4. Write summary report
    print("\n[7] Writing scaling report...")
    write_scaling_report(df, all_items)

    print("\n" + "=" * 60)
    print("D2 complete. All outputs in:")
    print(f"  {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
