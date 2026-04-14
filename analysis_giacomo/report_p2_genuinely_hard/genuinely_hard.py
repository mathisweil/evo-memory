"""
P2: Characterise the genuinely-hard samples — where NAMM's value lives.

For each genuinely-hard sample (only NAMM/full-context conditions succeed):
  - Task distribution
  - Document length statistics
  - Per-condition F1 (all conditions, not just the classification subset)
  - Unique-to-NAMM wins vs unique-to-Trunc/lora wins
  - Per-task NAMM advantage within this stratum

Note: "M4" in file paths = "M3" in the paper (LoRA + frozen NAMM).

Output: analysis_giacomo/report_p2_genuinely_hard/
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "main_table_5t"
OUT = Path(__file__).resolve().parent

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_SHORT = {
    "lb/qasper": "Qasper", "lb/2wikimqa": "2WikiMQA", "lb/qasper_e": "Qasper-E",
    "lb/hotpotqa_e": "HotpotQA-E", "lb/2wikimqa_e": "2WikiMQA-E",
}
SHORT_TO_FULL = {v: k for k, v in TASK_SHORT.items()}

ALL_CONDS = {
    "B0":               RESULTS / "B0",
    "B1/cs1024":        RESULTS / "B1" / "cs1024",
    "B1/cs2048":        RESULTS / "B1" / "cs2048",
    "M1":               RESULTS / "M1",
    "M2/cs1024":        RESULTS / "M2" / "cs1024",
    "M2/cs2048":        RESULTS / "M2" / "cs2048",
    "M4/cs1024":        RESULTS / "M4" / "cs1024",
    "M4/cs2048":        RESULTS / "M4" / "cs2048",
    "A4/cs1024":        RESULTS / "A4" / "cs1024_no_namm",
    "A4/cs2048":        RESULTS / "A4" / "cs2048_no_namm",
    "Trunc/lora_1024":  RESULTS / "Trunc" / "lora_m1_1024",
    "Trunc/lora_2048":  RESULTS / "Trunc" / "lora_m1_2048",
    "Trunc/plain_1024": RESULTS / "Trunc" / "plain_1024",
    "Trunc/plain_2048": RESULTS / "Trunc" / "plain_2048",
}

DIFFICULTY_CSV = (
    ROOT / "analysis_giacomo" / "report_benchmark_difficulty"
    / "benchmark_difficulty_{split}.csv"
)

SUCCESS_THRESH = 0.3


# ── data loading ──────────────────────────────────────────────────

def load_per_prompt_f1_all(split: str = "extended_test") -> dict[str, dict[tuple, float]]:
    """Load per-prompt F1 from results.json for all conditions.
    Returns {cond: {(task_full, prompt_idx_str): f1_proportion}}.
    """
    data = {}
    for cond, base_path in ALL_CONDS.items():
        rpath = base_path / "results.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            raw = json.load(f)
        if "scores_per_split" in raw:
            split_data = raw["scores_per_split"][split]
        elif "results" in raw:
            split_data = raw["results"][split]
        else:
            continue
        ppf = split_data["per_prompt_f1"]
        cond_data = {}
        for task in TASKS:
            if task in ppf:
                for idx, f1 in ppf[task].items():
                    cond_data[(task, idx)] = f1
        data[cond] = cond_data
    return data


def load_generations(split: str = "extended_test") -> dict[str, dict[tuple, dict]]:
    """Load generations for document lengths and predictions.
    Returns {cond: {(task_full, prompt_idx_str): {pred, answers, length}}}.
    """
    data = {}
    for cond, base_path in ALL_CONDS.items():
        gpath = base_path / "generations.json"
        if not gpath.exists():
            continue
        with open(gpath) as f:
            raw = json.load(f)
        if split not in raw:
            continue
        cond_data = {}
        for task in TASKS:
            if task not in raw[split]:
                continue
            for item in raw[split][task]:
                key = (task, str(item["prompt_idx"]))
                cond_data[key] = {
                    "pred": item["pred"],
                    "answers": item["answers"],
                    "length": item.get("length", 0),
                }
        data[cond] = cond_data
    return data


def load_difficulty(split: str) -> pd.DataFrame:
    """Load pre-computed difficulty classification."""
    path = str(DIFFICULTY_CSV).format(split=split)
    return pd.read_csv(path)


# ── analysis ──────────────────────────────────────────────────────

def analyse_stratum(split: str = "extended_test"):
    print(f"\n{'='*70}")
    print(f"P2: Genuinely-hard sample characterisation ({split})")
    print(f"{'='*70}")

    diff_df = load_difficulty(split)
    f1_data = load_per_prompt_f1_all(split)
    gen_data = load_generations(split)

    # Get genuinely-hard sample keys
    gh = diff_df[diff_df["difficulty_simple"] == "genuinely-hard"]
    print(f"\n  Total samples: {len(diff_df)}")
    print(f"  Genuinely-hard: {len(gh)} ({len(gh)/len(diff_df)*100:.1f}%)")

    # ── 1. Task distribution ──
    print(f"\n  Task distribution of genuinely-hard samples:")
    task_counts = gh["task"].value_counts()
    for task in ["2WikiMQA", "2WikiMQA-E", "HotpotQA-E", "Qasper", "Qasper-E"]:
        n = task_counts.get(task, 0)
        total_task = len(diff_df[diff_df["task"] == task])
        pct = n / total_task * 100 if total_task > 0 else 0
        print(f"    {task:<15}: {n:3d} / {total_task:3d} ({pct:.0f}% of task is genuinely-hard)")

    # ── 2. Document length analysis ──
    # Get lengths from generations (use B0 or M1 as reference for document length)
    lengths = {}
    ref_cond = "B0" if "B0" in gen_data else "M1"
    for _, row in gh.iterrows():
        task_full = SHORT_TO_FULL.get(row["task"], "")
        key = (task_full, str(row["prompt_idx"]))
        info = gen_data.get(ref_cond, {}).get(key, {})
        lengths[key] = info.get("length", 0)

    gh_lengths = np.array([lengths.get((SHORT_TO_FULL.get(r["task"], ""), str(r["prompt_idx"])), 0)
                           for _, r in gh.iterrows()])
    gh_lengths = gh_lengths[gh_lengths > 0]  # filter missing

    # Compare with other strata
    strata_lengths = {}
    for stratum in ["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]:
        sub = diff_df[diff_df["difficulty_simple"] == stratum]
        lens = []
        for _, r in sub.iterrows():
            task_full = SHORT_TO_FULL.get(r["task"], "")
            key = (task_full, str(r["prompt_idx"]))
            info = gen_data.get(ref_cond, {}).get(key, {})
            l = info.get("length", 0)
            if l > 0:
                lens.append(l)
        strata_lengths[stratum] = np.array(lens) if lens else np.array([0])

    print(f"\n  Document length by difficulty stratum:")
    print(f"  {'Stratum':<25} {'n':>4} {'mean':>7} {'median':>7} {'min':>6} {'max':>6}")
    for stratum in ["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]:
        lens = strata_lengths[stratum]
        if len(lens) > 0 and lens[0] > 0:
            print(f"  {stratum:<25} {len(lens):4d} {lens.mean():7.0f} {np.median(lens):7.0f} "
                  f"{lens.min():6.0f} {lens.max():6.0f}")

    # ── 3. Per-condition F1 on genuinely-hard ──
    print(f"\n  Mean F1 (%) by condition on genuinely-hard samples (n={len(gh)}):")
    cond_f1s = {}
    for cond in ALL_CONDS:
        if cond not in f1_data:
            continue
        f1s = []
        for _, r in gh.iterrows():
            task_full = SHORT_TO_FULL.get(r["task"], "")
            key = (task_full, str(r["prompt_idx"]))
            f1 = f1_data[cond].get(key, -1)
            if f1 >= 0:
                f1s.append(f1 * 100)
        if f1s:
            cond_f1s[cond] = np.array(f1s)

    sorted_conds = sorted(cond_f1s.items(), key=lambda x: x[1].mean(), reverse=True)
    for cond, f1s in sorted_conds:
        print(f"    {cond:<25}: {f1s.mean():6.2f} (median={np.median(f1s):.1f}, "
              f"n_success={int((f1s >= 30).sum())}/{len(f1s)})")

    # ── 4. Unique wins analysis (M4/cs1024 vs Trunc/lora_1024) ──
    print(f"\n  Head-to-head: M4/cs1024 vs Trunc/lora_1024 on genuinely-hard:")
    m4_wins, tl_wins, ties = [], [], []
    rows_detail = []
    for _, r in gh.iterrows():
        task_full = SHORT_TO_FULL.get(r["task"], "")
        key = (task_full, str(r["prompt_idx"]))
        f1_m4 = f1_data.get("M4/cs1024", {}).get(key, -1) * 100
        f1_tl = f1_data.get("Trunc/lora_1024", {}).get(key, -1) * 100
        if f1_m4 < 0 or f1_tl < 0:
            continue
        delta = f1_m4 - f1_tl
        if delta > 0.5:
            m4_wins.append((key, f1_m4, f1_tl, delta))
        elif delta < -0.5:
            tl_wins.append((key, f1_m4, f1_tl, delta))
        else:
            ties.append((key, f1_m4, f1_tl, delta))
        rows_detail.append({
            "task": r["task"],
            "prompt_idx": r["prompt_idx"],
            "length": lengths.get(key, 0),
            "f1_m4_cs1024": round(f1_m4, 1),
            "f1_trunc_lora_1024": round(f1_tl, 1),
            "delta": round(delta, 1),
            "winner": "M4" if delta > 0.5 else "Trunc" if delta < -0.5 else "tie",
        })

    print(f"    M4/cs1024 wins: {len(m4_wins)}")
    print(f"    Trunc/lora wins: {len(tl_wins)}")
    print(f"    Ties (|delta| < 0.5): {len(ties)}")

    # Show biggest M4 wins
    if m4_wins:
        m4_wins.sort(key=lambda x: x[3], reverse=True)
        print(f"\n    Top 5 M4/cs1024 wins on genuinely-hard:")
        for key, f1_m4, f1_tl, delta in m4_wins[:5]:
            task_short = TASK_SHORT.get(key[0], key[0])
            print(f"      {task_short}[{key[1]}]: M4={f1_m4:.1f}, Trunc={f1_tl:.1f}, delta={delta:+.1f}")

    if tl_wins:
        tl_wins.sort(key=lambda x: x[3])
        print(f"\n    Top 5 Trunc/lora_1024 wins on genuinely-hard:")
        for key, f1_m4, f1_tl, delta in tl_wins[:5]:
            task_short = TASK_SHORT.get(key[0], key[0])
            print(f"      {task_short}[{key[1]}]: M4={f1_m4:.1f}, Trunc={f1_tl:.1f}, delta={delta:+.1f}")

    # Per-task breakdown of head-to-head
    detail_df = pd.DataFrame(rows_detail)
    if not detail_df.empty:
        print(f"\n  Per-task head-to-head on genuinely-hard (M4/cs1024 vs Trunc/lora_1024):")
        print(f"  {'Task':<15} {'n':>4} {'M4 wins':>8} {'Trunc':>6} {'Ties':>5} {'Mean Δ':>8}")
        for task in sorted(detail_df["task"].unique()):
            sub = detail_df[detail_df["task"] == task]
            print(f"  {task:<15} {len(sub):4d} {(sub['winner']=='M4').sum():8d} "
                  f"{(sub['winner']=='Trunc').sum():6d} {(sub['winner']=='tie').sum():5d} "
                  f"{sub['delta'].mean():+8.1f}")

    # Save detail CSV
    detail_df.to_csv(OUT / f"genuinely_hard_detail_{split}.csv", index=False)

    # ── 5. Unique-to-M4 samples (M4 succeeds, Trunc/lora fails) ──
    unique_m4 = []
    unique_tl = []
    for _, r in gh.iterrows():
        task_full = SHORT_TO_FULL.get(r["task"], "")
        key = (task_full, str(r["prompt_idx"]))
        f1_m4 = f1_data.get("M4/cs1024", {}).get(key, -1)
        f1_tl = f1_data.get("Trunc/lora_1024", {}).get(key, -1)
        if f1_m4 >= 0 and f1_tl >= 0:
            if f1_m4 >= 0.3 and f1_tl < 0.3:
                unique_m4.append((key, f1_m4 * 100, f1_tl * 100))
            elif f1_tl >= 0.3 and f1_m4 < 0.3:
                unique_tl.append((key, f1_m4 * 100, f1_tl * 100))

    print(f"\n  Unique successes (F1 >= 30%):")
    print(f"    Only M4/cs1024 succeeds: {len(unique_m4)}")
    print(f"    Only Trunc/lora_1024 succeeds: {len(unique_tl)}")

    if unique_m4:
        print(f"\n    Samples uniquely solved by M4/cs1024:")
        for key, f1_m4, f1_tl in sorted(unique_m4, key=lambda x: x[1], reverse=True)[:10]:
            task_short = TASK_SHORT.get(key[0], key[0])
            l = lengths.get(key, 0)
            print(f"      {task_short}[{key[1]}] len={l}: M4={f1_m4:.1f}, Trunc={f1_tl:.1f}")

    # ── Plots ──

    # Plot 1: Per-condition F1 on genuinely-hard
    fig, ax = plt.subplots(figsize=(14, 6))
    conds_sorted = [c for c, _ in sorted_conds]
    means = [cond_f1s[c].mean() for c in conds_sorted]
    colors = []
    for c in conds_sorted:
        if c.startswith("M4"):
            colors.append("#e74c3c")
        elif c == "M1":
            colors.append("#2ecc71")
        elif "Trunc/lora" in c:
            colors.append("#3498db")
        elif "Trunc/plain" in c:
            colors.append("#85c1e9")
        elif c.startswith("M2"):
            colors.append("#f39c12")
        elif c.startswith("B"):
            colors.append("#95a5a6")
        else:
            colors.append("#8e44ad")
    ax.barh(range(len(conds_sorted)), means, color=colors, edgecolor="white")
    for i, (c, m) in enumerate(zip(conds_sorted, means)):
        ax.text(m + 0.3, i, f"{m:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(conds_sorted)))
    ax.set_yticklabels(conds_sorted, fontsize=8)
    ax.set_xlabel("Mean F1 (%)")
    ax.set_title(f"Genuinely-hard samples only (n={len(gh)}, {split})\n"
                 "These are the samples where NAMM's contribution matters")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / f"condition_f1_genuinely_hard_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved condition_f1_genuinely_hard_{split}.png")

    # Plot 2: Document length distribution by stratum
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_s = {"tail-solvable": "#2ecc71", "needs-more-context": "#f1c40f",
                "genuinely-hard": "#e74c3c", "universally-hard": "#95a5a6"}
    positions = []
    labels_s = []
    pos = 0
    for stratum in ["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]:
        lens = strata_lengths[stratum]
        if len(lens) > 0 and lens[0] > 0:
            bp = ax.boxplot(lens, positions=[pos], widths=0.6, patch_artist=True,
                            boxprops={"facecolor": colors_s[stratum], "alpha": 0.7})
            positions.append(pos)
            labels_s.append(f"{stratum}\n(n={len(lens)})")
            pos += 1
    ax.set_xticks(range(pos))
    ax.set_xticklabels(labels_s, fontsize=8)
    ax.set_ylabel("Document length (tokens)")
    ax.set_title(f"Document length by difficulty stratum ({split})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"length_by_stratum_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved length_by_stratum_{split}.png")

    # Plot 3: Task breakdown of genuinely-hard
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3a: Task counts
    ax = axes[0]
    task_names = task_counts.index.tolist()
    task_vals = task_counts.values
    ax.bar(range(len(task_names)), task_vals, color="#e74c3c", edgecolor="white")
    for i, v in enumerate(task_vals):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, fontsize=8, rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Genuinely-hard: task distribution")
    ax.grid(axis="y", alpha=0.3)

    # 3b: Head-to-head per task
    if not detail_df.empty:
        ax = axes[1]
        tasks_h2h = sorted(detail_df["task"].unique())
        x = np.arange(len(tasks_h2h))
        w = 0.3
        m4_w = [(detail_df[(detail_df["task"]==t) & (detail_df["winner"]=="M4")].shape[0]) for t in tasks_h2h]
        tl_w = [(detail_df[(detail_df["task"]==t) & (detail_df["winner"]=="Trunc")].shape[0]) for t in tasks_h2h]
        ax.bar(x - w/2, m4_w, w, label="M4/cs1024 wins", color="#e74c3c", edgecolor="white")
        ax.bar(x + w/2, tl_w, w, label="Trunc/lora wins", color="#3498db", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_h2h, fontsize=8, rotation=15)
        ax.set_ylabel("Win count")
        ax.set_title("Head-to-head on genuinely-hard")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / f"genuinely_hard_breakdown_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved genuinely_hard_breakdown_{split}.png")

    return gh, detail_df


def main():
    for split in ["extended_test", "test"]:
        analyse_stratum(split)

    print("\n" + "=" * 70)
    print("P2 complete.")
    print(f"All outputs: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
