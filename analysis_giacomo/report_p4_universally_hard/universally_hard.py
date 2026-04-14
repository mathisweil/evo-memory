"""
P4: Investigate the universally-hard floor.

Questions:
  1. What fraction of samples are unsolvable by any method?
  2. Is F1=0 a metric artefact (model produces valid text but normalisation kills match)?
  3. What is the effective benchmark size (samples where at least one condition scores)?
  4. Do aggregate numbers change when we exclude universally-hard samples?

Note: "M4" in file paths = "M3" in the paper (LoRA + frozen NAMM).

Output: analysis_giacomo/report_p4_universally_hard/
"""

import json
import logging
import re
import string
from collections import Counter
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

SUCCESS_THRESH = 0.3  # F1 proportion


# ── F1 computation (matches eval code) ────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not gt and not pt:
        return 1.0
    if not gt or not pt:
        return 0.0
    c = sum((Counter(pt) & Counter(gt)).values())
    if c == 0:
        return 0.0
    p, r = c / len(pt), c / len(gt)
    return 2 * p * r / (p + r)


def best_f1(pred: str, answers: list[str]) -> float:
    return max(token_f1(pred, a) for a in answers) if answers else 0.0


# ── data loading ──────────────────────────────────────────────────

def load_per_prompt_f1_all(split: str) -> dict[str, dict[tuple, float]]:
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


def load_generations_all(split: str) -> dict[str, dict[tuple, dict]]:
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
    path = str(DIFFICULTY_CSV).format(split=split)
    return pd.read_csv(path)


# ── analysis ──────────────────────────────────────────────────────

def investigate(split: str = "extended_test"):
    print(f"\n{'='*70}")
    print(f"P4: Universally-hard floor investigation ({split})")
    print(f"{'='*70}")

    diff_df = load_difficulty(split)
    f1_data = load_per_prompt_f1_all(split)
    gen_data = load_generations_all(split)

    uh = diff_df[diff_df["difficulty_simple"] == "universally-hard"]
    print(f"\n  Total samples: {len(diff_df)}")
    print(f"  Universally-hard: {len(uh)} ({len(uh)/len(diff_df)*100:.1f}%)")

    # ── 1. Per-task breakdown ──
    print(f"\n  Universally-hard by task:")
    print(f"  {'Task':<15} {'UH':>4} {'Total':>6} {'%':>6}")
    for task in sorted(diff_df["task"].unique()):
        total = len(diff_df[diff_df["task"] == task])
        n_uh = len(uh[uh["task"] == task])
        print(f"  {task:<15} {n_uh:4d} {total:6d} {n_uh/total*100:6.1f}")

    # ── 2. Metric artefact check ──
    # For universally-hard samples, look at predictions across conditions.
    # Check if models produce text that looks like a valid answer but F1=0.
    print(f"\n  Metric artefact check on universally-hard samples:")
    print(f"  Checking if models produce plausible text that scores F1=0...\n")

    artefact_candidates = []
    ref_conds = ["M1", "M4/cs1024", "B0"]  # strongest conditions

    for _, row in uh.iterrows():
        task_full = SHORT_TO_FULL.get(row["task"], "")
        key = (task_full, str(row["prompt_idx"]))

        for cond in ref_conds:
            info = gen_data.get(cond, {}).get(key, {})
            if not info:
                continue
            pred = info["pred"]
            answers = info["answers"]
            f1 = best_f1(pred, answers)

            # Check: non-empty prediction, not a refusal, but F1 near 0
            pred_norm = normalize_answer(pred)
            is_refusal = "unanswerable" in pred_norm or len(pred_norm) < 3
            is_nonempty = len(pred_norm.split()) >= 2

            if is_nonempty and not is_refusal and f1 < 0.05:
                # Check if answer tokens appear in prediction at all
                gold_tokens = set()
                for a in answers:
                    gold_tokens.update(normalize_answer(a).split())
                pred_tokens = set(pred_norm.split())
                overlap = gold_tokens & pred_tokens

                artefact_candidates.append({
                    "task": row["task"],
                    "prompt_idx": row["prompt_idx"],
                    "condition": cond,
                    "pred_len_words": len(pred_norm.split()),
                    "gold_answers": "; ".join(answers[:3]),
                    "pred_preview": pred[:120].replace("\n", " "),
                    "f1": round(f1, 3),
                    "token_overlap": len(overlap),
                    "gold_tokens": len(gold_tokens),
                })

    art_df = pd.DataFrame(artefact_candidates)
    if not art_df.empty:
        art_df.to_csv(OUT / f"artefact_candidates_{split}.csv", index=False)

        # Classify artefacts
        n_total_uh = len(uh)
        # Unique samples where at least one condition produces non-refusal text with F1~0
        unique_art = art_df.groupby(["task", "prompt_idx"]).first().reset_index()
        n_artefact = len(unique_art)
        n_zero_overlap = len(unique_art[unique_art["token_overlap"] == 0])
        n_some_overlap = len(unique_art[unique_art["token_overlap"] > 0])

        print(f"  Of {n_total_uh} universally-hard samples:")
        print(f"    {n_artefact} have non-refusal predictions from strong conditions")
        print(f"    {n_zero_overlap} have zero token overlap with gold (genuinely wrong)")
        print(f"    {n_some_overlap} have SOME token overlap but F1<5% (possible artefact)")

        if n_some_overlap > 0:
            print(f"\n  Possible metric artefacts (some overlap, F1~0):")
            overlap_cases = unique_art[unique_art["token_overlap"] > 0].head(10)
            for _, r in overlap_cases.iterrows():
                print(f"    {r['task']}[{r['prompt_idx']}] ({r['condition']}): "
                      f"F1={r['f1']}, overlap={r['token_overlap']}/{r['gold_tokens']} tokens")
                print(f"      Gold: {r['gold_answers'][:80]}")
                print(f"      Pred: {r['pred_preview'][:80]}")
    else:
        print("  No artefact candidates found — all UH predictions are refusals or very short")

    # ── 3. Prediction type distribution for UH samples ──
    print(f"\n  Prediction types on universally-hard samples (from M1):")
    pred_types = Counter()
    for _, row in uh.iterrows():
        task_full = SHORT_TO_FULL.get(row["task"], "")
        key = (task_full, str(row["prompt_idx"]))
        info = gen_data.get("M1", {}).get(key, {})
        if not info:
            pred_types["missing"] += 1
            continue
        pred = info["pred"]
        pred_norm = normalize_answer(pred)
        if "unanswerable" in pred_norm:
            pred_types["refusal"] += 1
        elif len(pred_norm.split()) < 2:
            pred_types["very_short"] += 1
        elif len(pred_norm.split()) > 50:
            pred_types["verbose"] += 1
        else:
            pred_types["normal_text"] += 1

    for ptype, count in pred_types.most_common():
        print(f"    {ptype:<15}: {count:3d} ({count/len(uh)*100:.0f}%)")

    # ── 4. Effective benchmark size and re-computed aggregates ──
    print(f"\n  Effective benchmark (excluding universally-hard):")
    solvable = diff_df[diff_df["difficulty_simple"] != "universally-hard"]
    n_solvable = len(solvable)
    print(f"    Solvable samples: {n_solvable} / {len(diff_df)} "
          f"({n_solvable/len(diff_df)*100:.0f}%)")

    # Build solvable key set
    solvable_keys = set()
    for _, r in solvable.iterrows():
        task_full = SHORT_TO_FULL.get(r["task"], "")
        solvable_keys.add((task_full, str(r["prompt_idx"])))

    # Re-compute condition aggregates on solvable subset
    print(f"\n  Re-computed F1 on solvable subset (n={n_solvable}):")
    print(f"  {'Condition':<25} {'Micro (all)':>12} {'Micro (solv)':>13} {'Δ':>6} "
          f"{'Macro (all)':>12} {'Macro (solv)':>13} {'Δ':>6}")
    print(f"  {'─'*90}")

    recomputed_rows = []
    for cond in ALL_CONDS:
        if cond not in f1_data:
            continue
        cond_f1 = f1_data[cond]

        # All samples
        all_f1s = [f1 * 100 for f1 in cond_f1.values()]
        micro_all = np.mean(all_f1s) if all_f1s else 0

        # Per-task for macro
        task_means_all = []
        for task in TASKS:
            task_f1s = [f1 * 100 for (t, _), f1 in cond_f1.items() if t == task]
            if task_f1s:
                task_means_all.append(np.mean(task_f1s))
        macro_all = np.mean(task_means_all) if task_means_all else 0

        # Solvable only
        solv_f1s = [f1 * 100 for key, f1 in cond_f1.items() if key in solvable_keys]
        micro_solv = np.mean(solv_f1s) if solv_f1s else 0

        task_means_solv = []
        for task in TASKS:
            task_f1s = [f1 * 100 for (t, idx), f1 in cond_f1.items()
                        if t == task and (t, idx) in solvable_keys]
            if task_f1s:
                task_means_solv.append(np.mean(task_f1s))
        macro_solv = np.mean(task_means_solv) if task_means_solv else 0

        print(f"  {cond:<25} {micro_all:12.2f} {micro_solv:13.2f} {micro_solv-micro_all:+6.2f} "
              f"{macro_all:12.2f} {macro_solv:13.2f} {macro_solv-macro_all:+6.2f}")

        recomputed_rows.append({
            "condition": cond,
            "micro_all": round(micro_all, 2),
            "micro_solvable": round(micro_solv, 2),
            "micro_delta": round(micro_solv - micro_all, 2),
            "macro_all": round(macro_all, 2),
            "macro_solvable": round(macro_solv, 2),
            "macro_delta": round(macro_solv - macro_all, 2),
            "n_all": len(all_f1s),
            "n_solvable": len(solv_f1s),
        })

    recomp_df = pd.DataFrame(recomputed_rows)
    recomp_df.to_csv(OUT / f"recomputed_aggregates_{split}.csv", index=False)

    # ── 5. Key comparisons on solvable subset ──
    print(f"\n  Key comparisons on solvable subset:")
    key_pairs = [
        ("M4/cs1024", "Trunc/lora_1024"),
        ("M4/cs2048", "Trunc/lora_2048"),
        ("M4/cs1024", "M4/cs2048"),
        ("M1", "M4/cs1024"),
    ]
    for a, b in key_pairs:
        if a not in f1_data or b not in f1_data:
            continue
        shared = sorted(set(f1_data[a]) & set(f1_data[b]) & solvable_keys)
        deltas = np.array([(f1_data[a][k] - f1_data[b][k]) * 100 for k in shared])
        print(f"    {a} vs {b}: n={len(shared)}, "
              f"micro Δ={deltas.mean():+.2f}, "
              f"A wins {(deltas > 0.5).sum()}, B wins {(deltas < -0.5).sum()}")

    # ── Plots ──

    # Plot 1: Universally-hard by task
    fig, ax = plt.subplots(figsize=(12, 5))
    tasks_sorted = sorted(diff_df["task"].unique())
    x = np.arange(len(tasks_sorted))
    w = 0.35
    uh_counts = [len(uh[uh["task"] == t]) for t in tasks_sorted]
    total_counts = [len(diff_df[diff_df["task"] == t]) for t in tasks_sorted]
    solv_counts = [t - u for t, u in zip(total_counts, uh_counts)]

    ax.bar(x - w/2, solv_counts, w, label="Solvable (≥1 condition scores)", color="#2ecc71",
           edgecolor="white")
    ax.bar(x + w/2, uh_counts, w, label="Universally hard", color="#95a5a6", edgecolor="white")
    for i in range(len(tasks_sorted)):
        pct = uh_counts[i] / total_counts[i] * 100
        ax.text(i + w/2, uh_counts[i] + 0.5, f"{pct:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_sorted, fontsize=9, rotation=15)
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Universally-hard samples by task ({split})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"universally_hard_by_task_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved universally_hard_by_task_{split}.png")

    # Plot 2: Aggregates comparison (all vs solvable)
    if not recomp_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Sort by micro_solvable for both
        recomp_sorted = recomp_df.sort_values("micro_solvable", ascending=True)

        for ax_idx, metric in enumerate(["micro", "macro"]):
            ax = axes[ax_idx]
            conds = recomp_sorted["condition"].tolist()
            y = np.arange(len(conds))
            all_vals = recomp_sorted[f"{metric}_all"].values
            solv_vals = recomp_sorted[f"{metric}_solvable"].values

            ax.barh(y - 0.15, all_vals, 0.3, label="All samples", color="#95a5a6",
                    edgecolor="white", alpha=0.7)
            ax.barh(y + 0.15, solv_vals, 0.3, label="Solvable only", color="#2ecc71",
                    edgecolor="white")
            ax.set_yticks(y)
            ax.set_yticklabels(conds, fontsize=7)
            ax.set_xlabel(f"{metric.title()} F1 (%)")
            ax.set_title(f"{metric.title()} F1: all vs solvable ({split})")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT / f"all_vs_solvable_{split}.png", dpi=150)
        plt.close()
        print(f"  Saved all_vs_solvable_{split}.png")

    # Plot 3: Best F1 distribution for universally-hard (across all conditions)
    fig, ax = plt.subplots(figsize=(10, 5))
    best_f1s = []
    for _, row in uh.iterrows():
        task_full = SHORT_TO_FULL.get(row["task"], "")
        key = (task_full, str(row["prompt_idx"]))
        max_f1 = max(
            (f1_data.get(c, {}).get(key, 0) for c in ALL_CONDS),
            default=0,
        )
        best_f1s.append(max_f1 * 100)

    best_f1s = np.array(best_f1s)
    ax.hist(best_f1s, bins=30, color="#e74c3c", edgecolor="white", alpha=0.7)
    ax.axvline(30, color="black", linewidth=1, linestyle="--", label=f"Threshold ({SUCCESS_THRESH*100}%)")
    ax.set_xlabel("Best F1 across all conditions (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Universally-hard: best achievable F1 (n={len(uh)}, {split})\n"
                 f"Zero F1: {(best_f1s == 0).sum()}, "
                 f"Nonzero but <30%: {((best_f1s > 0) & (best_f1s < 30)).sum()}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"universally_hard_best_f1_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved universally_hard_best_f1_{split}.png")

    return uh, recomp_df


def main():
    for split in ["extended_test", "test"]:
        investigate(split)

    print("\n" + "=" * 70)
    print("P4 complete.")
    print(f"All outputs: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
