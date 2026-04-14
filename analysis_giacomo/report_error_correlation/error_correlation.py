"""
Analysis 3: Cross-condition error correlation.

When M4 gets a sample wrong, does Trunc/lora also get it wrong?
Computes Jaccard similarity of failure/success sets across conditions.
High correlation = conditions fail for same reasons (same structural limitation).
Low correlation = genuinely different strategies.
"""

import json
import re
import string
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "main_table_5t"
OUT = Path(__file__).resolve().parent

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_SHORT = {
    "lb/qasper": "Qasper", "lb/2wikimqa": "2WikiMQA", "lb/qasper_e": "Qasper-E",
    "lb/hotpotqa_e": "HotpotQA-E", "lb/2wikimqa_e": "2WikiMQA-E",
}

COND_GENS = {
    "B0":               RESULTS / "B0" / "generations.json",
    "M1":               RESULTS / "M1" / "generations.json",
    "M4/cs1024":        RESULTS / "M4" / "cs1024" / "generations.json",
    "M4/cs2048":        RESULTS / "M4" / "cs2048" / "generations.json",
    "M2/cs1024":        RESULTS / "M2" / "cs1024" / "generations.json",
    "B1/cs1024":        RESULTS / "B1" / "cs1024" / "generations.json",
    "A4/cs1024":        RESULTS / "A4" / "cs1024_no_namm" / "generations.json",
    "A4/cs2048":        RESULTS / "A4" / "cs2048_no_namm" / "generations.json",
    "Trunc/lora_1024":  RESULTS / "Trunc" / "lora_m1_1024" / "generations.json",
    "Trunc/lora_2048":  RESULTS / "Trunc" / "lora_m1_2048" / "generations.json",
    "Trunc/plain_1024": RESULTS / "Trunc" / "plain_1024" / "generations.json",
    "Trunc/plain_2048": RESULTS / "Trunc" / "plain_2048" / "generations.json",
}

KEY_CONDS = ["B0", "M1", "M4/cs1024", "M4/cs2048", "A4/cs1024", "A4/cs2048",
             "Trunc/lora_1024", "Trunc/lora_2048"]


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def token_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not gt and not pt: return 1.0
    if not gt or not pt: return 0.0
    c = sum((Counter(pt) & Counter(gt)).values())
    if c == 0: return 0.0
    p, r = c/len(pt), c/len(gt)
    return 2*p*r/(p+r)

def best_f1(pred, answers):
    return max(token_f1(pred, a) for a in answers) if answers else 0.0


def load_per_sample_f1(split="test"):
    """Returns {cond: {(task, prompt_idx): f1}}"""
    data = {}
    for cond, path in COND_GENS.items():
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        if split not in raw:
            continue
        cond_data = {}
        for task in TASKS:
            if task not in raw[split]:
                continue
            for it in raw[split][task]:
                f1 = best_f1(it["pred"], it["answers"])
                cond_data[(task, it["prompt_idx"])] = f1
        data[cond] = cond_data
    return data


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def main():
    print("=" * 60)
    print("Analysis 3: Cross-condition error correlation")
    print("=" * 60)

    for split in ["test", "extended_test"]:
        print(f"\n--- {split} ---")
        data = load_per_sample_f1(split)
        conds = [c for c in KEY_CONDS if c in data]
        print(f"  Loaded {len(conds)} conditions")

        # Get shared samples across all conditions
        all_keys = set.intersection(*(set(data[c].keys()) for c in conds))
        print(f"  Shared samples: {len(all_keys)}")

        # Build failure sets (F1 < 0.1 = failure) and success sets (F1 >= 0.5)
        fail_thresh = 0.1
        success_thresh = 0.5

        failure_sets = {c: {k for k in all_keys if data[c][k] < fail_thresh} for c in conds}
        success_sets = {c: {k for k in all_keys if data[c][k] >= success_thresh} for c in conds}

        # ── Jaccard similarity matrices ──
        n = len(conds)
        fail_jaccard = np.zeros((n, n))
        succ_jaccard = np.zeros((n, n))

        for i, ca in enumerate(conds):
            for j, cb in enumerate(conds):
                fail_jaccard[i, j] = jaccard(failure_sets[ca], failure_sets[cb])
                succ_jaccard[i, j] = jaccard(success_sets[ca], success_sets[cb])

        # ── Print key comparisons ──
        print(f"\n  Failure set sizes (F1 < {fail_thresh}):")
        for c in conds:
            print(f"    {c:<20}: {len(failure_sets[c])}/{len(all_keys)} ({len(failure_sets[c])/len(all_keys)*100:.0f}%)")

        print(f"\n  Key failure Jaccard similarities:")
        pairs = [
            ("M4/cs1024", "Trunc/lora_1024"),
            ("M4/cs2048", "Trunc/lora_2048"),
            ("M4/cs1024", "M1"),
            ("M4/cs1024", "B0"),
            ("Trunc/lora_2048", "M1"),
            ("M4/cs1024", "A4/cs1024"),
        ]
        for ca, cb in pairs:
            if ca in conds and cb in conds:
                i, j = conds.index(ca), conds.index(cb)
                n_shared_fails = len(failure_sets[ca] & failure_sets[cb])
                n_only_a = len(failure_sets[ca] - failure_sets[cb])
                n_only_b = len(failure_sets[cb] - failure_sets[ca])
                print(f"    {ca} vs {cb}: J={fail_jaccard[i,j]:.3f} "
                      f"(shared={n_shared_fails}, only_{ca.split('/')[0]}={n_only_a}, only_{cb.split('/')[0]}={n_only_b})")

        # ── Plot failure Jaccard heatmap ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for ax, mat, title in [(ax1, fail_jaccard, f"Failure overlap (F1<{fail_thresh})"),
                                (ax2, succ_jaccard, f"Success overlap (F1>={success_thresh})")]:
            im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks(range(n))
            ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(conds, fontsize=8)
            ax.set_title(title)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if mat[i,j] > 0.6 else "black")
            fig.colorbar(im, ax=ax, shrink=0.7, label="Jaccard")

        fig.suptitle(f"Cross-Condition Error Correlation ({split})", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUT / f"error_correlation_{split}.png", dpi=150)
        plt.close()
        print(f"  Saved error_correlation_{split}.png")

        # ── Per-task breakdown for critical pair ──
        report = [f"# Cross-Condition Error Correlation ({split})\n\n"]

        for ca, cb in [("M4/cs2048", "Trunc/lora_2048"), ("M4/cs1024", "Trunc/lora_1024")]:
            if ca not in conds or cb not in conds:
                continue
            report.append(f"## {ca} vs {cb}\n\n")
            report.append(f"| Task | Both fail | Only {ca} | Only {cb} | Both succeed | Jaccard(fail) |\n")
            report.append(f"|---|---|---|---|---|---|\n")

            for task in TASKS:
                task_keys = {k for k in all_keys if k[0] == task}
                fails_a = {k for k in task_keys if data[ca][k] < fail_thresh}
                fails_b = {k for k in task_keys if data[cb][k] < fail_thresh}
                succs_a = {k for k in task_keys if data[ca][k] >= success_thresh}
                succs_b = {k for k in task_keys if data[cb][k] >= success_thresh}

                both_fail = len(fails_a & fails_b)
                only_a = len(fails_a - fails_b)
                only_b = len(fails_b - fails_a)
                both_succ = len(succs_a & succs_b)
                j = jaccard(fails_a, fails_b)

                report.append(f"| {TASK_SHORT[task]} | {both_fail} | {only_a} | {only_b} | {both_succ} | {j:.3f} |\n")

            # Overall
            both_fail = len(failure_sets[ca] & failure_sets[cb])
            only_a = len(failure_sets[ca] - failure_sets[cb])
            only_b = len(failure_sets[cb] - failure_sets[ca])
            both_succ = len(success_sets[ca] & success_sets[cb])
            j = jaccard(failure_sets[ca], failure_sets[cb])
            report.append(f"| **Overall** | **{both_fail}** | **{only_a}** | **{only_b}** | **{both_succ}** | **{j:.3f}** |\n\n")

            # Unique failures: samples where one condition fails and the other succeeds
            unique_a_fails = {k for k in all_keys if data[ca][k] < fail_thresh and data[cb][k] >= success_thresh}
            unique_b_fails = {k for k in all_keys if data[cb][k] < fail_thresh and data[ca][k] >= success_thresh}
            report.append(f"**{ca} fails but {cb} succeeds**: {len(unique_a_fails)} samples\n")
            report.append(f"**{cb} fails but {ca} succeeds**: {len(unique_b_fails)} samples\n\n")

            # Show the actual samples
            if unique_a_fails:
                report.append(f"### Samples where {cb} succeeds but {ca} fails:\n\n")
                for task, idx in sorted(unique_a_fails)[:5]:
                    report.append(f"- {TASK_SHORT[task]}[{idx}]: {ca} F1={data[ca][(task,idx)]:.3f}, {cb} F1={data[cb][(task,idx)]:.3f}\n")
            if unique_b_fails:
                report.append(f"\n### Samples where {ca} succeeds but {cb} fails:\n\n")
                for task, idx in sorted(unique_b_fails)[:5]:
                    report.append(f"- {TASK_SHORT[task]}[{idx}]: {ca} F1={data[ca][(task,idx)]:.3f}, {cb} F1={data[cb][(task,idx)]:.3f}\n")
            report.append("\n")

        with open(OUT / f"error_correlation_{split}.md", "w") as f:
            f.writelines(report)
        print(f"  Saved error_correlation_{split}.md")

        # ── Save CSVs ──
        pd.DataFrame(fail_jaccard, index=conds, columns=conds).to_csv(
            OUT / f"fail_jaccard_{split}.csv")
        pd.DataFrame(succ_jaccard, index=conds, columns=conds).to_csv(
            OUT / f"succ_jaccard_{split}.csv")

    print("\n" + "=" * 60)
    print("Analysis 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
