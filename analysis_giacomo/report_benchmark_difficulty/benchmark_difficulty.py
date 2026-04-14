"""
Investigation B: Is LongBench actually testing long-context ability?
Investigation E: Unanswerable analysis for Qasper.
Investigation F: Prompt template token budget.

B: For each sample, classify as:
  - "tail-solvable": Trunc/plain_1024 gets F1 > 0.3
  - "needs-more-context": Trunc/plain_1024 fails but Trunc/plain_2048 succeeds
  - "genuinely-hard": both truncation budgets fail, but M1 or B0 succeeds
  - "universally-hard": all conditions fail

E: Count "unanswerable" responses across conditions for Qasper tasks.

F: Measure actual token budget consumed by chat template + question.
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

os.environ["HF_HOME"] = "/cs/student/project_msc/2025/csml/gmaralla/.hf_cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "main_table_5t"
OUT = Path(__file__).resolve().parent

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_SHORT = {
    "lb/qasper": "Qasper", "lb/2wikimqa": "2WikiMQA", "lb/qasper_e": "Qasper-E",
    "lb/hotpotqa_e": "HotpotQA-E", "lb/2wikimqa_e": "2WikiMQA-E",
}
TASK_LB = {
    "lb/qasper": "qasper", "lb/2wikimqa": "2wikimqa", "lb/qasper_e": "qasper_e",
    "lb/hotpotqa_e": "hotpotqa_e", "lb/2wikimqa_e": "2wikimqa_e",
}

COND_GENS = {
    "B0":               RESULTS / "B0" / "generations.json",
    "M1":               RESULTS / "M1" / "generations.json",
    "M4/cs1024":        RESULTS / "M4" / "cs1024" / "generations.json",
    "M4/cs2048":        RESULTS / "M4" / "cs2048" / "generations.json",
    "M2/cs1024":        RESULTS / "M2" / "cs1024" / "generations.json",
    "Trunc/lora_1024":  RESULTS / "Trunc" / "lora_m1_1024" / "generations.json",
    "Trunc/lora_2048":  RESULTS / "Trunc" / "lora_m1_2048" / "generations.json",
    "Trunc/plain_1024": RESULTS / "Trunc" / "plain_1024" / "generations.json",
    "Trunc/plain_2048": RESULTS / "Trunc" / "plain_2048" / "generations.json",
    "A4/cs1024":        RESULTS / "A4" / "cs1024_no_namm" / "generations.json",
    "B1/cs1024":        RESULTS / "B1" / "cs1024" / "generations.json",
}

SUCCESS_THRESH = 0.3


def normalize_answer(s):
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


def load_all_f1(split="test"):
    """Returns {cond: {(task, idx): f1}}"""
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
                cond_data[(task, it["prompt_idx"])] = {
                    "f1": f1, "pred": it["pred"], "answers": it["answers"],
                    "length": it.get("length", 0),
                }
        data[cond] = cond_data
    return data


# ═══════════════════════════════════════════════════════════════
# Investigation B: Benchmark difficulty stratification
# ═══════════════════════════════════════════════════════════════

def investigation_b(split="test"):
    print(f"\n{'='*60}")
    print(f"Investigation B: Benchmark difficulty ({split})")
    print(f"{'='*60}")

    data = load_all_f1(split)
    all_keys = set()
    for c in data.values():
        all_keys.update(c.keys())

    rows = []
    for key in sorted(all_keys):
        task, idx = key
        tp1 = data.get("Trunc/plain_1024", {}).get(key, {}).get("f1", -1)
        tp2 = data.get("Trunc/plain_2048", {}).get(key, {}).get("f1", -1)
        tl1 = data.get("Trunc/lora_1024", {}).get(key, {}).get("f1", -1)
        tl2 = data.get("Trunc/lora_2048", {}).get(key, {}).get("f1", -1)
        m1  = data.get("M1", {}).get(key, {}).get("f1", -1)
        m4_1 = data.get("M4/cs1024", {}).get(key, {}).get("f1", -1)
        b0  = data.get("B0", {}).get(key, {}).get("f1", -1)
        length = data.get("B0", {}).get(key, {}).get("length", 0)
        if length == 0:
            length = data.get("M1", {}).get(key, {}).get("length", 0)

        # Classify difficulty
        any_succeed = any(v >= SUCCESS_THRESH for v in [tp1, tp2, tl1, tl2, m1, m4_1, b0] if v >= 0)

        if tp1 >= SUCCESS_THRESH:
            difficulty = "tail-solvable (1K)"
        elif tp2 >= SUCCESS_THRESH:
            difficulty = "needs 2K context"
        elif tl1 >= SUCCESS_THRESH:
            difficulty = "needs LoRA (1K tail)"
        elif tl2 >= SUCCESS_THRESH:
            difficulty = "needs LoRA (2K tail)"
        elif m1 >= SUCCESS_THRESH or m4_1 >= SUCCESS_THRESH:
            difficulty = "needs NAMM/full-context"
        elif b0 >= SUCCESS_THRESH:
            difficulty = "needs full-context (B0)"
        elif any_succeed:
            difficulty = "condition-specific"
        else:
            difficulty = "universally hard"

        # Simpler 4-way classification
        if tp1 >= SUCCESS_THRESH:
            difficulty_simple = "tail-solvable"
        elif tp2 >= SUCCESS_THRESH or tl1 >= SUCCESS_THRESH:
            difficulty_simple = "needs-more-context"
        elif any_succeed:
            difficulty_simple = "genuinely-hard"
        else:
            difficulty_simple = "universally-hard"

        rows.append({
            "task": TASK_SHORT[task],
            "prompt_idx": idx,
            "length": length,
            "f1_tp1": tp1, "f1_tp2": tp2, "f1_tl1": tl1, "f1_tl2": tl2,
            "f1_m1": m1, "f1_m4_1": m4_1, "f1_b0": b0,
            "difficulty": difficulty,
            "difficulty_simple": difficulty_simple,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"benchmark_difficulty_{split}.csv", index=False)

    # Summary
    print(f"\n  Detailed difficulty breakdown ({split}, n={len(df)}):")
    for d in ["tail-solvable (1K)", "needs 2K context", "needs LoRA (1K tail)",
              "needs LoRA (2K tail)", "needs NAMM/full-context", "needs full-context (B0)",
              "condition-specific", "universally hard"]:
        n = (df["difficulty"] == d).sum()
        if n > 0:
            print(f"    {d:<30}: {n:3d} ({n/len(df)*100:.1f}%)")

    print(f"\n  Simple difficulty breakdown:")
    for d in ["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]:
        n = (df["difficulty_simple"] == d).sum()
        print(f"    {d:<25}: {n:3d} ({n/len(df)*100:.1f}%)")

    # Per-task breakdown
    print(f"\n  Per-task difficulty (simple):")
    pivot = df.groupby(["task", "difficulty_simple"]).size().unstack(fill_value=0)
    for d in ["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]:
        if d not in pivot.columns:
            pivot[d] = 0
    pivot = pivot[["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]]
    pivot["total"] = pivot.sum(axis=1)
    pivot["tail_pct"] = (pivot["tail-solvable"] / pivot["total"] * 100).round(1)
    print(pivot.to_string())

    # ── Plot: stacked bar by task ──
    fig, ax = plt.subplots(figsize=(12, 6))
    tasks = pivot.index.tolist()
    x = np.arange(len(tasks))
    colors = {"tail-solvable": "#2ecc71", "needs-more-context": "#f1c40f",
              "genuinely-hard": "#e74c3c", "universally-hard": "#95a5a6"}
    bottom = np.zeros(len(tasks))
    for d in ["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]:
        vals = pivot[d].values
        ax.bar(x, vals, bottom=bottom, label=d, color=colors[d], edgecolor="white")
        # Label counts
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottom[i] + v/2, str(v), ha="center", va="center", fontsize=8, fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Benchmark Difficulty: Is LongBench testing long-context? ({split})\n"
                 f"(tail-solvable = Trunc/plain_1024 gets F1>{SUCCESS_THRESH})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"benchmark_difficulty_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved benchmark_difficulty_{split}.png")

    # ── Plot: CDF of "context needed" ──
    fig, ax = plt.subplots(figsize=(10, 6))
    simple_order = ["tail-solvable", "needs-more-context", "genuinely-hard", "universally-hard"]
    cumulative = 0
    for d in simple_order:
        n = (df["difficulty_simple"] == d).sum()
        cumulative += n
        ax.barh(d, n, color=colors[d], edgecolor="white")
        ax.text(n + 1, d, f"{n} ({n/len(df)*100:.0f}%)", va="center", fontsize=10)

    ax.set_xlabel("Number of samples")
    ax.set_title(f"How many samples actually need long-context? ({split})")
    plt.tight_layout()
    plt.savefig(OUT / f"context_needed_cdf_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved context_needed_cdf_{split}.png")

    # ── Where does NAMM help, stratified by difficulty? ──
    print(f"\n  NAMM advantage (M4/cs1024 - Trunc/lora_1024) by difficulty:")
    for d in simple_order:
        sub = df[df["difficulty_simple"] == d]
        if sub.empty or sub["f1_m4_1"].min() < 0 or sub["f1_tl1"].min() < 0:
            continue
        valid = sub[(sub["f1_m4_1"] >= 0) & (sub["f1_tl1"] >= 0)]
        if valid.empty:
            continue
        delta = valid["f1_m4_1"] - valid["f1_tl1"]
        print(f"    {d:<25}: mean delta={delta.mean()*100:+.1f} F1, "
              f"M4 wins {(delta>0.01).sum()}/{len(valid)}, "
              f"Trunc wins {(delta<-0.01).sum()}/{len(valid)}")

    return df


# ═══════════════════════════════════════════════════════════════
# Investigation E: Unanswerable analysis
# ═══════════════════════════════════════════════════════════════

def investigation_e(split="test"):
    print(f"\n{'='*60}")
    print(f"Investigation E: Unanswerable analysis ({split})")
    print(f"{'='*60}")

    data = load_all_f1(split)
    qasper_tasks = [t for t in TASKS if "qasper" in t]

    rows = []
    for cond in COND_GENS:
        if cond not in data:
            continue
        for task in qasper_tasks:
            n_total = 0
            n_pred_unans = 0
            n_gold_unans = 0
            n_correct_unans = 0  # both say unanswerable
            n_false_refusal = 0  # pred=unans, gold has answer
            n_hallucination = 0  # gold=unans, pred fabricates

            for key, info in data[cond].items():
                if key[0] != task:
                    continue
                n_total += 1
                pred_norm = normalize_answer(info["pred"])
                gold_norms = [normalize_answer(a) for a in info["answers"]]
                pred_unans = "unanswerable" in pred_norm
                gold_unans = any("unanswerable" in g for g in gold_norms)

                if pred_unans:
                    n_pred_unans += 1
                if gold_unans:
                    n_gold_unans += 1
                if pred_unans and gold_unans:
                    n_correct_unans += 1
                if pred_unans and not gold_unans:
                    n_false_refusal += 1
                if not pred_unans and gold_unans:
                    n_hallucination += 1

            if n_total > 0:
                rows.append({
                    "condition": cond,
                    "task": TASK_SHORT[task],
                    "n_total": n_total,
                    "n_pred_unans": n_pred_unans,
                    "n_gold_unans": n_gold_unans,
                    "n_correct_unans": n_correct_unans,
                    "n_false_refusal": n_false_refusal,
                    "n_hallucination": n_hallucination,
                    "refusal_rate": n_pred_unans / n_total * 100,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"unanswerable_{split}.csv", index=False)

    print(f"\n  Qasper unanswerable analysis ({split}):")
    print(f"  {'Condition':<20} {'Task':<10} {'Total':>5} {'Pred UA':>7} {'Gold UA':>7} "
          f"{'Correct':>7} {'FalseRef':>8} {'Halluc':>6} {'Ref%':>5}")
    print("  " + "-" * 90)
    for _, r in df.iterrows():
        print(f"  {r['condition']:<20} {r['task']:<10} {r['n_total']:5d} {r['n_pred_unans']:7d} "
              f"{r['n_gold_unans']:7d} {r['n_correct_unans']:7d} {r['n_false_refusal']:8d} "
              f"{r['n_hallucination']:6d} {r['refusal_rate']:5.1f}")

    # Plot: refusal rate by condition for Qasper
    qasper_df = df[df["task"] == "Qasper"]
    if not qasper_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        qasper_df = qasper_df.sort_values("refusal_rate", ascending=True)
        colors = []
        for c in qasper_df["condition"]:
            if "Trunc" in c: colors.append("#3498db")
            elif c.startswith("M4"): colors.append("#e74c3c")
            elif c == "M1": colors.append("#2ecc71")
            elif c == "B0": colors.append("#f39c12")
            else: colors.append("#95a5a6")
        ax.barh(range(len(qasper_df)), qasper_df["refusal_rate"], color=colors, edgecolor="white")
        # Annotate with false refusals and hallucinations
        for i, (_, r) in enumerate(qasper_df.iterrows()):
            ax.text(r["refusal_rate"] + 1, i,
                    f"correct={r['n_correct_unans']}, false_ref={r['n_false_refusal']}, halluc={r['n_hallucination']}",
                    fontsize=7, va="center")
        ax.set_yticks(range(len(qasper_df)))
        ax.set_yticklabels(qasper_df["condition"], fontsize=8)
        ax.set_xlabel("Refusal rate (% predictions containing 'unanswerable')")
        ax.set_title(f"Qasper: Who refuses to answer? ({split})")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"unanswerable_qasper_{split}.png", dpi=150)
        plt.close()
        print(f"  Saved unanswerable_qasper_{split}.png")

    return df


# ═══════════════════════════════════════════════════════════════
# Investigation F: Prompt template token budget
# ═══════════════════════════════════════════════════════════════

def investigation_f():
    print(f"\n{'='*60}")
    print("Investigation F: Prompt template token budget")
    print(f"{'='*60}")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except Exception as e:
        print(f"  Could not load tokenizer: {e}")
        return

    # Load prompt templates
    with open(ROOT / "data" / "longbench" / "dataset2prompt.json") as f:
        templates = json.load(f)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets not available")
        return

    rows = []
    for task_lb in ["qasper", "2wikimqa", "hotpotqa_e"]:
        try:
            ds = load_dataset("THUDM/LongBench", task_lb, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Could not load {task_lb}: {e}")
            continue

        for idx in range(min(10, len(ds))):
            sample = ds[idx]
            context = sample.get("context", "")
            question = sample.get("input", "")

            # Template
            if task_lb in templates:
                text = templates[task_lb].format(**sample)
            else:
                text = context + "\nQuestion: " + question + "\nAnswer:"

            # Tokenize just the question + instruction part
            question_part = text[len(context):]  # everything after the context

            # Chat template
            messages = [{"role": "user", "content": text}]
            chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            total_tokens = len(tokenizer.encode(chat_text))
            context_tokens = len(tokenizer.encode(context))
            overhead_tokens = total_tokens - context_tokens

            rows.append({
                "task": task_lb,
                "prompt_idx": idx,
                "total_tokens": total_tokens,
                "context_tokens": context_tokens,
                "overhead_tokens": overhead_tokens,
                "question_chars": len(question),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No data")
        return

    df.to_csv(OUT / "token_budget.csv", index=False)

    # Summary
    print(f"\n  Token budget analysis:")
    print(f"  {'Task':<15} {'Total':>6} {'Context':>7} {'Overhead':>8} {'Question chars':>13}")
    print("  " + "-" * 55)
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        print(f"  {task:<15} {sub['total_tokens'].mean():6.0f} {sub['context_tokens'].mean():7.0f} "
              f"{sub['overhead_tokens'].mean():8.0f} {sub['question_chars'].mean():13.0f}")

    mean_overhead = df["overhead_tokens"].mean()
    print(f"\n  Mean overhead (chat template + question + instruction): {mean_overhead:.0f} tokens")
    print(f"  Effective document budget at 1024 cache: {1024 - mean_overhead:.0f} tokens")
    print(f"  Effective document budget at 2048 cache: {2048 - mean_overhead:.0f} tokens")
    print(f"  Overhead as % of 1024 budget: {mean_overhead/1024*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    tasks = df["task"].unique()
    x = np.arange(len(tasks))
    ctx = [df[df["task"]==t]["context_tokens"].mean() for t in tasks]
    ovh = [df[df["task"]==t]["overhead_tokens"].mean() for t in tasks]
    ax.bar(x, ctx, label="Document context", color="#3498db")
    ax.bar(x, ovh, bottom=ctx, label="Template + question overhead", color="#e74c3c")
    ax.axhline(1024, color="black", linestyle="--", linewidth=1, label="1024 budget")
    ax.axhline(2048, color="grey", linestyle="--", linewidth=1, label="2048 budget")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel("Tokens")
    ax.set_title("Token budget breakdown: how much overhead does the prompt consume?")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "token_budget.png", dpi=150)
    plt.close()
    print(f"  Saved token_budget.png")


# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT, exist_ok=True)

    for split in ["test", "extended_test"]:
        investigation_b(split)
        investigation_e(split)

    investigation_f()

    print("\n" + "=" * 60)
    print("Investigations B, E, F complete.")
    print(f"All outputs: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
