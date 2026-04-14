"""
Recency Investigation: Has NAMM learned recency? Is truncation good because
of answer position bias in LongBench?

Analyses:
1. NAMM recency profile: mean cache age of kept tokens per layer (from eviction traces)
2. Per-task recency breakdown: does NAMM lean more on recency for some tasks?
3. Trunc vs NAMM head-to-head: per-sample comparison — when does NAMM win/lose vs truncation?
4. Answer position analysis: where in the document do gold answers appear?
5. Trunc/lora vs M4 per-task deep dive: which tasks explain the parity?
6. Cache saturation: does NAMM actually fill the budget?

Output: analysis_giacomo/report_recency_investigation/
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

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "main_table_5t"
OUT = Path(__file__).resolve().parent

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e", "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_SHORT = {
    "lb/qasper": "Qasper",
    "lb/2wikimqa": "2WikiMQA",
    "lb/qasper_e": "Qasper-E",
    "lb/hotpotqa_e": "HotpotQA-E",
    "lb/2wikimqa_e": "2WikiMQA-E",
}

# F1 computation
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def token_f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not gt and not pt: return 1.0
    if not gt or not pt: return 0.0
    c = sum((Counter(pt) & Counter(gt)).values())
    if c == 0: return 0.0
    p, r = c/len(pt), c/len(gt)
    return 2*p*r/(p+r)

def best_f1(pred, answers):
    return max(token_f1(pred, a) for a in answers) if answers else 0.0

# Condition paths
COND_TRACES = {
    "M1":                RESULTS / "M1" / "eviction_traces.npz",
    "M4/cs1024":         RESULTS / "M4" / "cs1024" / "eviction_traces.npz",
    "M4/cs2048":         RESULTS / "M4" / "cs2048" / "eviction_traces.npz",
    "M2/cs1024":         RESULTS / "M2" / "cs1024" / "eviction_traces.npz",
    "M2/cs2048":         RESULTS / "M2" / "cs2048" / "eviction_traces.npz",
    "B1/cs1024":         RESULTS / "B1" / "cs1024" / "eviction_traces.npz",
    "B1/cs2048":         RESULTS / "B1" / "cs2048" / "eviction_traces.npz",
    "Trunc/plain_1024":  RESULTS / "Trunc" / "plain_1024" / "eviction_traces.npz",
    "Trunc/plain_2048":  RESULTS / "Trunc" / "plain_2048" / "eviction_traces.npz",
    "A4/cs1024":         RESULTS / "A4" / "cs1024_no_namm" / "eviction_traces.npz",
    "A4/cs2048":         RESULTS / "A4" / "cs2048_no_namm" / "eviction_traces.npz",
    "Trunc/lora_1024":   RESULTS / "Trunc" / "lora_m1_1024" / "eviction_traces.npz",
    "Trunc/lora_2048":   RESULTS / "Trunc" / "lora_m1_2048" / "eviction_traces.npz",
}

COND_GENS = {
    "B0":               RESULTS / "B0" / "generations.json",
    "M1":               RESULTS / "M1" / "generations.json",
    "M4/cs1024":        RESULTS / "M4" / "cs1024" / "generations.json",
    "M4/cs2048":        RESULTS / "M4" / "cs2048" / "generations.json",
    "Trunc/lora_1024":  RESULTS / "Trunc" / "lora_m1_1024" / "generations.json",
    "Trunc/lora_2048":  RESULTS / "Trunc" / "lora_m1_2048" / "generations.json",
    "Trunc/plain_1024": RESULTS / "Trunc" / "plain_1024" / "generations.json",
    "Trunc/plain_2048": RESULTS / "Trunc" / "plain_2048" / "generations.json",
    "M2/cs1024":        RESULTS / "M2" / "cs1024" / "generations.json",
    "A4/cs1024":        RESULTS / "A4" / "cs1024_no_namm" / "generations.json",
    "A4/cs2048":        RESULTS / "A4" / "cs2048_no_namm" / "generations.json",
}

# ═════════════════════════════════════════════════════════════════════
# Analysis 1: Recency profiles from eviction traces
# ═════════════════════════════════════════════════════════════════════

def analysis_recency_profiles(split="test"):
    """How recent are the tokens NAMM keeps? Compare to pure recency and random baselines."""
    print("\n[1] Recency profiles from eviction traces...")

    rows = []
    for cond, path in COND_TRACES.items():
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        rec_key = f"{split}/recorded_final_recencies"
        cs_key = f"{split}/dynamic_cache_sizes"
        fcs_key = f"{split}/final_dynamic_cache_sizes"

        if rec_key not in data:
            continue

        rec = data[rec_key]
        n_layers = len(rec)

        # Mean recency across layers and samples
        layer_means = []
        for layer in range(n_layers):
            r = np.array(rec[layer])
            if r.size > 0:
                layer_means.append(r.mean())
        overall_mean_rec = np.mean(layer_means) if layer_means else 0

        # Cache sizes
        if cs_key in data:
            cs = data[cs_key]
            # Compute mean final cache size across layers and samples
            final_sizes = []
            for layer in range(len(cs)):
                c = np.array(cs[layer])
                if c.size > 0:
                    final_sizes.append(c[-1] if c.ndim == 1 else c.mean())
            mean_cs = np.mean(final_sizes) if final_sizes else 0
        else:
            mean_cs = 0

        rows.append({
            "condition": cond,
            "mean_recency": overall_mean_rec,
            "mean_cache_size": mean_cs,
            "n_layers": n_layers,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"recency_profiles_{split}.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values("mean_recency")
    colors = []
    for c in df_sorted["condition"]:
        if "Trunc" in c:
            colors.append("#3498db")
        elif c.startswith("M4"):
            colors.append("#e74c3c")
        elif c.startswith("M2"):
            colors.append("#f39c12")
        elif c.startswith("B1"):
            colors.append("#95a5a6")
        elif c == "M1":
            colors.append("#2ecc71")
        else:
            colors.append("#9b59b6")

    bars = ax.barh(range(len(df_sorted)), df_sorted["mean_recency"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["condition"], fontsize=9)
    ax.set_xlabel("Mean recency of kept tokens (higher = older tokens kept)")
    ax.set_title(f"NAMM Recency Profile ({split} split)\n"
                 "Blue=truncation, Red=M4(NAMM+LoRA), Orange=M2(NAMM+randLoRA), Green=M1(NAMM)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"recency_profiles_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved recency_profiles_{split}.png")

    # Print summary
    print(f"\n  Recency summary ({split}):")
    for _, row in df_sorted.iterrows():
        print(f"    {row['condition']:<20} mean_rec={row['mean_recency']:7.1f}")

    return df


# ═════════════════════════════════════════════════════════════════════
# Analysis 2: Per-layer recency profile (does NAMM evict differently at different depths?)
# ═════════════════════════════════════════════════════════════════════

def analysis_per_layer_recency(split="test"):
    """Per-layer recency for key NAMM conditions."""
    print("\n[2] Per-layer recency profiles...")

    key_conds = ["M4/cs1024", "M4/cs2048", "M2/cs1024", "B1/cs1024", "Trunc/plain_1024"]

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for ci, cond in enumerate(key_conds):
        path = COND_TRACES.get(cond)
        if not path or not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        rec = data[f"{split}/recorded_final_recencies"]

        layer_means = []
        for layer in range(len(rec)):
            r = np.array(rec[layer])
            layer_means.append(r.mean() if r.size > 0 else 0)

        ax.plot(range(len(layer_means)), layer_means, "o-", label=cond,
                color=cmap(ci), linewidth=1.5, markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean recency of kept tokens")
    ax.set_title(f"Per-Layer Recency Profile ({split})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"per_layer_recency_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved per_layer_recency_{split}.png")


# ═════════════════════════════════════════════════════════════════════
# Analysis 3: Cache saturation — does NAMM fill the budget?
# ═════════════════════════════════════════════════════════════════════

def analysis_cache_saturation(split="test"):
    """Check if NAMM fills the cache budget or leaves empty slots."""
    print("\n[3] Cache saturation analysis...")

    rows = []
    for cond in ["M4/cs1024", "M4/cs2048", "M2/cs1024", "M2/cs2048", "B1/cs1024", "B1/cs2048"]:
        path = COND_TRACES.get(cond)
        if not path or not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        cs_key = f"{split}/dynamic_cache_sizes"
        fcs_key = f"{split}/final_dynamic_cache_sizes"

        if cs_key not in data:
            continue

        cs = data[cs_key]
        # dynamic_cache_sizes per layer: array of cache sizes at each decoding step
        # Check the final sizes vs budget
        budget = 1024 if "1024" in cond else 2048

        for layer in range(len(cs)):
            c = np.array(cs[layer])
            if c.size == 0:
                continue
            # Final cache sizes for this layer
            final_sizes = c.reshape(-1)
            # How many steps hit the budget?
            at_budget = (final_sizes >= budget).sum()
            below_budget = (final_sizes < budget).sum()
            mean_size = final_sizes.mean()
            rows.append({
                "condition": cond,
                "layer": layer,
                "budget": budget,
                "mean_cache_size": mean_size,
                "saturation_pct": at_budget / len(final_sizes) * 100,
                "mean_utilization": mean_size / budget * 100,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No saturation data available")
        return df

    # Aggregate by condition
    agg = df.groupby("condition").agg({
        "budget": "first",
        "mean_cache_size": "mean",
        "saturation_pct": "mean",
        "mean_utilization": "mean",
    }).round(1)

    print(f"\n  Cache saturation ({split}):")
    print(agg.to_string())

    # Plot per-layer saturation for M4/cs1024
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cond in zip(axes, ["M4/cs1024", "M4/cs2048"]):
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        budget = sub["budget"].iloc[0]
        ax.bar(sub["layer"], sub["mean_utilization"], color="#e74c3c", alpha=0.8)
        ax.axhline(100, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cache utilization (%)")
        ax.set_title(f"{cond}: Cache Utilization per Layer (budget={budget})")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / f"cache_saturation_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved cache_saturation_{split}.png")

    df.to_csv(OUT / f"cache_saturation_{split}.csv", index=False)
    return df


# ═════════════════════════════════════════════════════════════════════
# Analysis 4: Head-to-head — Trunc/lora vs M4 per sample
# ═════════════════════════════════════════════════════════════════════

def analysis_head_to_head(split="test"):
    """Per-sample comparison: when does Trunc/lora beat M4 and vice versa?"""
    print(f"\n[4] Head-to-head: Trunc/lora vs M4 ({split})...")

    comparisons = [
        ("Trunc/lora_2048", "M4/cs2048", "Same 2K budget"),
        ("Trunc/lora_1024", "M4/cs1024", "Same 1K budget"),
        ("Trunc/lora_2048", "M4/cs1024", "Trunc has 2x budget"),
    ]

    all_rows = []
    report_lines = [f"# Head-to-Head: Trunc/lora vs M4 ({split})\n\n"]

    for cond_a, cond_b, desc in comparisons:
        gen_a_path = COND_GENS.get(cond_a)
        gen_b_path = COND_GENS.get(cond_b)
        if not gen_a_path or not gen_b_path:
            continue
        if not gen_a_path.exists() or not gen_b_path.exists():
            continue

        with open(gen_a_path) as f:
            gen_a = json.load(f)
        with open(gen_b_path) as f:
            gen_b = json.load(f)

        if split not in gen_a or split not in gen_b:
            continue

        report_lines.append(f"## {cond_a} vs {cond_b} ({desc})\n\n")

        task_results = {}
        for task in TASKS:
            if task not in gen_a[split] or task not in gen_b[split]:
                continue

            items_a = {it["prompt_idx"]: it for it in gen_a[split][task]}
            items_b = {it["prompt_idx"]: it for it in gen_b[split][task]}
            shared = set(items_a.keys()) & set(items_b.keys())

            a_wins = 0
            b_wins = 0
            ties = 0
            a_total_f1 = 0
            b_total_f1 = 0

            for idx in shared:
                f1_a = best_f1(items_a[idx]["pred"], items_a[idx]["answers"])
                f1_b = best_f1(items_b[idx]["pred"], items_b[idx]["answers"])
                a_total_f1 += f1_a
                b_total_f1 += f1_b

                if f1_a > f1_b + 0.01:
                    a_wins += 1
                elif f1_b > f1_a + 0.01:
                    b_wins += 1
                else:
                    ties += 1

                all_rows.append({
                    "comparison": f"{cond_a} vs {cond_b}",
                    "task": TASK_SHORT.get(task, task),
                    "prompt_idx": idx,
                    "f1_a": f1_a,
                    "f1_b": f1_b,
                    "delta": f1_a - f1_b,
                    "winner": cond_a if f1_a > f1_b + 0.01 else (cond_b if f1_b > f1_a + 0.01 else "tie"),
                    "length": items_a[idx].get("length", 0),
                })

            n = len(shared)
            mean_a = a_total_f1 / n * 100 if n else 0
            mean_b = b_total_f1 / n * 100 if n else 0
            task_results[TASK_SHORT.get(task, task)] = {
                "n": n, "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
                "mean_a": mean_a, "mean_b": mean_b,
            }

        # Write task-level table
        report_lines.append(f"| Task | n | {cond_a} wins | {cond_b} wins | Ties | {cond_a} F1 | {cond_b} F1 | Delta |\n")
        report_lines.append("|---|---|---|---|---|---|---|---|\n")
        for task_name, r in task_results.items():
            delta = r["mean_a"] - r["mean_b"]
            report_lines.append(
                f"| {task_name} | {r['n']} | {r['a_wins']} | {r['b_wins']} | {r['ties']} | "
                f"{r['mean_a']:.1f} | {r['mean_b']:.1f} | {delta:+.1f} |\n"
            )
        report_lines.append("\n")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_csv(OUT / f"head_to_head_{split}.csv", index=False)

    with open(OUT / f"head_to_head_{split}.md", "w") as f:
        f.writelines(report_lines)
    print(f"  Saved head_to_head_{split}.md")

    # Plot: per-task win counts for same-budget comparison
    if not df.empty:
        comp = f"Trunc/lora_2048 vs M4/cs2048"
        sub = df[df["comparison"] == comp]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            tasks = sub["task"].unique()
            x = np.arange(len(tasks))
            a_wins = []
            b_wins = []
            for t in tasks:
                ts = sub[sub["task"] == t]
                a_wins.append((ts["winner"] == "Trunc/lora_2048").sum())
                b_wins.append((ts["winner"] == "M4/cs2048").sum())

            width = 0.35
            ax.bar(x - width/2, a_wins, width, label="Trunc/lora_2048 wins", color="#3498db")
            ax.bar(x + width/2, b_wins, width, label="M4/cs2048 wins", color="#e74c3c")
            ax.set_xticks(x)
            ax.set_xticklabels(tasks, fontsize=9)
            ax.set_ylabel("Number of samples won")
            ax.set_title("Per-task sample wins: Trunc/lora_2048 vs M4/cs2048 (same 2K budget)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(OUT / f"head_to_head_wins_{split}.png", dpi=150)
            plt.close()
            print(f"  Saved head_to_head_wins_{split}.png")

    return df


# ═════════════════════════════════════════════════════════════════════
# Analysis 5: Answer position in documents
# ═════════════════════════════════════════════════════════════════════

def analysis_answer_position(split="test"):
    """Where in the document does the gold answer appear?
    If answers cluster at the end, truncation's strength is explained by position bias."""
    print(f"\n[5] Answer position analysis ({split})...")

    # Load B0 generations (full context, so pred location reflects true position)
    gen_path = COND_GENS.get("B0")
    if not gen_path or not gen_path.exists():
        print("  B0 generations not found")
        return

    with open(gen_path) as f:
        gen_data = json.load(f)

    if split not in gen_data:
        print(f"  Split '{split}' not in B0 generations")
        return

    # We can't know exact answer position from generations.json alone.
    # But we CAN analyze: for each sample, how long is the document and
    # does truncation (keeping last K tokens) include the answer?
    # Proxy: compare Trunc/plain vs B0 per sample — if Trunc gets it right,
    # the answer was in the tail.

    trunc_gens = {}
    for cond in ["Trunc/plain_1024", "Trunc/plain_2048"]:
        path = COND_GENS.get(cond)
        if path and path.exists():
            with open(path) as f:
                trunc_gens[cond] = json.load(f)

    rows = []
    for task in TASKS:
        if task not in gen_data[split]:
            continue

        b0_items = {it["prompt_idx"]: it for it in gen_data[split][task]}

        for cond, tg in trunc_gens.items():
            if split not in tg or task not in tg[split]:
                continue
            trunc_items = {it["prompt_idx"]: it for it in tg[split][task]}

            for idx in set(b0_items.keys()) & set(trunc_items.keys()):
                b0 = b0_items[idx]
                tr = trunc_items[idx]
                f1_b0 = best_f1(b0["pred"], b0["answers"])
                f1_tr = best_f1(tr["pred"], tr["answers"])
                length = b0.get("length", 0)

                rows.append({
                    "task": TASK_SHORT.get(task, task),
                    "prompt_idx": idx,
                    "length": length,
                    "truncation": cond,
                    "f1_full": f1_b0,
                    "f1_trunc": f1_tr,
                    "trunc_wins": f1_tr > f1_b0 + 0.01,
                    "trunc_matches": abs(f1_tr - f1_b0) <= 0.01,
                    "answer_in_tail": f1_tr > 0.1,  # proxy: if trunc gets any F1, answer is partly in tail
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No data for answer position analysis")
        return df

    df.to_csv(OUT / f"answer_position_{split}.csv", index=False)

    # Per-task: fraction of samples where answer is in the tail
    print(f"\n  Answer-in-tail analysis ({split}):")
    for trunc in df["truncation"].unique():
        sub = df[df["truncation"] == trunc]
        budget = "1024" if "1024" in trunc else "2048"
        print(f"\n  {trunc} (keep last {budget} tokens):")
        for task in sub["task"].unique():
            ts = sub[sub["task"] == task]
            n = len(ts)
            in_tail = ts["answer_in_tail"].sum()
            matches = ts["trunc_matches"].sum()
            wins = ts["trunc_wins"].sum()
            print(f"    {task:<12}: {in_tail}/{n} ({in_tail/n*100:.0f}%) answer in tail, "
                  f"{matches}/{n} matches B0, {wins}/{n} trunc beats B0")

    # Plot: answer-in-tail fraction by task and truncation budget
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, trunc in zip(axes, sorted(df["truncation"].unique())):
        sub = df[df["truncation"] == trunc]
        tasks = sorted(sub["task"].unique())
        tail_fracs = []
        match_fracs = []
        for task in tasks:
            ts = sub[sub["task"] == task]
            tail_fracs.append(ts["answer_in_tail"].mean() * 100)
            match_fracs.append(ts["trunc_matches"].mean() * 100)

        x = np.arange(len(tasks))
        ax.bar(x, tail_fracs, color="#e74c3c", alpha=0.8, label="Answer in tail (F1>0.1)")
        ax.bar(x, match_fracs, color="#2ecc71", alpha=0.6, label="Matches full-context B0")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("% of samples")
        ax.set_title(f"{trunc}")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Answer Position Bias: Is the gold answer in the truncated tail?", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / f"answer_position_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved answer_position_{split}.png")

    return df


# ═════════════════════════════════════════════════════════════════════
# Analysis 6: LoRA attribution — how much does LoRA vs eviction strategy contribute?
# ═════════════════════════════════════════════════════════════════════

def analysis_lora_attribution(split="test"):
    """Decompose: how much of M4's advantage comes from LoRA vs NAMM?"""
    print(f"\n[6] LoRA vs NAMM attribution ({split})...")

    with open(RESULTS / "all_results.json") as f:
        all_res = json.load(f)

    # Attribution at 2048 budget:
    #   plain truncation → +LoRA → +NAMM(M4)
    # Attribution at 1024 budget:
    #   plain truncation → +LoRA → +NAMM(M4)
    report = [f"# LoRA vs NAMM Attribution ({split})\n\n"]

    for budget, trunc_plain, trunc_lora, m4, m2 in [
        (1024, "Trunc/plain_1024", "Trunc/lora_m1_1024", "M4/cs1024", "M2/cs1024"),
        (2048, "Trunc/plain_2048", "Trunc/lora_m1_2048", "M4/cs2048", "M2/cs2048"),
    ]:
        report.append(f"## Budget: {budget} tokens\n\n")
        report.append(f"| Task | Trunc/plain | +LoRA (delta) | +NAMM→M4 (delta) | LoRA% | NAMM% |\n")
        report.append(f"|---|---|---|---|---|---|\n")

        total_lora_delta = 0
        total_namm_delta = 0

        for task in TASKS + ["mean"]:
            tp = all_res.get(trunc_plain, {}).get(split, {}).get(task, 0)
            tl = all_res.get(trunc_lora, {}).get(split, {}).get(task, 0)
            m4v = all_res.get(m4, {}).get(split, {}).get(task, 0)

            lora_delta = tl - tp
            namm_delta = m4v - tl
            total_delta = m4v - tp

            lora_pct = lora_delta / total_delta * 100 if abs(total_delta) > 0.1 else 0
            namm_pct = namm_delta / total_delta * 100 if abs(total_delta) > 0.1 else 0

            task_name = TASK_SHORT.get(task, task) if task != "mean" else "**MEAN**"
            report.append(f"| {task_name} | {tp:.1f} | {tl:.1f} ({lora_delta:+.1f}) | "
                         f"{m4v:.1f} ({namm_delta:+.1f}) | {lora_pct:.0f}% | {namm_pct:.0f}% |\n")

            if task == "mean":
                total_lora_delta = lora_delta
                total_namm_delta = namm_delta

        report.append(f"\n**LoRA contributes {total_lora_delta:+.1f} F1, NAMM adds {total_namm_delta:+.1f} F1**\n\n")

        # Also: NAMM-only comparison (M2 = trained NAMM + random LoRA)
        m2v_mean = all_res.get(m2, {}).get(split, {}).get("mean", 0)
        tp_mean = all_res.get(trunc_plain, {}).get(split, {}).get("mean", 0)
        report.append(f"NAMM-only (M2 vs Trunc/plain): M2={m2v_mean:.1f}, Trunc={tp_mean:.1f}, "
                     f"NAMM alone adds {m2v_mean-tp_mean:+.1f}\n\n")

    with open(OUT / f"lora_attribution_{split}.md", "w") as f:
        f.writelines(report)
    print(f"  Saved lora_attribution_{split}.md")

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, budget, trunc_plain, trunc_lora, m4 in [
        (axes[0], 1024, "Trunc/plain_1024", "Trunc/lora_m1_1024", "M4/cs1024"),
        (axes[1], 2048, "Trunc/plain_2048", "Trunc/lora_m1_2048", "M4/cs2048"),
    ]:
        task_names = [TASK_SHORT[t] for t in TASKS]
        tp_vals = [all_res.get(trunc_plain, {}).get(split, {}).get(t, 0) for t in TASKS]
        tl_vals = [all_res.get(trunc_lora, {}).get(split, {}).get(t, 0) for t in TASKS]
        m4_vals = [all_res.get(m4, {}).get(split, {}).get(t, 0) for t in TASKS]

        x = np.arange(len(task_names))
        width = 0.25
        ax.bar(x - width, tp_vals, width, label=f"Trunc/plain", color="#95a5a6")
        ax.bar(x, tl_vals, width, label=f"+ LoRA", color="#3498db")
        ax.bar(x + width, m4_vals, width, label=f"+ NAMM (M4)", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("F1 (%)")
        ax.set_title(f"Budget: {budget} tokens")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Attribution: Truncation → +LoRA → +NAMM ({split})", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / f"lora_attribution_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved lora_attribution_{split}.png")


# ═════════════════════════════════════════════════════════════════════
# Analysis 7: The per-task decomposition — WHY does Trunc/lora match M4?
# ═════════════════════════════════════════════════════════════════════

def analysis_per_task_decomposition(split="test"):
    """Which specific tasks explain Trunc/lora nearly matching M4?"""
    print(f"\n[7] Per-task decomposition ({split})...")

    with open(RESULTS / "all_results.json") as f:
        all_res = json.load(f)

    report = [f"# Per-Task Decomposition: Why Trunc/lora ≈ M4 ({split})\n\n"]

    for budget, tl, m4 in [
        (2048, "Trunc/lora_m1_2048", "M4/cs2048"),
        (1024, "Trunc/lora_m1_1024", "M4/cs1024"),
    ]:
        report.append(f"## {budget}-token budget\n\n")
        report.append(f"| Task | Trunc/lora | M4 | M4-Trunc | Verdict |\n")
        report.append(f"|---|---|---|---|---|\n")

        for task in TASKS:
            tl_v = all_res.get(tl, {}).get(split, {}).get(task, 0)
            m4_v = all_res.get(m4, {}).get(split, {}).get(task, 0)
            delta = m4_v - tl_v

            if delta > 3:
                verdict = "**NAMM helps**"
            elif delta < -3:
                verdict = "**Trunc wins**"
            else:
                verdict = "~tied"

            report.append(f"| {TASK_SHORT[task]} | {tl_v:.1f} | {m4_v:.1f} | {delta:+.1f} | {verdict} |\n")

        tl_mean = all_res.get(tl, {}).get(split, {}).get("mean", 0)
        m4_mean = all_res.get(m4, {}).get(split, {}).get("mean", 0)
        report.append(f"| **Mean** | **{tl_mean:.1f}** | **{m4_mean:.1f}** | **{m4_mean-tl_mean:+.1f}** | |\n\n")

    with open(OUT / f"per_task_decomposition_{split}.md", "w") as f:
        f.writelines(report)
    print(f"  Saved per_task_decomposition_{split}.md")


# ═════════════════════════════════════════════════════════════════════

def write_master_report():
    """Write the synthesis report."""
    print("\n[8] Writing master report...")

    lines = [
        "# Recency Investigation: Has NAMM Learned Recency?\n\n",
        "## The question\n\n",
        "Trunc/lora_2048 (keep last 2048 tokens + M1's LoRA) nearly matches M4/cs2048 (NAMM eviction + jointly trained LoRA). "
        "On the extended_test split, Trunc/lora_2048 actually BEATS M4. Is this because:\n",
        "1. NAMM has essentially learned recency (keeping last N tokens)\n",
        "2. Recency IS a strong prior for these tasks (answers at document end)\n",
        "3. LoRA does most of the work and eviction strategy barely matters\n\n",
        "## Evidence from this investigation\n\n",
        "See the individual analysis files for detailed tables and plots.\n\n",
        "### Key outputs\n",
        "- `recency_profiles_*.png/csv` — Mean recency of kept tokens per condition\n",
        "- `per_layer_recency_*.png` — Per-layer recency profile\n",
        "- `cache_saturation_*.png/csv` — Does NAMM fill the cache budget?\n",
        "- `head_to_head_*.md/csv/png` — Per-sample Trunc/lora vs M4 comparison\n",
        "- `answer_position_*.png/csv` — Is the gold answer in the truncated tail?\n",
        "- `lora_attribution_*.md/png` — Decomposition: LoRA vs NAMM contribution\n",
        "- `per_task_decomposition_*.md` — Which tasks explain Trunc/lora ≈ M4?\n",
    ]

    with open(OUT / "recency_report.md", "w") as f:
        f.writelines(lines)
    print("  Saved recency_report.md")


# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("RECENCY INVESTIGATION")
    print("=" * 60)

    for split in ["test", "extended_test"]:
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        analysis_recency_profiles(split)
        analysis_per_layer_recency(split)
        analysis_cache_saturation(split)
        analysis_head_to_head(split)
        analysis_answer_position(split)
        analysis_lora_attribution(split)
        analysis_per_task_decomposition(split)

    write_master_report()

    print("\n" + "=" * 60)
    print("RECENCY INVESTIGATION COMPLETE")
    print(f"All outputs: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
