"""Analysis 10 — Maskfix Comparison: Buggy vs Fixed Attention Mask.

Compares NAMM training (M2) and LoRA+NAMM training (M3) between the
original buggy attention mask and the fixed version (maskfix runs).

Pulls data from WandB and generates comparison plots + report.

Usage:
    PYTHONPATH=. .venv/bin/python analysis/report_10/generate_plots.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = SCRIPT_DIR

logger = logging.getLogger("analysis_10")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"
TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_LABELS = {
    "qasper": "Qasper", "2wikimqa": "2WikiMQA", "qasper_e": "Qasper-E",
    "hotpotqa_e": "HotpotQA-E", "2wikimqa_e": "2WikiMQA-E",
}

# Run IDs
M2_BUGGY = ["lenhmfb1"]
M2_MASKFIX = ["z5bo4n8k"]
M3_BUGGY = ["ovosogkj"]
M3_MASKFIX = ["h0bzg6on"]
M1_SEGMENTS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]
B0_RUN = "kz6vqo2o"


def get_api():
    return wandb.Api()


def fetch_namm_history(api, run_ids):
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        h = r.history(pandas=True, samples=10000)
        if not h.empty:
            if frames:
                prev_max = frames[-1]["iter"].max()
                h = h[h["iter"] > prev_max]
            frames.append(h)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_lora_history(api, run_ids):
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        h = r.history(pandas=True, samples=10000)
        if not h.empty:
            if frames:
                prev_max = frames[-1]["lora/global_step"].max()
                h = h[h["lora/global_step"] > prev_max]
            frames.append(h)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["lora/global_step"])
    df = df.drop_duplicates(subset=["lora/global_step"], keep="first")
    return df.sort_values("lora/global_step").reset_index(drop=True)


def fetch_b0(api):
    r = api.run(f"{ENTITY}/{PROJECT}/{B0_RUN}")
    keys = [f"lora/baseline_lb_{t}" for t in TASKS] + ["lora/baseline_lb_avg_f1"]
    h = r.history(keys=keys, pandas=True, samples=10000)
    h = h.dropna(subset=["lora/baseline_lb_avg_f1"])
    row = h.iloc[0]
    return {t: row[f"lora/baseline_lb_{t}"] for t in TASKS}, row["lora/baseline_lb_avg_f1"]


def fetch_m1_best(api):
    best_avg = -1.0
    best_row = None
    for rid in M1_SEGMENTS:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        keys = ["lora/val_lb_avg_f1"] + [f"lora/val_lb_{t}" for t in TASKS]
        h = r.history(keys=keys, pandas=True, samples=10000)
        if h.empty:
            continue
        h = h.dropna(subset=["lora/val_lb_avg_f1"])
        if h.empty:
            continue
        idx = h["lora/val_lb_avg_f1"].idxmax()
        val = h.loc[idx, "lora/val_lb_avg_f1"]
        if val > best_avg:
            best_avg = val
            best_row = {t: h.loc[idx, f"lora/val_lb_{t}"] for t in TASKS}
            best_row["mean"] = val
    return best_row


def main():
    api = get_api()
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Fetching data from WandB...")

    b0_scores, b0_mean = fetch_b0(api)
    m1_best = fetch_m1_best(api)
    logger.info("B0 mean: %.2f, M1 best mean: %.2f", b0_mean, m1_best["mean"])

    m2_buggy = fetch_namm_history(api, M2_BUGGY)
    m2_maskfix = fetch_namm_history(api, M2_MASKFIX)
    m3_buggy = fetch_lora_history(api, M3_BUGGY)
    m3_maskfix = fetch_lora_history(api, M3_MASKFIX)

    logger.info("M2 buggy: %d rows, M2 maskfix: %d rows", len(m2_buggy), len(m2_maskfix))
    logger.info("M3 buggy: %d rows, M3 maskfix: %d rows", len(m3_buggy), len(m3_maskfix))

    # ── Plot 1: M2 mean val F1 comparison ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, df, color, ls in [
        ("M2 buggy (cs1024)", m2_buggy, "#1f77b4", "--"),
        ("M2 maskfix (cs1024)", m2_maskfix, "#d62728", "-"),
    ]:
        sub = df.dropna(subset=[f"val_lb/{TASKS[0]}"])
        if sub.empty:
            continue
        sub = sub.copy()
        sub["mean_f1"] = sub[[f"val_lb/{t}" for t in TASKS]].mean(axis=1)
        ax.plot(sub["iter"], sub["mean_f1"], label=label, color=color,
                linewidth=2, linestyle=ls)

    ax.axhline(b0_mean, color="#7f7f7f", linestyle=":", linewidth=1.5,
               label=f"B0 ({b0_mean:.1f})")
    ax.axhline(m1_best["mean"], color="#ff7f0e", linestyle=":", linewidth=1.5,
               label=f"M1 ({m1_best['mean']:.1f})")
    ax.set_xlabel("CMA-ES Iteration", fontsize=12)
    ax.set_ylabel("Val Mean F1", fontsize=12)
    ax.set_title("M2 (NAMM-only cs1024) — Buggy vs Maskfix", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m2_buggy_vs_maskfix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved m2_buggy_vs_maskfix.png")

    # ── Plot 2: M3 mean val F1 comparison ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, df, color, ls in [
        ("M3 buggy (cs1024)", m3_buggy, "#1f77b4", "--"),
        ("M3 maskfix (cs1024)", m3_maskfix, "#d62728", "-"),
    ]:
        sub = df.dropna(subset=["lora/val_lb_avg_f1"])
        if sub.empty:
            continue
        # Raw faint
        ax.plot(sub["lora/global_step"], sub["lora/val_lb_avg_f1"],
                color=color, alpha=0.15, linewidth=1)
        # Smoothed
        smoothed = sub["lora/val_lb_avg_f1"].rolling(5, min_periods=1, center=True).mean()
        ax.plot(sub["lora/global_step"], smoothed, label=label, color=color,
                linewidth=2, linestyle=ls)

    ax.axhline(m1_best["mean"], color="#ff7f0e", linestyle=":", linewidth=1.5,
               label=f"M1 ({m1_best['mean']:.1f})")
    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Val Mean F1", fontsize=12)
    ax.set_title("M3 (LoRA + frozen NAMM cs1024) — Buggy vs Maskfix\n(smoothed, window=5; faint = raw)",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m3_buggy_vs_maskfix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved m3_buggy_vs_maskfix.png")

    # ── Plot 3: M2 per-task comparison ───────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes[i]
        val_key = f"val_lb/{task}"

        for label, df, color, ls in [
            ("M2 buggy", m2_buggy, "#1f77b4", "--"),
            ("M2 maskfix", m2_maskfix, "#d62728", "-"),
        ]:
            sub = df.dropna(subset=[val_key])
            if not sub.empty:
                ax.plot(sub["iter"], sub[val_key], label=label, color=color,
                        linewidth=1.5, linestyle=ls)

        if task in b0_scores:
            ax.axhline(b0_scores[task], color="#7f7f7f", linestyle=":",
                       linewidth=1, label=f"B0 ({b0_scores[task]:.1f})")
        if task in m1_best:
            ax.axhline(m1_best[task], color="#ff7f0e", linestyle=":",
                       linewidth=1, label=f"M1 ({m1_best[task]:.1f})")

        ax.set_title(TASK_LABELS[task], fontsize=13, fontweight="bold")
        ax.set_xlabel("CMA-ES Iteration", fontsize=10)
        ax.set_ylabel("Val F1", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    axes[5].set_visible(False)
    fig.suptitle("M2 Per-Task: Buggy vs Maskfix (cs1024)", fontsize=15,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "m2_per_task_buggy_vs_maskfix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved m2_per_task_buggy_vs_maskfix.png")

    # ── Plot 4: M3 per-task comparison ───────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes[i]
        val_key = f"lora/val_lb_{task}"

        for label, df, color, ls in [
            ("M3 buggy", m3_buggy, "#1f77b4", "--"),
            ("M3 maskfix", m3_maskfix, "#d62728", "-"),
        ]:
            sub = df.dropna(subset=[val_key])
            if sub.empty:
                continue
            # Raw faint
            ax.plot(sub["lora/global_step"], sub[val_key],
                    color=color, alpha=0.15, linewidth=1)
            # Smoothed
            smoothed = sub[val_key].rolling(5, min_periods=1, center=True).mean()
            ax.plot(sub["lora/global_step"], smoothed, label=label,
                    color=color, linewidth=1.5, linestyle=ls)

        if task in m1_best:
            ax.axhline(m1_best[task], color="#ff7f0e", linestyle=":",
                       linewidth=1, label=f"M1 ({m1_best[task]:.1f})")

        ax.set_title(TASK_LABELS[task], fontsize=13, fontweight="bold")
        ax.set_xlabel("Global Step", fontsize=10)
        ax.set_ylabel("Val F1", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    axes[5].set_visible(False)
    fig.suptitle("M3 Per-Task: Buggy vs Maskfix (cs1024)\n(smoothed, window=5; faint = raw)",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "m3_per_task_buggy_vs_maskfix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved m3_per_task_buggy_vs_maskfix.png")

    # ── Plot 5: Bar chart — best val F1 comparison ──────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    # Compute best per-task for all conditions
    conditions = {}

    # B0
    conditions["B0"] = {t: b0_scores[t] for t in TASKS}
    conditions["B0"]["mean"] = b0_mean

    # M1
    conditions["M1"] = m1_best

    # M2 buggy
    sub = m2_buggy.dropna(subset=[f"val_lb/{TASKS[0]}"])
    if len(sub) > 0:
        sub = sub.copy()
        sub["mean_f1"] = sub[[f"val_lb/{t}" for t in TASKS]].mean(axis=1)
        best = sub.loc[sub["mean_f1"].idxmax()]
        conditions["M2 buggy"] = {t: best[f"val_lb/{t}"] for t in TASKS}
        conditions["M2 buggy"]["mean"] = best["mean_f1"]

    # M2 maskfix
    sub = m2_maskfix.dropna(subset=[f"val_lb/{TASKS[0]}"])
    if len(sub) > 0:
        sub = sub.copy()
        sub["mean_f1"] = sub[[f"val_lb/{t}" for t in TASKS]].mean(axis=1)
        best = sub.loc[sub["mean_f1"].idxmax()]
        conditions["M2 maskfix"] = {t: best[f"val_lb/{t}"] for t in TASKS}
        conditions["M2 maskfix"]["mean"] = best["mean_f1"]

    # M3 buggy
    sub = m3_buggy.dropna(subset=["lora/val_lb_avg_f1"])
    if len(sub) > 0:
        best = sub.loc[sub["lora/val_lb_avg_f1"].idxmax()]
        conditions["M3 buggy"] = {t: best[f"lora/val_lb_{t}"] for t in TASKS}
        conditions["M3 buggy"]["mean"] = best["lora/val_lb_avg_f1"]

    # M3 maskfix
    sub = m3_maskfix.dropna(subset=["lora/val_lb_avg_f1"])
    if len(sub) > 0:
        best = sub.loc[sub["lora/val_lb_avg_f1"].idxmax()]
        conditions["M3 maskfix"] = {t: best[f"lora/val_lb_{t}"] for t in TASKS}
        conditions["M3 maskfix"]["mean"] = best["lora/val_lb_avg_f1"]

    cond_names = list(conditions.keys())
    cond_colors = {
        "B0": "#7f7f7f", "M1": "#ff7f0e",
        "M2 buggy": "#aec7e8", "M2 maskfix": "#1f77b4",
        "M3 buggy": "#ffbb78", "M3 maskfix": "#d62728",
    }

    group_labels = [TASK_LABELS[t] for t in TASKS] + ["Mean"]
    x = np.arange(len(group_labels))
    n_bars = len(cond_names)
    width = 0.8 / n_bars

    for i, cond in enumerate(cond_names):
        vals = [conditions[cond].get(t, 0) for t in TASKS] + [conditions[cond].get("mean", 0)]
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=cond,
                      color=cond_colors.get(cond, "#333"), edgecolor="black",
                      linewidth=0.3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel("Best Val F1", fontsize=12)
    ax.set_title("Best Validation F1: Buggy vs Maskfix (cs1024)\n"
                 "Note: M3 maskfix still running (step 294/~684)",
                 fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_val_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved best_val_comparison.png")

    # ── Save data ────────────────────────────────────────────────────────
    results = {
        "conditions": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in conditions.items()},
        "m3_maskfix_status": "running (step 294, ~43% complete)",
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results.json")

    # ── Write report ─────────────────────────────────────────────────────
    report = f"""# Analysis 10: Maskfix Comparison — Buggy vs Fixed Attention Mask

## Overview

This analysis compares NAMM and LoRA+NAMM training between the original
implementation (with the attention mask bug causing entropy collapse from
chunk 9 onward) and the maskfix version (with correct attention throughout
prefill). All comparisons are at cache_size=1024.

> **M3 maskfix is still running** (step 294 of ~684, ~43% complete).
> The results below reflect the best checkpoint so far, not the final result.

## Results

### Best validation F1 (cs1024)

| Condition     | Qasper | 2WikiMQA | Qasper-E | HotpotQA-E | 2WikiMQA-E | Mean  |
| ------------- | -----: | -------: | -------: | ---------: | ---------: | ----: |
| B0            | {conditions['B0']['qasper']:.2f} | {conditions['B0']['2wikimqa']:.2f} | {conditions['B0']['qasper_e']:.2f} | {conditions['B0']['hotpotqa_e']:.2f} | {conditions['B0']['2wikimqa_e']:.2f} | {conditions['B0']['mean']:.2f} |
| M1            | {conditions['M1']['qasper']:.2f} | {conditions['M1']['2wikimqa']:.2f} | {conditions['M1']['qasper_e']:.2f} | {conditions['M1']['hotpotqa_e']:.2f} | {conditions['M1']['2wikimqa_e']:.2f} | {conditions['M1']['mean']:.2f} |
| M2 buggy      | {conditions['M2 buggy']['qasper']:.2f} | {conditions['M2 buggy']['2wikimqa']:.2f} | {conditions['M2 buggy']['qasper_e']:.2f} | {conditions['M2 buggy']['hotpotqa_e']:.2f} | {conditions['M2 buggy']['2wikimqa_e']:.2f} | {conditions['M2 buggy']['mean']:.2f} |
| M2 maskfix    | {conditions['M2 maskfix']['qasper']:.2f} | {conditions['M2 maskfix']['2wikimqa']:.2f} | {conditions['M2 maskfix']['qasper_e']:.2f} | {conditions['M2 maskfix']['hotpotqa_e']:.2f} | {conditions['M2 maskfix']['2wikimqa_e']:.2f} | {conditions['M2 maskfix']['mean']:.2f} |
| M3 buggy      | {conditions['M3 buggy']['qasper']:.2f} | {conditions['M3 buggy']['2wikimqa']:.2f} | {conditions['M3 buggy']['qasper_e']:.2f} | {conditions['M3 buggy']['hotpotqa_e']:.2f} | {conditions['M3 buggy']['2wikimqa_e']:.2f} | {conditions['M3 buggy']['mean']:.2f} |
| M3 maskfix*   | {conditions['M3 maskfix']['qasper']:.2f} | {conditions['M3 maskfix']['2wikimqa']:.2f} | {conditions['M3 maskfix']['qasper_e']:.2f} | {conditions['M3 maskfix']['hotpotqa_e']:.2f} | {conditions['M3 maskfix']['2wikimqa_e']:.2f} | {conditions['M3 maskfix']['mean']:.2f} |

*M3 maskfix is still running (step 294/~684). These are interim results.

### Key findings

1. **M2 maskfix (14.90) is substantially WORSE than M2 buggy (27.90).**
   The NAMM-only eviction policy performs worse with correct attention.
   This is surprising — better attention should help, not hurt.
   Possible explanations:
   - The CMA-ES optimisation landscape is different with correct attention,
     and the same hyperparameters (pop_size=8, sigma=0.065) may not be
     sufficient. The buggy regime's uniform attention may have been easier
     to optimise over (fewer "modes" to learn).
   - HotpotQA-E drops from 39.54 to 14.00 — the biggest single-task
     regression. Under buggy attention, NAMM's late-chunk scoring was
     effectively random, which may have accidentally preserved useful
     distractor-removal behaviour for HotpotQA-E.

2. **M3 maskfix (52.06) substantially EXCEEDS M3 buggy (45.59) and M1 (45.48).**
   Even at only 43% through training, the LoRA + fixed NAMM already
   outperforms both the buggy M3 and the full-context M1 by ~6.5 points.
   The biggest gains are on multi-hop tasks:
   - HotpotQA-E: 74.00 (maskfix) vs 59.67 (buggy) — +14.3 points
   - 2WikiMQA-E: 75.64 vs 56.15 — +19.5 points
   - 2WikiMQA: 63.06 vs 51.11 — +12.0 points

3. **Correct attention primarily helps the LoRA, not the NAMM policy.**
   M2 (NAMM-only) gets worse while M3 (LoRA+NAMM) gets much better.
   This suggests the LoRA adapter is the main beneficiary of correct
   prefill attention — it can now properly cross-reference question
   tokens with context during training, producing better gradient signal.

4. **The multi-hop tasks benefit most from the fix.** HotpotQA-E and
   2WikiMQA variants see the largest gains. These require comparing
   information across distant passages — exactly the kind of reasoning
   that requires functioning cross-attention during prefill.

## Figures

| File | Description |
| ---- | ----------- |
| `m2_buggy_vs_maskfix.png` | M2 mean val F1 over CMA-ES iterations |
| `m3_buggy_vs_maskfix.png` | M3 mean val F1 over training steps |
| `m2_per_task_buggy_vs_maskfix.png` | M2 per-task val F1 comparison |
| `m3_per_task_buggy_vs_maskfix.png` | M3 per-task val F1 comparison |
| `best_val_comparison.png` | Bar chart of best val F1 across all conditions |
"""

    with open(OUT_DIR / "_report.md", "w") as f:
        f.write(report)
    logger.info("Saved _report.md")

    logger.info("Done.")


if __name__ == "__main__":
    main()
