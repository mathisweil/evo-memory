"""Plot M2 (NAMM-only) per-task validation F1 over CMA-ES iterations.

One subplot per task, each showing M2 cs1024/cs2048/cs3072 curves
with B0 and M1 per-task baselines as horizontal reference lines.

Produces: results/M2/per_task_val_f1.png

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_m2_per_task_val.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_LABELS = {
    "qasper": "Qasper",
    "2wikimqa": "2WikiMQA",
    "qasper_e": "Qasper-E",
    "hotpotqa_e": "HotpotQA-E",
    "2wikimqa_e": "2WikiMQA-E",
}

# M2 NAMM runs — buggy (from experiment_specification.md)
M2_RUNS = {
    "1024": ["lenhmfb1"],
    "2048": ["y5fdw0f9", "ccflnsds"],
    "3072": ["quc95irz"],
}

# M2 NAMM runs — maskfix (corrected attention mask)
M2_MASKFIX_RUNS = {
    "1024": ["z5bo4n8k"],
    # "2048": ["jip3a3dm"],  # TODO: uncomment when M2 maskfix cs2048 finishes
}

# M1 segments (for per-task baselines)
M1_SEGMENTS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]
B0_RUN = "kz6vqo2o"  # baseline logged at start of M1

CS_COLORS = {"1024": "#1f77b4", "2048": "#ff7f0e", "3072": "#2ca02c"}


def get_api():
    return wandb.Api()


def fetch_namm_per_task(api, run_ids):
    """Fetch per-task val F1 from NAMM runs."""
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        keys = ["iter"] + [f"val_lb/{t}" for t in TASKS]
        h = r.history(keys=keys, pandas=True, samples=10000)
        if not h.empty:
            if frames and len(h) > 0:
                prev_max = frames[-1]["iter"].max()
                h = h[h["iter"] > prev_max]
            frames.append(h)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["iter"])
    return df


def fetch_b0_per_task(api):
    """Get B0 baseline per-task F1."""
    r = api.run(f"{ENTITY}/{PROJECT}/{B0_RUN}")
    keys = [f"lora/baseline_lb_{t}" for t in TASKS]
    h = r.history(keys=keys, pandas=True, samples=10000)
    h = h.dropna(subset=[f"lora/baseline_lb_{TASKS[0]}"])
    if h.empty:
        return {}
    row = h.iloc[0]
    return {t: row[f"lora/baseline_lb_{t}"] for t in TASKS}


def fetch_m1_best_per_task(api):
    """Get M1 best val F1 per task."""
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
        row_avg = h.loc[idx, "lora/val_lb_avg_f1"]
        if row_avg > best_avg:
            best_avg = row_avg
            best_row = {t: h.loc[idx, f"lora/val_lb_{t}"] for t in TASKS}
    return best_row or {}


def main():
    api = get_api()

    print("Fetching B0 baselines...")
    b0 = fetch_b0_per_task(api)
    print(f"  B0: {b0}")

    print("Fetching M1 best per-task...")
    m1 = fetch_m1_best_per_task(api)
    print(f"  M1: {m1}")

    print("Fetching M2 buggy data...")
    m2_data = {}
    for cs, rids in M2_RUNS.items():
        print(f"  M2 cs{cs}...")
        m2_data[cs] = fetch_namm_per_task(api, rids)
        print(f"    {len(m2_data[cs])} rows")

    print("Fetching M2 maskfix data...")
    m2_maskfix_data = {}
    for cs, rids in M2_MASKFIX_RUNS.items():
        print(f"  M2 maskfix cs{cs}...")
        m2_maskfix_data[cs] = fetch_namm_per_task(api, rids)
        print(f"    {len(m2_maskfix_data[cs])} rows")

    # Plot: one subplot per task
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes[i]
        val_key = f"val_lb/{task}"

        for cs in sorted(m2_data.keys()):
            df = m2_data[cs]
            sub = df.dropna(subset=[val_key])
            if sub.empty:
                continue
            ax.plot(
                sub["iter"], sub[val_key],
                label=f"M2 cs={cs} (buggy)",
                color=CS_COLORS[cs],
                linewidth=1.5, alpha=0.4, linestyle="--",
            )

        # Maskfix curves (solid, prominent)
        for cs in sorted(m2_maskfix_data.keys()):
            df = m2_maskfix_data[cs]
            sub = df.dropna(subset=[val_key])
            if sub.empty:
                continue
            ax.plot(
                sub["iter"], sub[val_key],
                label=f"M2 cs={cs} (maskfix)",
                color=CS_COLORS[cs],
                linewidth=2.0, alpha=0.9,
            )

        # B0 baseline
        if task in b0:
            ax.axhline(
                b0[task], color="#7f7f7f", linestyle="--", linewidth=1.5,
                label=f"B0 ({b0[task]:.1f})",
            )

        # M1 best
        if task in m1:
            ax.axhline(
                m1[task], color="#d62728", linestyle="--", linewidth=1.5,
                label=f"M1 ({m1[task]:.1f})",
            )

        ax.set_title(TASK_LABELS[task], fontsize=13, fontweight="bold")
        ax.set_xlabel("CMA-ES Iteration", fontsize=10)
        ax.set_ylabel("Val F1", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)

    # Hide the 6th subplot
    axes[5].set_visible(False)

    fig.suptitle(
        "M2 (Standalone NAMM) — Per-Task Validation F1 Over Training",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = "results/M2/per_task_val_f1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
