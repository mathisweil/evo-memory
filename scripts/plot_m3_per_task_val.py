"""Plot M3 (LoRA + frozen NAMM) per-task validation F1 over training steps.

One subplot per task, each showing M3 cs1024/cs2048/cs3072 and M1 curves.

Produces: results/M3/per_task_val_f1.png

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_m3_per_task_val.py
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

# M1 LoRA segments
M1_SEGMENTS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]

# M3 (LoRA + frozen NAMM) runs
M3_RUNS = {
    "1024": ["ovosogkj"],
    "2048": ["m4knrhmr"],
    "3072": ["4sgkswa6"],
}

CS_COLORS = {"1024": "#1f77b4", "2048": "#ff7f0e", "3072": "#2ca02c"}
M1_COLOR = "#d62728"

SMOOTH_WINDOW = 5


def get_api():
    return wandb.Api()


def smooth(series, window=SMOOTH_WINDOW):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def fetch_lora_history(api, run_ids):
    """Fetch per-task val F1 from LoRA runs."""
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        keys = ["lora/global_step"] + [f"lora/val_lb_{t}" for t in TASKS]
        h = r.history(keys=keys, pandas=True, samples=10000)
        if not h.empty:
            if frames and len(h) > 0:
                prev_max = frames[-1]["lora/global_step"].max()
                h = h[h["lora/global_step"] > prev_max]
            frames.append(h)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["lora/global_step"])
    df = df.drop_duplicates(subset=["lora/global_step"], keep="first")
    df = df.sort_values("lora/global_step").reset_index(drop=True)
    return df


def main():
    api = get_api()

    print("Fetching M1 data...")
    m1_df = fetch_lora_history(api, M1_SEGMENTS)
    print(f"  M1: {len(m1_df)} rows, steps {m1_df['lora/global_step'].min():.0f}-{m1_df['lora/global_step'].max():.0f}")

    m3_data = {}
    for cs, rids in M3_RUNS.items():
        print(f"Fetching M3 cs{cs}...")
        m3_data[cs] = fetch_lora_history(api, rids)
        df = m3_data[cs]
        print(f"  M3 cs{cs}: {len(df)} rows, steps {df['lora/global_step'].min():.0f}-{df['lora/global_step'].max():.0f}")

    # Plot: one subplot per task
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes[i]
        val_key = f"lora/val_lb_{task}"

        # M1
        m1_sub = m1_df.dropna(subset=[val_key])
        if not m1_sub.empty:
            # Raw as faint
            ax.plot(
                m1_sub["lora/global_step"], m1_sub[val_key],
                color=M1_COLOR, alpha=0.15, linewidth=1,
            )
            # Smoothed
            ax.plot(
                m1_sub["lora/global_step"], smooth(m1_sub[val_key]),
                color=M1_COLOR, linewidth=2, label="M1 (no NAMM)",
            )

        # M3 per cache size
        for cs in sorted(m3_data.keys()):
            df = m3_data[cs]
            sub = df.dropna(subset=[val_key])
            if sub.empty:
                continue
            # Raw as faint
            ax.plot(
                sub["lora/global_step"], sub[val_key],
                color=CS_COLORS[cs], alpha=0.15, linewidth=1,
            )
            # Smoothed
            ax.plot(
                sub["lora/global_step"], smooth(sub[val_key]),
                color=CS_COLORS[cs], linewidth=2,
                label=f"M3 cs={cs}",
            )

        ax.set_title(TASK_LABELS[task], fontsize=13, fontweight="bold")
        ax.set_xlabel("Global Step", fontsize=10)
        ax.set_ylabel("Val F1", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide the 6th subplot
    axes[5].set_visible(False)

    fig.suptitle(
        "M3 (LoRA + Frozen NAMM) vs M1 — Per-Task Validation F1 Over Training\n"
        "(smoothed, window=5; faint = raw)",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = "results/M3/per_task_val_f1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
