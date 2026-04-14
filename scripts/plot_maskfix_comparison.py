"""Plot maskfix vs buggy comparison for M2 and M3 at cs1024.

Generates:
  results/M2/comparison_val_f1.png  -- M2 aggregate val F1 over training
  results/M3/comparison_val_f1.png  -- M3 aggregate val F1 over training
  results/M3/comparison_loss.png    -- M3 training loss over steps

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_maskfix_comparison.py
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"

# Run IDs
M1_SEGMENTS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]
M2_BUGGY_1024 = ["lenhmfb1"]
M2_MASKFIX_1024 = ["z5bo4n8k"]
# M2_MASKFIX_2048 = ["jip3a3dm"]  # TODO: uncomment when finished
M3_BUGGY_1024 = ["ovosogkj"]
M3_MASKFIX_1024 = ["h0bzg6on"]
# M3_MASKFIX_2048 = ["<TBD>"]  # TODO: add when M3 maskfix cs2048 starts


def get_api():
    return wandb.Api()


def fetch_namm(api, run_ids, keys):
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        h = r.history(keys=keys, pandas=True, samples=10000)
        if not h.empty:
            if frames:
                prev_max = frames[-1][keys[0]].max()
                h = h[h[keys[0]] > prev_max]
            frames.append(h)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=[keys[0]])


def fetch_lora(api, run_ids, keys):
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        h = r.history(keys=keys, pandas=True, samples=10000)
        if not h.empty:
            if frames:
                prev_max = frames[-1][keys[0]].max()
                h = h[h[keys[0]] > prev_max]
            frames.append(h)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[keys[0]])
    df = df.drop_duplicates(subset=[keys[0]], keep="first")
    return df.sort_values(keys[0]).reset_index(drop=True)


def smooth(s, w=5):
    return s.rolling(window=w, min_periods=1, center=True).mean()


def plot_m2_val_f1(api):
    print("Plotting M2 comparison_val_f1...")
    keys = ["iter", "val_lb/mean_f1"]
    buggy = fetch_namm(api, M2_BUGGY_1024, keys)
    maskfix = fetch_namm(api, M2_MASKFIX_1024, keys)

    fig, ax = plt.subplots(figsize=(10, 6))
    if not buggy.empty:
        ax.plot(buggy["iter"], buggy["val_lb/mean_f1"],
                color="#1f77b4", alpha=0.5, linestyle="--",
                linewidth=1.5, label="M2 cs1024 (buggy)")
    if not maskfix.empty:
        ax.plot(maskfix["iter"], maskfix["val_lb/mean_f1"],
                color="#1f77b4", linewidth=2.5,
                label="M2 cs1024 (maskfix)")

    ax.set_xlabel("CMA-ES Iteration", fontsize=12)
    ax.set_ylabel("Val Mean F1", fontsize=12)
    ax.set_title("M2 Standalone NAMM -- Validation F1 (cs1024)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/M2/comparison_val_f1.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/M2/comparison_val_f1.png")


def plot_m3_val_f1(api):
    print("Plotting M3 comparison_val_f1...")
    keys = ["lora/global_step", "lora/val_lb_avg_f1"]
    m1 = fetch_lora(api, M1_SEGMENTS, keys)
    buggy = fetch_lora(api, M3_BUGGY_1024, keys)
    maskfix = fetch_lora(api, M3_MASKFIX_1024, keys)

    fig, ax = plt.subplots(figsize=(10, 6))
    k = "lora/val_lb_avg_f1"
    s = "lora/global_step"

    if not m1.empty:
        ax.plot(m1[s], smooth(m1[k]),
                color="#d62728", linewidth=2, label="M1 (no NAMM)")
    if not buggy.empty:
        ax.plot(buggy[s], smooth(buggy[k]),
                color="#1f77b4", alpha=0.5, linestyle="--",
                linewidth=1.5, label="M3 cs1024 (buggy)")
    if not maskfix.empty:
        ax.plot(maskfix[s], smooth(maskfix[k]),
                color="#9467bd", linewidth=2.5,
                label="M3 cs1024 (maskfix)")

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Val Avg F1", fontsize=12)
    ax.set_title(
        "M3 (LoRA + Frozen NAMM) vs M1 -- Validation F1 (cs1024)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/M3/comparison_val_f1.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/M3/comparison_val_f1.png")


def plot_m3_loss(api):
    print("Plotting M3 comparison_loss...")
    keys = ["lora/global_step", "lora/train_loss"]
    m1 = fetch_lora(api, M1_SEGMENTS, keys)
    buggy = fetch_lora(api, M3_BUGGY_1024, keys)
    maskfix = fetch_lora(api, M3_MASKFIX_1024, keys)

    fig, ax = plt.subplots(figsize=(10, 6))
    k = "lora/train_loss"
    s = "lora/global_step"

    if not m1.empty:
        ax.plot(m1[s], smooth(m1[k], 10),
                color="#d62728", linewidth=2, label="M1 (no NAMM)")
    if not buggy.empty:
        ax.plot(buggy[s], smooth(buggy[k], 10),
                color="#1f77b4", alpha=0.5, linestyle="--",
                linewidth=1.5, label="M3 cs1024 (buggy)")
    if not maskfix.empty:
        ax.plot(maskfix[s], smooth(maskfix[k], 10),
                color="#9467bd", linewidth=2.5,
                label="M3 cs1024 (maskfix)")

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(
        "M3 (LoRA + Frozen NAMM) vs M1 -- Training Loss (cs1024)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/M3/comparison_loss.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/M3/comparison_loss.png")


def plot_m2_cs1024_val(api):
    """Regenerate results/M2/cs1024/val_f1.png with maskfix overlay."""
    print("Plotting M2/cs1024/val_f1...")
    keys = ["iter", "val_lb/mean_f1"]
    buggy = fetch_namm(api, M2_BUGGY_1024, keys)
    maskfix = fetch_namm(api, M2_MASKFIX_1024, keys)

    fig, ax = plt.subplots(figsize=(10, 6))
    if not buggy.empty:
        ax.plot(buggy["iter"], buggy["val_lb/mean_f1"],
                color="#1f77b4", alpha=0.4, linestyle="--",
                linewidth=1.5, label="buggy (lenhmfb1)")
    if not maskfix.empty:
        ax.plot(maskfix["iter"], maskfix["val_lb/mean_f1"],
                color="#1f77b4", linewidth=2.5,
                label="maskfix (z5bo4n8k)")

    ax.set_xlabel("CMA-ES Iteration", fontsize=12)
    ax.set_ylabel("Val Mean F1", fontsize=12)
    ax.set_title("M2 cs1024 -- Validation F1", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/M2/cs1024/val_f1.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/M2/cs1024/val_f1.png")


def plot_m3_cs1024_val(api):
    """Regenerate results/M3/cs1024/val_f1.png with maskfix overlay."""
    print("Plotting M3/cs1024/val_f1...")
    keys = ["lora/global_step", "lora/val_lb_avg_f1"]
    m1 = fetch_lora(api, M1_SEGMENTS, keys)
    buggy = fetch_lora(api, M3_BUGGY_1024, keys)
    maskfix = fetch_lora(api, M3_MASKFIX_1024, keys)

    fig, ax = plt.subplots(figsize=(10, 6))
    k = "lora/val_lb_avg_f1"
    s = "lora/global_step"

    if not m1.empty:
        ax.plot(m1[s], smooth(m1[k]),
                color="#d62728", linewidth=2, alpha=0.7,
                label="M1 (no NAMM)")
    if not buggy.empty:
        ax.plot(buggy[s], smooth(buggy[k]),
                color="#1f77b4", alpha=0.4, linestyle="--",
                linewidth=1.5, label="buggy (ovosogkj)")
    if not maskfix.empty:
        ax.plot(maskfix[s], maskfix[k],
                color="#9467bd", alpha=0.15, linewidth=1)
        ax.plot(maskfix[s], smooth(maskfix[k]),
                color="#9467bd", linewidth=2.5,
                label="maskfix (h0bzg6on)")

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Val Avg F1", fontsize=12)
    ax.set_title("M3 cs1024 -- Validation F1", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/M3/cs1024/val_f1.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/M3/cs1024/val_f1.png")


def plot_m3_cs1024_loss(api):
    """Regenerate results/M3/cs1024/loss.png with maskfix overlay."""
    print("Plotting M3/cs1024/loss...")
    keys = ["lora/global_step", "lora/train_loss"]
    m1 = fetch_lora(api, M1_SEGMENTS, keys)
    buggy = fetch_lora(api, M3_BUGGY_1024, keys)
    maskfix = fetch_lora(api, M3_MASKFIX_1024, keys)

    fig, ax = plt.subplots(figsize=(10, 6))
    k = "lora/train_loss"
    s = "lora/global_step"

    if not m1.empty:
        ax.plot(m1[s], smooth(m1[k], 10),
                color="#d62728", linewidth=2, alpha=0.7,
                label="M1 (no NAMM)")
    if not buggy.empty:
        ax.plot(buggy[s], smooth(buggy[k], 10),
                color="#1f77b4", alpha=0.4, linestyle="--",
                linewidth=1.5, label="buggy (ovosogkj)")
    if not maskfix.empty:
        ax.plot(maskfix[s], smooth(maskfix[k], 10),
                color="#9467bd", linewidth=2.5,
                label="maskfix (h0bzg6on)")

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("M3 cs1024 -- Training Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/M3/cs1024/loss.png", dpi=150)
    plt.close(fig)
    print("  Saved: results/M3/cs1024/loss.png")


def main():
    api = get_api()
    # Top-level comparison plots
    plot_m2_val_f1(api)
    plot_m3_val_f1(api)
    plot_m3_loss(api)
    # Per-cache-size subdir plots (cs1024 only)
    plot_m2_cs1024_val(api)
    plot_m3_cs1024_val(api)
    plot_m3_cs1024_loss(api)
    print("\nDone.")


if __name__ == "__main__":
    main()
