"""
P1: Bootstrap confidence intervals for key condition comparisons.

Paired bootstrap on per-sample F1 deltas (extended_test, n=224).
Reports both micro (per-sample) and macro (stratified per-task) CIs.

Key comparisons:
  1. M4/cs1024 vs Trunc/lora_1024 — overall
  2. M4/cs1024 vs Trunc/lora_1024 — genuinely-hard stratum only
  3. M4/cs2048 vs Trunc/lora_2048 — overall (is NAMM significantly worse?)
  4. M4/cs1024 vs M4/cs2048 — is the budget inversion significant?

Note: "M4" in file paths corresponds to "M3" in the paper (LoRA + frozen NAMM).
      True M4 (joint training) was not evaluated separately here.

Output: analysis_giacomo/report_p1_bootstrap/
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

COND_RESULTS = {
    "M4/cs1024":        RESULTS / "M4" / "cs1024" / "results.json",
    "M4/cs2048":        RESULTS / "M4" / "cs2048" / "results.json",
    "M1":               RESULTS / "M1" / "results.json",
    "M2/cs1024":        RESULTS / "M2" / "cs1024" / "results.json",
    "M2/cs2048":        RESULTS / "M2" / "cs2048" / "results.json",
    "B0":               RESULTS / "B0" / "results.json",
    "Trunc/lora_1024":  RESULTS / "Trunc" / "lora_m1_1024" / "results.json",
    "Trunc/lora_2048":  RESULTS / "Trunc" / "lora_m1_2048" / "results.json",
    "Trunc/plain_1024": RESULTS / "Trunc" / "plain_1024" / "results.json",
    "Trunc/plain_2048": RESULTS / "Trunc" / "plain_2048" / "results.json",
    "A4/cs1024":        RESULTS / "A4" / "cs1024_no_namm" / "results.json",
    "A4/cs2048":        RESULTS / "A4" / "cs2048_no_namm" / "results.json",
    "B1/cs1024":        RESULTS / "B1" / "cs1024" / "results.json",
    "B1/cs2048":        RESULTS / "B1" / "cs2048" / "results.json",
}

DIFFICULTY_CSV = (
    ROOT / "analysis_giacomo" / "report_benchmark_difficulty"
    / "benchmark_difficulty_extended_test.csv"
)

N_BOOT = 10_000
SEED = 42
ALPHA = 0.05  # 95% CI


# ── data loading ──────────────────────────────────────────────────

def load_per_prompt_f1(split: str = "extended_test") -> dict[str, dict[tuple, float]]:
    """Load per-prompt F1 for all conditions.
    Returns {cond: {(task, prompt_idx_str): f1_proportion}}.
    """
    data = {}
    for cond, path in COND_RESULTS.items():
        if not path.exists():
            logger.warning("Missing results: %s", path)
            continue
        with open(path) as f:
            raw = json.load(f)
        # Handle both formats: scores_per_split (most) vs results (B0)
        if "scores_per_split" in raw:
            split_data = raw["scores_per_split"][split]
        elif "results" in raw:
            split_data = raw["results"][split]
        else:
            logger.warning("Unknown format: %s", path)
            continue
        ppf = split_data["per_prompt_f1"]
        cond_data = {}
        for task in TASKS:
            if task in ppf:
                for idx, f1 in ppf[task].items():
                    cond_data[(task, idx)] = f1
        data[cond] = cond_data
    return data


def load_difficulty_labels() -> dict[tuple, str]:
    """Load difficulty classification from existing CSV.
    Returns {(task_short, prompt_idx_str): difficulty_simple}.
    """
    df = pd.read_csv(DIFFICULTY_CSV)
    labels = {}
    # CSV uses short task names; we need (task_short, str(prompt_idx))
    for _, row in df.iterrows():
        labels[(row["task"], str(row["prompt_idx"]))] = row["difficulty_simple"]
    return labels


# ── bootstrap functions ───────────────────────────────────────────

def paired_bootstrap_micro(
    delta: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = ALPHA,
) -> dict:
    """Per-sample paired bootstrap (micro-level CI)."""
    rng = np.random.default_rng(seed)
    n = len(delta)
    observed = delta.mean()

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = delta[idx].mean()

    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    # Two-sided p-value
    if observed >= 0:
        p = (boot_means <= 0).mean()
    else:
        p = (boot_means >= 0).mean()
    p = min(2 * p, 1.0)

    return {
        "observed": observed, "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p, "n": n, "boot_means": boot_means,
    }


def paired_bootstrap_macro(
    delta: np.ndarray,
    task_arr: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> dict:
    """Stratified bootstrap: resample within each task independently,
    compute per-task mean delta, then average across tasks (= macro)."""
    rng = np.random.default_rng(seed)
    unique_tasks = sorted(set(task_arr))

    # Per-task arrays
    task_deltas = {t: delta[task_arr == t] for t in unique_tasks}
    observed = np.mean([d.mean() for d in task_deltas.values()])

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        task_means = []
        for t in unique_tasks:
            d = task_deltas[t]
            idx = rng.integers(0, len(d), size=len(d))
            task_means.append(d[idx].mean())
        boot_means[i] = np.mean(task_means)

    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    if observed >= 0:
        p = (boot_means <= 0).mean()
    else:
        p = (boot_means >= 0).mean()
    p = min(2 * p, 1.0)

    return {
        "observed": observed, "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p, "n": len(delta), "boot_means": boot_means,
    }


# ── comparison runner ─────────────────────────────────────────────

def run_comparison(
    name: str,
    f1_data: dict,
    cond_a: str,
    cond_b: str,
    subset_keys: set | None = None,
) -> dict:
    """Align samples between two conditions, compute deltas, run bootstrap.
    Delta = cond_a - cond_b (positive = cond_a better).
    F1 values converted to percentage points (×100).
    """
    a_data = f1_data[cond_a]
    b_data = f1_data[cond_b]
    shared_keys = sorted(set(a_data) & set(b_data))

    if subset_keys is not None:
        # subset_keys uses (task_short, idx); a_data uses (task_full, idx)
        short_to_full = {v: k for k, v in TASK_SHORT.items()}
        # Try both formats
        filtered = []
        for key in shared_keys:
            task_full, idx = key
            task_short = TASK_SHORT.get(task_full, task_full)
            if (task_short, idx) in subset_keys or (task_short, str(idx)) in subset_keys:
                filtered.append(key)
        shared_keys = filtered

    if not shared_keys:
        logger.warning("No shared samples for %s", name)
        return {"name": name, "n": 0}

    delta_list = []
    task_list = []
    for key in shared_keys:
        d = (a_data[key] - b_data[key]) * 100  # convert to F1 percentage points
        delta_list.append(d)
        task_list.append(key[0])

    delta = np.array(delta_list)
    task_arr = np.array(task_list)

    micro = paired_bootstrap_micro(delta)
    macro = paired_bootstrap_macro(delta, task_arr)

    # Per-task breakdown
    per_task = {}
    for t in sorted(set(task_list)):
        mask = task_arr == t
        t_delta = delta[mask]
        per_task[TASK_SHORT[t]] = {
            "n": int(mask.sum()),
            "mean_delta": float(t_delta.mean()),
            "a_wins": int((t_delta > 0.5).sum()),
            "b_wins": int((t_delta < -0.5).sum()),
            "ties": int(((t_delta >= -0.5) & (t_delta <= 0.5)).sum()),
        }

    return {
        "name": name,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n": len(delta),
        "micro": {k: v for k, v in micro.items() if k != "boot_means"},
        "macro": {k: v for k, v in macro.items() if k != "boot_means"},
        "micro_boot": micro["boot_means"],
        "macro_boot": macro["boot_means"],
        "per_task": per_task,
    }


# ── output ────────────────────────────────────────────────────────

def print_result(r: dict) -> str:
    """Format one comparison result as text."""
    lines = []
    lines.append(f"\n{'─'*70}")
    lines.append(f"  {r['name']}")
    lines.append(f"  {r['cond_a']} minus {r['cond_b']}, n={r['n']}")
    lines.append(f"{'─'*70}")

    for mode in ["micro", "macro"]:
        m = r[mode]
        sig = "***" if m["p_value"] < 0.001 else "**" if m["p_value"] < 0.01 else "*" if m["p_value"] < 0.05 else "ns"
        ci_str = f"[{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]"
        excludes_zero = "YES" if (m["ci_low"] > 0 or m["ci_high"] < 0) else "no"
        lines.append(
            f"  {mode:>5}: delta={m['observed']:+.2f} F1  "
            f"95% CI {ci_str}  p={m['p_value']:.4f} {sig}  "
            f"excludes 0: {excludes_zero}"
        )

    lines.append(f"\n  Per-task breakdown:")
    lines.append(f"  {'Task':<15} {'n':>4} {'delta':>8} {'A wins':>7} {'B wins':>7} {'Ties':>5}")
    for task, info in r["per_task"].items():
        lines.append(
            f"  {task:<15} {info['n']:4d} {info['mean_delta']:+8.2f} "
            f"{info['a_wins']:7d} {info['b_wins']:7d} {info['ties']:5d}"
        )

    text = "\n".join(lines)
    print(text)
    return text


def forest_plot(results: list[dict], split: str) -> None:
    """Forest plot of all comparisons (micro and macro CIs)."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 1.2)))

    y_labels = []
    y_pos = []
    pos = 0

    for r in results:
        if r["n"] == 0:
            continue
        for mode, color in [("macro", "#e74c3c"), ("micro", "#3498db")]:
            m = r[mode]
            ax.errorbar(
                m["observed"],
                pos,
                xerr=[[m["observed"] - m["ci_low"]], [m["ci_high"] - m["observed"]]],
                fmt="o" if mode == "macro" else "s",
                color=color,
                capsize=4,
                markersize=7,
                label=mode if pos < 2 else None,
            )
            y_labels.append(f"{r['name']} ({mode})")
            y_pos.append(pos)
            pos += 1
        pos += 0.5  # gap between comparisons

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("F1 delta (percentage points)")
    ax.set_title(f"Bootstrap 95% CIs — paired comparisons ({split})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / f"forest_plot_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved forest_plot_{split}.png")


def boot_histograms(results: list[dict], split: str) -> None:
    """Histogram of bootstrap distributions for each comparison."""
    n_plots = sum(1 for r in results if r["n"] > 0)
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    row = 0
    for r in results:
        if r["n"] == 0:
            continue
        for col, mode in enumerate(["micro", "macro"]):
            ax = axes[row, col]
            boot = r[f"{mode}_boot"]
            m = r[mode]
            ax.hist(boot, bins=80, color="#3498db" if mode == "micro" else "#e74c3c",
                    alpha=0.7, edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="black", linewidth=1, linestyle="--")
            ax.axvline(m["observed"], color="darkred", linewidth=1.5, linestyle="-")
            ax.axvline(m["ci_low"], color="grey", linewidth=1, linestyle=":")
            ax.axvline(m["ci_high"], color="grey", linewidth=1, linestyle=":")
            ax.set_title(f"{r['name']} ({mode})", fontsize=8)
            ax.set_xlabel("F1 delta", fontsize=7)
            ax.tick_params(labelsize=7)
        row += 1

    plt.suptitle(f"Bootstrap distributions ({split}, n_boot={N_BOOT})", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / f"boot_distributions_{split}.png", dpi=150)
    plt.close()
    print(f"  Saved boot_distributions_{split}.png")


# ── main ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("P1: Bootstrap confidence intervals")
    print("=" * 70)

    # Load difficulty labels for subsetting
    diff_labels = load_difficulty_labels()
    genuinely_hard = {k for k, v in diff_labels.items() if v == "genuinely-hard"}
    print(f"  Genuinely-hard samples (extended_test): {len(genuinely_hard)}")

    all_text = []

    for split in ["extended_test", "test"]:
        print(f"\n{'='*70}")
        print(f"  Split: {split}")
        print(f"{'='*70}")

        f1_data = load_per_prompt_f1(split)

        # Use extended_test difficulty labels for subsetting;
        # for test split, recompute from data
        if split == "extended_test":
            gh_keys = genuinely_hard
        else:
            # Recompute for test split
            all_keys = set()
            for c in f1_data.values():
                all_keys.update(c.keys())
            gh_keys = set()
            for key in all_keys:
                task_full, idx = key
                task_short = TASK_SHORT.get(task_full, task_full)
                tp1 = f1_data.get("Trunc/plain_1024", {}).get(key, -1)
                tp2 = f1_data.get("Trunc/plain_2048", {}).get(key, -1)
                tl1 = f1_data.get("Trunc/lora_1024", {}).get(key, -1)
                m1 = f1_data.get("M1", {}).get(key, -1)
                m4_1 = f1_data.get("M4/cs1024", {}).get(key, -1)
                b0 = f1_data.get("B0", {}).get(key, -1)
                any_s = any(v >= 0.3 for v in [tp1, tp2, tl1, m1, m4_1, b0] if v >= 0)
                if tp1 < 0.3 and not (tp2 >= 0.3 or tl1 >= 0.3) and any_s:
                    gh_keys.add((task_short, str(idx)))

        comparisons = [
            ("M4/cs1024 vs Trunc/lora_1024 (overall)", "M4/cs1024", "Trunc/lora_1024", None),
            ("M4/cs1024 vs Trunc/lora_1024 (genuinely-hard)", "M4/cs1024", "Trunc/lora_1024", gh_keys),
            ("M4/cs2048 vs Trunc/lora_2048 (overall)", "M4/cs2048", "Trunc/lora_2048", None),
            ("M4/cs1024 vs M4/cs2048 (budget inversion)", "M4/cs1024", "M4/cs2048", None),
            ("M1 vs M4/cs1024 (full-context headroom)", "M1", "M4/cs1024", None),
            ("M4/cs1024 vs M2/cs1024 (LoRA contribution)", "M4/cs1024", "M2/cs1024", None),
        ]

        results = []
        for name, a, b, subset in comparisons:
            r = run_comparison(name, f1_data, a, b, subset)
            all_text.append(print_result(r))
            results.append(r)

        forest_plot(results, split)
        boot_histograms(results, split)

    # Save summary CSV
    rows = []
    for split in ["extended_test", "test"]:
        f1_data = load_per_prompt_f1(split)
        if split == "extended_test":
            gh_keys = genuinely_hard
        else:
            gh_keys = None  # skip for test

        comparisons = [
            ("M4/cs1024 vs Trunc/lora_1024 (overall)", "M4/cs1024", "Trunc/lora_1024", None),
            ("M4/cs1024 vs Trunc/lora_1024 (genuinely-hard)", "M4/cs1024", "Trunc/lora_1024", gh_keys),
            ("M4/cs2048 vs Trunc/lora_2048 (overall)", "M4/cs2048", "Trunc/lora_2048", None),
            ("M4/cs1024 vs M4/cs2048 (budget inversion)", "M4/cs1024", "M4/cs2048", None),
            ("M1 vs M4/cs1024 (full-context headroom)", "M1", "M4/cs1024", None),
            ("M4/cs1024 vs M2/cs1024 (LoRA contribution)", "M4/cs1024", "M2/cs1024", None),
        ]
        for name, a, b, subset in comparisons:
            if subset is None or split == "extended_test":
                r = run_comparison(name, f1_data, a, b, subset)
                if r["n"] == 0:
                    continue
                for mode in ["micro", "macro"]:
                    m = r[mode]
                    rows.append({
                        "split": split,
                        "comparison": name,
                        "mode": mode,
                        "n": r["n"],
                        "observed_delta": round(m["observed"], 3),
                        "ci_low": round(m["ci_low"], 3),
                        "ci_high": round(m["ci_high"], 3),
                        "p_value": round(m["p_value"], 5),
                        "significant_05": m["p_value"] < 0.05,
                        "ci_excludes_zero": m["ci_low"] > 0 or m["ci_high"] < 0,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bootstrap_results.csv", index=False)
    print(f"\n  Saved bootstrap_results.csv ({len(df)} rows)")

    # Save text report
    with open(OUT / "bootstrap_report.txt", "w") as f:
        f.write("\n".join(all_text))
    print(f"  Saved bootstrap_report.txt")

    print("\n" + "=" * 70)
    print("P1 complete.")
    print(f"All outputs: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
