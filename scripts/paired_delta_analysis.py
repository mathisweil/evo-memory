"""Paired per-prompt Δ-F1 analysis across conditions in main_table_5t.

Reads per_prompt_f1 from results/main_table_5t/<cond>/results.json and
generations.json (for prompt lengths) and produces:

  results/main_table_5t/plots/paired_delta_<A>_vs_<B>_<split>.png
    - panel 1: micro mean F1 bar (A, B) with paired Δ + bootstrap 95% CI
    - panel 2: sorted per-prompt Δ-F1 (waterfall) coloured by task
    - panel 3: F1 vs prompt length scatter, two series

By default runs the comparisons motivated by the LoRA × NAMM puzzle:

    M4/cs1024            vs   A4/cs1024_no_namm   (NAMM-OOD test, cs1024)
    M4/cs2048            vs   A4/cs2048_no_namm   (NAMM-OOD test, cs2048)
    M1                   vs   A4/cs1024_no_namm   (does NAMM training hurt?)
    M1                   vs   A4/cs2048_no_namm   (same, cs2048 LoRA)
    B1/cs1024            vs   M2/cs1024           (recency vs learned NAMM)
    B1/cs2048            vs   M2/cs2048
    M4/cs1024            vs   M4/cs2048           (cs1024 LoRA vs cs2048 LoRA)
    A4/cs1024_no_namm    vs   A4/cs2048_no_namm   (same as above, no NAMM)

All comparisons run on both `test` and `extended_test`.

Usage:
    /cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python \
        scripts/paired_delta_analysis.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message=".*tight_layout.*")

PROJ = Path("/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo")
ROOT = PROJ / "results" / "main_table_5t"
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TASKS = ["lb/qasper", "lb/2wikimqa", "lb/qasper_e",
         "lb/hotpotqa_e", "lb/2wikimqa_e"]
TASK_COLOURS = {
    "lb/qasper": "#1f77b4",
    "lb/2wikimqa": "#ff7f0e",
    "lb/qasper_e": "#2ca02c",
    "lb/hotpotqa_e": "#d62728",
    "lb/2wikimqa_e": "#9467bd",
}

SPLITS = ["test", "extended_test"]

# Default comparisons (label_A, label_B, key_A, key_B, motivation)
COMPARISONS = [
    ("M4/cs1024", "A4/cs1024_no_namm",
     "M4/cs1024", "A4/cs1024_no_namm",
     "NAMM-OOD: same LoRA, with vs without NAMM (cs1024)"),
    ("M4/cs2048", "A4/cs2048_no_namm",
     "M4/cs2048", "A4/cs2048_no_namm",
     "NAMM-OOD: same LoRA, with vs without NAMM (cs2048)"),
    ("M1", "A4/cs1024_no_namm",
     "M1", "A4/cs1024_no_namm",
     "Does NAMM training hurt? Both eval no-NAMM (cs1024)"),
    ("M1", "A4/cs2048_no_namm",
     "M1", "A4/cs2048_no_namm",
     "Does NAMM training hurt? Both eval no-NAMM (cs2048)"),
    ("B1/cs1024", "M2/cs1024",
     "B1/cs1024", "M2/cs1024",
     "Recency vs learned NAMM (cs1024)"),
    ("B1/cs2048", "M2/cs2048",
     "B1/cs2048", "M2/cs2048",
     "Recency vs learned NAMM (cs2048)"),
    ("M4/cs1024", "M4/cs2048",
     "M4/cs1024", "M4/cs2048",
     "M4: cs1024 LoRA vs cs2048 LoRA"),
    ("A4/cs1024_no_namm", "A4/cs2048_no_namm",
     "A4/cs1024_no_namm", "A4/cs2048_no_namm",
     "M4 LoRAs evaluated no-NAMM: cs1024 vs cs2048"),
]


def load_per_prompt_f1(condition_key: str) -> dict:
    """Return {split: {task: {prompt_idx (int): f1 (0-1)}}}."""
    path = ROOT / condition_key / "results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    container = (d.get("results") if d.get("type") == "plain_llama_baseline"
                 else d.get("scores_per_split"))
    out = {}
    for split, scores in container.items():
        ppf1 = scores.get("per_prompt_f1", {})
        out[split] = {
            task: {int(k): float(v) for k, v in task_dict.items()}
            for task, task_dict in ppf1.items()
        }
    return out


def load_lengths(condition_key: str) -> dict:
    """Return {split: {task: {prompt_idx (int): length (int, words)}}}.

    Reads from generations.json — falls back to {} if not present.
    """
    path = ROOT / condition_key / "generations.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for split, by_task in d.items():
        out[split] = {}
        for task, dicts in by_task.items():
            out[split][task] = {
                int(g.get("prompt_idx", -1)): int(g.get("length", 0))
                for g in dicts
                if g.get("prompt_idx", -1) != -1
            }
    return out


def paired_arrays(ppf1_a, ppf1_b, lengths_a, split):
    """Build aligned per-prompt arrays for (A, B) by intersecting indices.

    Returns (idx_array, task_array, fa, fb, length_array).
    Each entry is one prompt seen by both A and B in the same task.
    """
    a = ppf1_a.get(split, {})
    b = ppf1_b.get(split, {})
    lens = lengths_a.get(split, {})
    rows = []
    for task in TASKS:
        ta = a.get(task, {})
        tb = b.get(task, {})
        tl = lens.get(task, {})
        common = sorted(set(ta) & set(tb))
        for idx in common:
            rows.append((idx, task, ta[idx], tb[idx], tl.get(idx, 0)))
    if not rows:
        return None
    return rows


def micro_mean(rows, col):
    return float(np.mean([r[col] for r in rows])) * 100.0


def bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI on the mean of `deltas` (already in % units)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    samples = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def plot_paired_comparison(label_a, label_b, key_a, key_b, motivation,
                           split, out_path):
    ppf1_a = load_per_prompt_f1(key_a)
    ppf1_b = load_per_prompt_f1(key_b)
    lengths_a = load_lengths(key_a)
    if not ppf1_a or not ppf1_b:
        print(f"  SKIP {label_a} vs {label_b} [{split}]: "
              f"missing per_prompt_f1")
        return False

    rows = paired_arrays(ppf1_a, ppf1_b, lengths_a, split)
    if not rows:
        print(f"  SKIP {label_a} vs {label_b} [{split}]: no aligned prompts")
        return False

    n = len(rows)
    fa = np.array([r[2] for r in rows]) * 100  # 0-100 scale
    fb = np.array([r[3] for r in rows]) * 100
    lens = np.array([r[4] for r in rows])
    tasks = [r[1] for r in rows]

    micro_a = float(fa.mean())
    micro_b = float(fb.mean())
    deltas = fb - fa  # B − A; positive = B better
    mean_delta = float(deltas.mean())
    lo, hi = bootstrap_ci(deltas)

    # Per-task breakdown
    by_task_delta = defaultdict(list)
    for t, d in zip(tasks, deltas):
        by_task_delta[t].append(d)
    task_means = {t: float(np.mean(v)) if v else float("nan")
                  for t, v in by_task_delta.items()}

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1.5], wspace=0.3)
    ax_bar = fig.add_subplot(gs[0])
    ax_wf = fig.add_subplot(gs[1])
    ax_sc = fig.add_subplot(gs[2])

    # ── Panel 1: micro mean bars (A, B) with Δ + CI annotation ─────────
    bars = ax_bar.bar([0, 1], [micro_a, micro_b],
                      color=["#7f7f7f", "#1f77b4"],
                      edgecolor="black", linewidth=0.5)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels([label_a, label_b], rotation=20, ha="right")
    ax_bar.set_ylabel("Micro mean F1 (%)")
    ax_bar.set_ylim(0, max(micro_a, micro_b) * 1.25)
    for b, v in zip(bars, [micro_a, micro_b]):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.4,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    sig = "*" if (lo > 0 or hi < 0) else "ns"
    ax_bar.set_title(f"Δ = {mean_delta:+.2f}  "
                     f"[95% CI {lo:+.2f}, {hi:+.2f}] {sig}",
                     fontsize=10)

    # ── Panel 2: sorted per-prompt Δ-F1 waterfall ──────────────────────
    order = np.argsort(deltas)
    ax_wf.bar(np.arange(n), deltas[order],
              color=[TASK_COLOURS[tasks[i]] for i in order],
              width=1.0, edgecolor="none")
    ax_wf.axhline(0, color="black", linewidth=0.5)
    ax_wf.set_xlabel("Prompts (sorted by Δ)")
    ax_wf.set_ylabel(f"Δ F1 ({label_b} − {label_a})")
    ax_wf.set_title("Per-prompt Δ (sorted), coloured by task")
    ax_wf.set_xlim(-0.5, n - 0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=t.replace("lb/", ""))
               for t, c in TASK_COLOURS.items()]
    ax_wf.legend(handles=handles, fontsize=7, loc="upper left",
                 frameon=False, ncol=2)

    # Per-task Δ summary in the corner
    summary_lines = ["Per-task Δ:"]
    for t in TASKS:
        if t in task_means:
            n_t = len(by_task_delta[t])
            summary_lines.append(
                f"  {t.replace('lb/', '')[:10]:<10} "
                f"{task_means[t]:+5.1f}  (n={n_t})"
            )
    ax_wf.text(0.02, -0.40, "\n".join(summary_lines),
               transform=ax_wf.transAxes, fontsize=7, va="top",
               family="monospace", color="#333")

    # ── Panel 3: F1 vs prompt length, two series ───────────────────────
    if lens.max() > 0:
        ax_sc.scatter(lens, fa, color="#7f7f7f", alpha=0.6, s=18,
                      label=label_a, edgecolors="none")
        ax_sc.scatter(lens, fb, color="#1f77b4", alpha=0.6, s=18,
                      label=label_b, edgecolors="none")
        ax_sc.set_xlabel("Prompt length (words from LongBench)")
        ax_sc.set_ylabel("F1 (%)")
        ax_sc.set_title("F1 vs prompt length")
        ax_sc.grid(True, linestyle=":", alpha=0.4)
        ax_sc.legend(fontsize=8, loc="upper right")
    else:
        ax_sc.text(0.5, 0.5, "no length data", transform=ax_sc.transAxes,
                   ha="center", va="center", color="#999")
        ax_sc.set_xticks([])
        ax_sc.set_yticks([])

    fig.suptitle(
        f"{motivation}  —  split={split}  (n={n} prompts, paired)",
        fontsize=11)
    fig.text(0.5, 0.005,
             "Δ panel uses paired prompts only. Micro mean = mean over "
             "prompt-level F1s. Bootstrap CI: percentile, B=2000.",
             ha="center", va="bottom", fontsize=7, color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main():
    written = []
    summary_rows = []
    for label_a, label_b, key_a, key_b, motivation in COMPARISONS:
        for split in SPLITS:
            short = (f"{label_a}_vs_{label_b}_{split}"
                     .replace("/", "-"))
            out_path = PLOTS / f"paired_delta_{short}.png"
            ok = plot_paired_comparison(label_a, label_b, key_a, key_b,
                                        motivation, split, out_path)
            if ok:
                written.append(out_path.name)
                # also collect for stdout summary
                ppf1_a = load_per_prompt_f1(key_a)
                ppf1_b = load_per_prompt_f1(key_b)
                lengths_a = load_lengths(key_a)
                rows = paired_arrays(ppf1_a, ppf1_b, lengths_a, split)
                if rows:
                    fa = np.array([r[2] for r in rows]) * 100
                    fb = np.array([r[3] for r in rows]) * 100
                    delta = float((fb - fa).mean())
                    lo, hi = bootstrap_ci(fb - fa)
                    summary_rows.append(
                        (split, label_a, label_b, len(rows),
                         float(fa.mean()), float(fb.mean()),
                         delta, lo, hi))

    print(f"Wrote {len(written)} paired-delta plots to {PLOTS}/")
    print()
    print("Summary table:")
    print(f"{'split':<14} {'A':<22} {'B':<22} {'n':>4} "
          f"{'micro A':>9} {'micro B':>9} {'delta':>8} "
          f"{'CI lo':>8} {'CI hi':>8}  sig")
    for s in summary_rows:
        sig = "*" if (s[7] > 0 or s[8] < 0) else "ns"
        print(f"{s[0]:<14} {s[1]:<22} {s[2]:<22} {s[3]:>4} "
              f"{s[4]:>9.2f} {s[5]:>9.2f} {s[6]:>+8.2f} "
              f"{s[7]:>+8.2f} {s[8]:>+8.2f}  {sig}")


if __name__ == "__main__":
    main()
